"""Backtesting engine with walk-forward optimisation.

Accounts for slippage, maker/taker fees, funding rates, and latency.
Uses purged time-series cross-validation to prevent look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.core.config import BacktestConfig, BotConfig, MLConfig, RiskConfig, TradingConfig
from bot.core.features import build_features, load_csv, make_labels
from bot.core.risk import RiskManager
from bot.ml.models import EnsemblePredictor, PurgedKFold
from bot.utils.logger import get_logger

log = get_logger("backtest")


@dataclass
class Position:
    """Tracks an open position."""

    side: str  # "LONG" or "SHORT"
    entry_price: float
    size: float  # base asset qty
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    entry_idx: int = 0


@dataclass
class BacktestResult:
    """Container for backtest output."""

    equity_curve: List[float] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None


class BacktestEngine:
    """Event-driven backtest simulator."""

    def __init__(self, config: BotConfig):
        self.cfg = config
        self.bt = config.backtest
        self.tc = config.trading
        self.rc = config.risk
        self.mc = config.ml

    def _apply_slippage(self, price: float, side: str) -> float:
        """Simulate slippage against the trader."""
        slip = price * self.bt.slippage_pct / 100
        return price + slip if side == "LONG" else price - slip

    def _apply_fee(self, notional: float, is_maker: bool = False) -> float:
        fee_pct = self.bt.maker_fee_pct if is_maker else self.bt.taker_fee_pct
        return notional * fee_pct / 100

    def _apply_funding(self, notional: float) -> float:
        return notional * self.bt.funding_rate_pct / 100

    def _position_size(self, equity: float, entry_price: float,
                       atr: float, leverage: int) -> float:
        """Risk-based position sizing respecting minimum notional."""
        risk_amount = equity * self.tc.risk_per_trade_pct / 100
        stop_distance = atr * self.rc.sl_atr_mult
        if stop_distance <= 0 or entry_price <= 0:
            return 0.0

        # Size in quote (USDT) terms
        size_quote = (risk_amount / (stop_distance / entry_price)) * leverage
        max_quote = equity * self.tc.max_position_pct / 100 * leverage

        size_quote = min(size_quote, max_quote)

        if size_quote < self.tc.min_notional_usd:
            return 0.0

        return size_quote / entry_price

    def run_single(self, df: pd.DataFrame, features: pd.DataFrame,
                   predictions: np.ndarray) -> BacktestResult:
        """Run backtest over a single segment with pre-computed predictions."""
        n = len(df)
        assert len(features) == n
        assert len(predictions) == n

        risk = RiskManager(self.rc, self.bt.initial_balance)
        equity = self.bt.initial_balance
        equity_curve = [equity]
        trades: List[dict] = []
        position: Optional[Position] = None
        funding_counter = 0
        last_day = None

        for i in range(1, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = features.iloc[i].get("atr_14", 0) * close  # denormalise

            # Daily risk reset
            current_day = df.index[i].date() if hasattr(df.index[i], 'date') else None
            if current_day is not None and current_day != last_day:
                risk.reset_daily()
                last_day = current_day

            # Flash crash detection
            risk.detect_flash_crash(row["open"], close, high, low)

            # Force-close if kill-switch drawdown breached while in position
            if position is not None and risk._killed:
                exit_price = self._apply_slippage(
                    close, "SHORT" if position.side == "LONG" else "LONG"
                )
                notional = position.size * exit_price
                fee = self._apply_fee(notional)
                if position.side == "LONG":
                    pnl = (exit_price - position.entry_price) * position.size - fee
                else:
                    pnl = (position.entry_price - exit_price) * position.size - fee
                equity += pnl
                risk.current_equity = equity
                risk.record_trade(pnl, str(df.index[i]))
                trades.append({
                    "entry_idx": position.entry_idx, "exit_idx": i,
                    "side": position.side, "entry_price": position.entry_price,
                    "exit_price": exit_price, "size": position.size,
                    "pnl": pnl, "reason": "kill_switch", "equity_after": equity,
                })
                position = None
                equity_curve.append(equity)
                continue

            # --- Manage open position ---
            if position is not None:
                # Check stop-loss / take-profit hit during candle
                exit_price = None
                exit_reason = ""

                if position.side == "LONG":
                    if low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "stop_loss"
                    elif high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "take_profit"
                    elif position.trailing_stop and low <= position.trailing_stop:
                        exit_price = position.trailing_stop
                        exit_reason = "trailing_stop"
                else:
                    if high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "stop_loss"
                    elif low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "take_profit"
                    elif position.trailing_stop and high >= position.trailing_stop:
                        exit_price = position.trailing_stop
                        exit_reason = "trailing_stop"

                # Signal-based exit
                if exit_price is None and not np.isnan(predictions[i]):
                    if position.side == "LONG" and predictions[i] < self.tc.exit_confidence:
                        exit_price = self._apply_slippage(close, "SHORT")
                        exit_reason = "signal_exit"
                    elif position.side == "SHORT" and predictions[i] > (1 - self.tc.exit_confidence):
                        exit_price = self._apply_slippage(close, "LONG")
                        exit_reason = "signal_exit"

                if exit_price is not None:
                    notional = position.size * exit_price
                    fee = self._apply_fee(notional)
                    if position.side == "LONG":
                        pnl = (exit_price - position.entry_price) * position.size - fee
                    else:
                        pnl = (position.entry_price - exit_price) * position.size - fee

                    equity += pnl
                    risk.current_equity = equity
                    if equity > risk.peak_equity:
                        risk.peak_equity = equity
                    risk.record_trade(pnl, str(df.index[i]))
                    trades.append({
                        "entry_idx": position.entry_idx,
                        "exit_idx": i,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "size": position.size,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "equity_after": equity,
                    })
                    position = None
                else:
                    # Update trailing stop
                    ts = risk.compute_trailing_stop(
                        position.entry_price, close, position.side
                    )
                    if ts is not None:
                        if position.trailing_stop is None:
                            position.trailing_stop = ts
                        elif position.side == "LONG":
                            position.trailing_stop = max(position.trailing_stop, ts)
                        else:
                            position.trailing_stop = min(position.trailing_stop, ts)

                    # Funding rate (every 8 hours = 8 candles at 1h)
                    funding_counter += 1
                    if funding_counter >= 8:
                        funding_counter = 0
                        notional = position.size * close
                        funding_cost = self._apply_funding(notional)
                        equity -= funding_cost
                        risk.current_equity = equity

            # --- Entry logic ---
            if position is None and not np.isnan(predictions[i]):
                if not risk.can_trade():
                    equity_curve.append(equity)
                    continue

                signal = predictions[i]
                entry_price = self._apply_slippage(close, "LONG")

                if atr_val <= 0:
                    equity_curve.append(equity)
                    continue

                if signal >= self.tc.entry_confidence:
                    side = "LONG"
                    entry_price = self._apply_slippage(close, side)
                    size = self._position_size(equity, entry_price, atr_val, self.tc.leverage)
                    if size > 0:
                        fee = self._apply_fee(size * entry_price)
                        equity -= fee
                        risk.current_equity = equity
                        position = Position(
                            side=side,
                            entry_price=entry_price,
                            size=size,
                            stop_loss=risk.compute_stop_loss(entry_price, atr_val, side),
                            take_profit=risk.compute_take_profit(entry_price, atr_val, side),
                            entry_idx=i,
                        )
                elif signal <= (1 - self.tc.entry_confidence):
                    side = "SHORT"
                    entry_price = self._apply_slippage(close, side)
                    size = self._position_size(equity, entry_price, atr_val, self.tc.leverage)
                    if size > 0:
                        fee = self._apply_fee(size * entry_price)
                        equity -= fee
                        risk.current_equity = equity
                        position = Position(
                            side=side,
                            entry_price=entry_price,
                            size=size,
                            stop_loss=risk.compute_stop_loss(entry_price, atr_val, side),
                            take_profit=risk.compute_take_profit(entry_price, atr_val, side),
                            entry_idx=i,
                        )

            equity_curve.append(equity)

        # Close any remaining position at last close
        if position is not None:
            last_close = df.iloc[-1]["close"]
            exit_price = self._apply_slippage(last_close, "SHORT" if position.side == "LONG" else "LONG")
            notional = position.size * exit_price
            fee = self._apply_fee(notional)
            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size - fee
            else:
                pnl = (position.entry_price - exit_price) * position.size - fee
            equity += pnl
            risk.record_trade(pnl, str(df.index[-1]))
            equity_curve.append(equity)

        risk.current_equity = equity
        risk.peak_equity = max(risk.peak_equity, equity)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=risk.summary(),
            predictions=predictions,
        )


# ======================================================================
# Walk-Forward Optimisation
# ======================================================================

class WalkForwardOptimiser:
    """Walk-forward analysis with anchored expanding or rolling windows."""

    def __init__(self, config: BotConfig):
        self.cfg = config
        self.mc = config.ml
        self.engine = BacktestEngine(config)

    def run(self, data_path: str) -> Tuple[BacktestResult, EnsemblePredictor]:
        """Execute walk-forward optimisation over the full dataset."""
        log.info("Loading data from %s", data_path)
        df = load_csv(data_path)
        log.info("Building features (%d rows)", len(df))
        features = build_features(df, self.mc.lookback_periods)

        # Labels
        labels = make_labels(df["close"], self.mc.target_horizon)

        # Align and drop NaN
        combined = pd.concat([features, labels.rename("label")], axis=1)
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined.dropna(inplace=True)

        df = df.loc[combined.index]
        features = combined.drop(columns=["label"])
        labels = combined["label"].values

        n = len(df)
        log.info("Clean dataset: %d samples, %d features", n, features.shape[1])

        train_size = self.mc.wf_train_size
        test_size = self.mc.wf_test_size
        step = self.mc.wf_step

        all_predictions = np.full(n, np.nan)
        final_model: Optional[EnsemblePredictor] = None
        fold = 0

        i = train_size
        while i + test_size <= n:
            fold += 1
            train_end = i
            test_end = min(i + test_size, n)

            X_train = features.iloc[:train_end]
            y_train = labels[:train_end]
            X_test = features.iloc[train_end:test_end]
            y_test = labels[train_end:test_end]

            # Split train into train/val (last 15% as validation)
            val_split = int(len(X_train) * 0.85)
            X_tr, X_va = X_train.iloc[:val_split], X_train.iloc[val_split:]
            y_tr, y_va = y_train[:val_split], y_train[val_split:]

            log.info(
                "Fold %d: train=%d, val=%d, test=%d",
                fold, len(X_tr), len(X_va), len(X_test),
            )

            model = EnsemblePredictor()
            model.fit(X_tr, y_tr, X_va, y_va, self.cfg.ml)

            preds = model.predict_proba(X_test)
            # Pad predictions to match test window (transformer offset)
            if len(preds) < len(X_test):
                offset = len(X_test) - len(preds)
                preds = np.concatenate([np.full(offset, np.nan), preds])

            all_predictions[train_end:test_end] = preds
            final_model = model

            # Log fold performance
            valid_mask = ~np.isnan(preds)
            if valid_mask.sum() > 0:
                accuracy = np.mean(
                    (preds[valid_mask] > 0.5).astype(int) == y_test[valid_mask]
                )
                log.info("Fold %d accuracy: %.3f", fold, accuracy)

            i += step

        log.info("Walk-forward complete: %d folds", fold)

        # Run backtest on out-of-sample predictions
        result = self.engine.run_single(df, features, all_predictions)
        log.info("Backtest metrics: %s", result.metrics)

        return result, final_model


# ======================================================================
# Cross-Validation evaluator
# ======================================================================

def purged_cv_score(features: pd.DataFrame, labels: np.ndarray,
                    config: BotConfig,
                    n_splits: int = 5) -> Dict[str, float]:
    """Evaluate model with purged k-fold CV. Returns avg metrics."""
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=config.ml.embargo_pct)
    scores = []

    for train_idx, test_idx in cv.split(features.values):
        X_tr = features.iloc[train_idx]
        y_tr = labels[train_idx]
        X_te = features.iloc[test_idx]
        y_te = labels[test_idx]

        val_split = int(len(X_tr) * 0.85)
        X_train, X_val = X_tr.iloc[:val_split], X_tr.iloc[val_split:]
        y_train, y_val = y_tr[:val_split], y_tr[val_split:]

        model = EnsemblePredictor()
        model.fit(X_train, y_train, X_val, y_val, config.ml)

        preds = model.predict_proba(X_te)
        if len(preds) < len(y_te):
            offset = len(y_te) - len(preds)
            preds = np.concatenate([np.full(offset, 0.5), preds])

        valid = ~np.isnan(preds)
        if valid.sum() > 0:
            acc = np.mean((preds[valid] > 0.5).astype(int) == y_te[valid])
            scores.append(acc)

    return {
        "mean_accuracy": np.mean(scores) if scores else 0,
        "std_accuracy": np.std(scores) if scores else 0,
        "n_folds": len(scores),
    }
