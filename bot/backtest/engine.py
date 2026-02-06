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
from bot.core.features import (
    build_features, load_csv, make_labels, make_threshold_labels, select_features,
)
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
    hold_candles: int = 0


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
                       atr: float, leverage: int,
                       risk_scale: float = 1.0) -> float:
        """Risk-based position sizing respecting minimum notional."""
        risk_amount = equity * self.tc.risk_per_trade_pct / 100 * risk_scale
        stop_distance = atr * self.rc.sl_atr_mult
        if stop_distance <= 0 or entry_price <= 0:
            return 0.0

        # Size in quote (USDT) terms — notional value
        # risk_amount / pct_stop = notional that loses risk_amount on SL hit
        # Do NOT multiply by leverage: P&L is on notional, not margin
        size_quote = risk_amount / (stop_distance / entry_price)
        max_quote = equity * self.tc.max_position_pct / 100 * leverage

        size_quote = min(size_quote, max_quote)

        if size_quote < self.tc.min_notional_usd:
            return 0.0

        return size_quote / entry_price

    def _close_position(self, position: Position, exit_price: float,
                        reason: str, equity: float, risk: RiskManager,
                        trades: List[dict], idx: int,
                        timestamp: str) -> float:
        """Close a position and return updated equity."""
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
        risk.record_trade(pnl, timestamp)
        trades.append({
            "entry_idx": position.entry_idx, "exit_idx": idx,
            "side": position.side, "entry_price": position.entry_price,
            "exit_price": exit_price, "size": position.size,
            "pnl": pnl, "reason": reason, "equity_after": equity,
            "hold_candles": position.hold_candles,
        })
        return equity

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
                equity = self._close_position(
                    position, exit_price, "kill_switch", equity,
                    risk, trades, i, str(df.index[i]),
                )
                position = None
                equity_curve.append(equity)
                continue

            # --- Manage open position ---
            if position is not None:
                position.hold_candles += 1
                exit_price = None
                exit_reason = ""

                # 1. SL / TP check
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

                # 2. Time-based forced exit
                if exit_price is None and position.hold_candles >= self.tc.max_hold_candles:
                    opp_side = "SHORT" if position.side == "LONG" else "LONG"
                    exit_price = self._apply_slippage(close, opp_side)
                    exit_reason = "time_exit"

                # 3. Signal-based exit (only if no SL/TP/time hit)
                if exit_price is None and not np.isnan(predictions[i]):
                    if position.side == "LONG" and predictions[i] < self.tc.signal_exit_long:
                        exit_price = self._apply_slippage(close, "SHORT")
                        exit_reason = "signal_exit"
                    elif position.side == "SHORT" and predictions[i] > self.tc.signal_exit_short:
                        exit_price = self._apply_slippage(close, "LONG")
                        exit_reason = "signal_exit"

                if exit_price is not None:
                    equity = self._close_position(
                        position, exit_price, exit_reason, equity,
                        risk, trades, i, str(df.index[i]),
                    )
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

                if atr_val <= 0:
                    equity_curve.append(equity)
                    continue

                side = None
                if signal >= self.tc.long_entry_threshold:
                    side = "LONG"
                elif signal <= self.tc.short_entry_threshold:
                    side = "SHORT"

                if side is not None:
                    entry_price = self._apply_slippage(close, side)
                    size = self._position_size(
                        equity, entry_price, atr_val, self.tc.leverage,
                        risk_scale=risk.risk_scale_factor(),
                    )
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
            opp_side = "SHORT" if position.side == "LONG" else "LONG"
            exit_price = self._apply_slippage(last_close, opp_side)
            equity = self._close_position(
                position, exit_price, "end_of_data", equity,
                risk, trades, n - 1, str(df.index[-1]),
            )
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
    """Walk-forward analysis with anchored expanding windows.

    Uses threshold-based labels and feature selection to prevent overfitting.
    """

    def __init__(self, config: BotConfig):
        self.cfg = config
        self.mc = config.ml
        self.engine = BacktestEngine(config)

    def run(self, data_path: str) -> Tuple[BacktestResult, EnsemblePredictor]:
        """Execute walk-forward optimisation over the full dataset."""
        log.info("Loading data from %s", data_path)
        df = load_csv(data_path)
        log.info("Building features (%d rows)", len(df))
        all_features = build_features(df, self.mc.lookback_periods)

        # Threshold-filtered labels (NaN for neutral moves)
        labels_raw = make_threshold_labels(
            df["close"], self.mc.target_horizon, self.mc.label_threshold,
        )

        # All features aligned (drop rows where features are NaN)
        all_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_mask = all_features.notna().all(axis=1)
        all_features = all_features[feat_mask]
        df = df.loc[all_features.index]
        labels_raw = labels_raw.loc[all_features.index]

        n = len(df)
        log.info("Feature matrix: %d rows, %d features", n, all_features.shape[1])

        # Feature selection on first chunk of labeled data
        labeled_mask = labels_raw.notna()
        labeled_feat = all_features[labeled_mask]
        labeled_y = labels_raw[labeled_mask].values

        log.info("Labeled samples: %d / %d (%.1f%%)",
                 len(labeled_feat), n, len(labeled_feat) / n * 100)

        selected_cols = select_features(
            labeled_feat, labeled_y, self.mc.top_n_features,
        )
        log.info("Selected %d features: %s", len(selected_cols), selected_cols[:10])

        # Keep only selected features (but maintain full index for backtest)
        features = all_features[selected_cols]

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

            # Training: use only labeled (non-neutral) rows
            train_features = features.iloc[:train_end]
            train_labels = labels_raw.iloc[:train_end]
            train_mask = train_labels.notna()

            X_train_labeled = train_features[train_mask]
            y_train_labeled = train_labels[train_mask].values

            if len(X_train_labeled) < 200:
                log.warning("Fold %d: insufficient labeled data (%d), skipping",
                            fold, len(X_train_labeled))
                i += step
                continue

            # Split labeled train into train/val (last 15%)
            val_split = int(len(X_train_labeled) * 0.85)
            X_tr = X_train_labeled.iloc[:val_split]
            y_tr = y_train_labeled[:val_split]
            X_va = X_train_labeled.iloc[val_split:]
            y_va = y_train_labeled[val_split:]

            log.info(
                "Fold %d: train=%d, val=%d, test=%d (labeled_train=%d)",
                fold, train_end, len(X_va), test_end - train_end,
                len(X_train_labeled),
            )

            model = EnsemblePredictor()
            model.fit(X_tr, y_tr, X_va, y_va, self.cfg.ml)

            # Predict on ALL test rows (not just labeled ones)
            X_test = features.iloc[train_end:test_end]
            preds = model.predict_proba(X_test)

            # Pad predictions to match test window (transformer offset)
            if len(preds) < len(X_test):
                offset = len(X_test) - len(preds)
                preds = np.concatenate([np.full(offset, np.nan), preds])

            all_predictions[train_end:test_end] = preds
            final_model = model

            # Log fold performance on labeled test data only
            test_labels = labels_raw.iloc[train_end:test_end]
            test_labeled_mask = test_labels.notna()
            valid_preds = preds[test_labeled_mask.values]
            valid_labels = test_labels[test_labeled_mask].values
            valid_mask = ~np.isnan(valid_preds)

            if valid_mask.sum() > 0:
                accuracy = np.mean(
                    (valid_preds[valid_mask] > 0.5).astype(int) == valid_labels[valid_mask]
                )
                log.info("Fold %d OOS accuracy (labeled): %.3f (n=%d)",
                         fold, accuracy, valid_mask.sum())

            i += step

        log.info("Walk-forward complete: %d folds", fold)

        # Run backtest — pass full features so atr_14 is always available
        result = self.engine.run_single(df, all_features, all_predictions)
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
