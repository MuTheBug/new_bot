"""Main bot orchestrator — autonomous lifecycle management.

Handles signal generation, order placement, position management,
and continuous operation with error recovery.
"""

from __future__ import annotations

import signal
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd

from bot.core.config import BotConfig
from bot.core.features import build_features
from bot.core.position_sizer import PositionSizer, SymbolSpec
from bot.core.risk import RiskManager
from bot.exchange.client import BinanceAPIError, BinanceFuturesClient
from bot.ml.models import EnsemblePredictor
from bot.utils.logger import get_logger
from bot.utils.telegram import TelegramNotifier

log = get_logger("bot")


class TradingBot:
    """Fully autonomous trading bot for Binance USDT-M Futures."""

    def __init__(self, config: BotConfig,
                 model: Optional[EnsemblePredictor] = None):
        self.cfg = config
        self.client = BinanceFuturesClient(config.api)
        self.risk = RiskManager(config.risk)
        self.model = model
        self._running = False
        self._sizer: Optional[PositionSizer] = None
        self._current_position: Optional[dict] = None
        self._last_signal_time: float = 0

        # Telegram notifications
        tg = config.telegram
        if tg.enabled:
            self.notifier = TelegramNotifier(tg.bot_token, tg.chat_id)
        else:
            self.notifier = TelegramNotifier("", "")  # disabled stub

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        log.info("Shutdown signal received — closing gracefully")
        self._running = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Set up exchange state: leverage, margin type, symbol info."""
        symbol = self.cfg.trading.symbols[0]
        log.info("Initialising bot for %s", symbol)

        # Set leverage
        try:
            self.client.set_leverage(symbol, self.cfg.trading.leverage)
            log.info("Leverage set to %dx", self.cfg.trading.leverage)
        except BinanceAPIError as e:
            log.warning("Leverage setting: %s", e)

        # Set margin type
        try:
            self.client.set_margin_type(symbol, self.cfg.trading.margin_type)
            log.info("Margin type set to %s", self.cfg.trading.margin_type)
        except BinanceAPIError as e:
            log.warning("Margin type setting: %s", e)

        # Load symbol precision
        try:
            prec = self.client.get_symbol_precision(symbol)
            spec = SymbolSpec(
                tick_size=prec["tick_size"],
                step_size=prec["step_size"],
                min_qty=prec["min_qty"],
                min_notional=prec["min_notional"],
                price_precision=prec["price_precision"],
                qty_precision=prec["qty_precision"],
            )
            self._sizer = PositionSizer(
                spec, self.cfg.trading.leverage,
                self.cfg.trading.risk_per_trade_pct,
                self.cfg.trading.max_position_pct,
            )
            log.info("Symbol spec loaded: %s", prec)
        except Exception as e:
            log.error("Failed to load symbol info: %s", e)
            raise

        # Check balance
        balance = self.client.get_balance()
        self.risk.current_equity = balance
        self.risk.initial_equity = balance
        self.risk.peak_equity = balance
        log.info("Account balance: $%.2f", balance)

        feasibility = self._sizer.feasibility_report(balance, 0.5)  # rough XRP price
        if not feasibility["can_trade"]:
            log.warning("Balance may be too low for trading: %s", feasibility)

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def _fetch_candles(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Fetch recent klines and return as DataFrame."""
        raw = self.client.get_klines(symbol, self.cfg.trading.primary_tf, limit)
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["datetime"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        for col in ["open", "high", "low", "close", "volume",
                     "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, df: pd.DataFrame) -> Optional[float]:
        """Run feature engineering + model inference on latest data."""
        if self.model is None:
            log.error("No model loaded — cannot generate signal")
            return None

        features = build_features(df, self.cfg.ml.lookback_periods)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)

        if len(features) < 2:
            return None

        proba = self.model.predict_proba(features)
        if len(proba) == 0 or np.isnan(proba[-1]):
            return None

        return float(proba[-1])

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _open_position(self, symbol: str, side: str, signal: float,
                       current_price: float, atr: float) -> None:
        """Place entry order with server-side SL/TP via algo orders."""
        if self._sizer is None:
            return

        stop_distance = atr * self.cfg.risk.sl_atr_mult
        qty = self._sizer.compute_quantity(
            self.risk.current_equity, current_price, stop_distance
        )
        if qty is None:
            log.info("Position sizing returned None — skipping trade")
            return

        sl_price = self._sizer.compute_stop_loss_price(
            current_price, atr, self.cfg.risk.sl_atr_mult, side
        )
        tp_price = self._sizer.compute_take_profit_price(
            current_price, atr, self.cfg.risk.tp_atr_mult, side
        )

        log.info(
            "Opening %s %.4f %s @ ~%.4f | SL=%.4f TP=%.4f | signal=%.3f",
            side, qty, symbol, current_price, sl_price, tp_price, signal,
        )

        try:
            # Entry order
            order_side = "BUY" if side == "LONG" else "SELL"
            entry_result = self.client.place_market_order(symbol, order_side, qty)
            log.info("Entry order placed: %s", entry_result.get("orderId"))

            # Server-side stop-loss
            sl_side = "SELL" if side == "LONG" else "BUY"
            self.client.place_stop_market(
                symbol, sl_side, qty, sl_price, close_position=True
            )
            log.info("SL order placed at %.4f", sl_price)

            # Server-side take-profit
            self.client.place_take_profit_market(
                symbol, sl_side, qty, tp_price, close_position=True
            )
            log.info("TP order placed at %.4f", tp_price)

            self._current_position = {
                "side": side,
                "entry_price": current_price,
                "qty": qty,
                "sl": sl_price,
                "tp": tp_price,
                "symbol": symbol,
            }

            self.notifier.notify_position_opened(
                side=side, symbol=symbol, qty=qty,
                entry_price=current_price, sl_price=sl_price,
                tp_price=tp_price, signal=signal,
                equity=self.risk.current_equity,
            )

        except BinanceAPIError as e:
            log.error("Order placement failed: %s", e)
            self.notifier.notify_error(str(e), context="Order placement")

    def _close_position(self, symbol: str, reason: str = "signal") -> None:
        """Close current position and cancel outstanding orders."""
        if self._current_position is None:
            return

        pos = self._current_position
        close_side = "SELL" if pos["side"] == "LONG" else "BUY"

        try:
            self.client.cancel_all_orders(symbol)
            self.client.place_market_order(symbol, close_side, pos["qty"])
            log.info("Position closed (%s): %s %.4f %s",
                     reason, pos["side"], pos["qty"], symbol)
            self.notifier.notify_position_closed(
                side=pos["side"], symbol=symbol, reason=reason,
                equity=self.risk.current_equity,
            )
        except BinanceAPIError as e:
            log.error("Close position failed: %s", e)
            self.notifier.notify_error(str(e), context="Close position")

        self._current_position = None

    def _sync_position(self, symbol: str) -> None:
        """Sync local state with exchange position state."""
        try:
            positions = self.client.get_positions()
            for p in positions:
                if p.get("symbol") == symbol:
                    amt = float(p.get("positionAmt", 0))
                    if amt != 0:
                        self._current_position = {
                            "side": "LONG" if amt > 0 else "SHORT",
                            "entry_price": float(p.get("entryPrice", 0)),
                            "qty": abs(amt),
                            "symbol": symbol,
                            "sl": 0, "tp": 0,
                        }
                        return
            # No position found on exchange
            if self._current_position is not None:
                # Position was closed by SL/TP on server side
                log.info("Position closed by server-side order")
                self._current_position = None
        except Exception as e:
            log.error("Position sync failed: %s", e)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main autonomous trading loop."""
        self._running = True
        symbol = self.cfg.trading.symbols[0]
        log.info("Starting autonomous trading loop for %s", symbol)

        self.notifier.notify_startup(
            symbol=symbol,
            leverage=self.cfg.trading.leverage,
            equity=self.risk.current_equity,
        )

        cycle = 0
        last_daily_reset = 0

        while self._running:
            cycle += 1
            try:
                # Daily risk reset and summary
                now = time.time()
                current_day = int(now // 86400)
                if current_day != last_daily_reset:
                    if last_daily_reset != 0:
                        # Send daily summary for the completed day
                        daily_pnl = (
                            self.risk.current_equity
                            - self.risk.daily_start_equity
                        )
                        daily_trades = len([
                            t for t in self.risk.trade_history
                            if t.timestamp and t.timestamp[:10] == time.strftime(
                                "%Y-%m-%d", time.gmtime(now - 86400)
                            )
                        ])
                        self.notifier.notify_daily_summary(
                            equity=self.risk.current_equity,
                            daily_pnl=daily_pnl,
                            trades_today=daily_trades,
                            drawdown_pct=self.risk.drawdown_pct(),
                        )
                    self.risk.reset_daily()
                    last_daily_reset = current_day

                # Sync position state
                self._sync_position(symbol)

                # Update equity
                balance = self.client.get_balance()
                self.risk.current_equity = balance
                if balance > self.risk.peak_equity:
                    self.risk.peak_equity = balance

                # Fetch latest data
                df = self._fetch_candles(symbol, limit=200)
                if len(df) < 100:
                    log.warning("Insufficient data: %d candles", len(df))
                    time.sleep(60)
                    continue

                # Extract ATR for current candle
                from bot.core.features import _atr
                atr_series = _atr(df["high"], df["low"], df["close"],
                                  self.cfg.risk.atr_period)
                current_atr = float(atr_series.iloc[-1])

                # Flash crash check
                last = df.iloc[-1]
                self.risk.detect_flash_crash(
                    last["open"], last["close"], last["high"], last["low"]
                )

                # Generate signal
                signal = self._generate_signal(df)
                current_price = float(df.iloc[-1]["close"])

                log.info(
                    "Cycle %d | price=%.4f | atr=%.5f | signal=%s | equity=$%.2f | dd=%.1f%%",
                    cycle, current_price, current_atr,
                    f"{signal:.3f}" if signal else "None",
                    balance, self.risk.drawdown_pct(),
                )

                if signal is None:
                    time.sleep(30)
                    continue

                # --- Position management ---
                if self._current_position is not None:
                    pos = self._current_position
                    # Check for signal-based exit
                    if pos["side"] == "LONG" and signal < self.cfg.trading.signal_exit_long:
                        self._close_position(symbol, "signal_reversal")
                    elif pos["side"] == "SHORT" and signal > self.cfg.trading.signal_exit_short:
                        self._close_position(symbol, "signal_reversal")
                    else:
                        # Update trailing stop
                        ts = self.risk.compute_trailing_stop(
                            pos["entry_price"], current_price, pos["side"]
                        )
                        if ts is not None:
                            log.debug("Trailing stop update candidate: %.4f", ts)

                # --- Entry logic ---
                if self._current_position is None:
                    if not self.risk.can_trade():
                        if self.risk._killed and not getattr(self, '_kill_notified', False):
                            self.notifier.notify_kill_switch(
                                drawdown_pct=self.risk.drawdown_pct(),
                                equity=self.risk.current_equity,
                            )
                            self._kill_notified = True
                    else:
                        if signal >= self.cfg.trading.long_entry_threshold:
                            self._open_position(
                                symbol, "LONG", signal, current_price, current_atr
                            )
                        elif signal <= self.cfg.trading.short_entry_threshold:
                            self._open_position(
                                symbol, "SHORT", signal, current_price, current_atr
                            )

                # Wait for next candle (1h intervals, check every 60s)
                time.sleep(60)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("Error in main loop: %s\n%s", e, traceback.format_exc())
                self.notifier.notify_error(
                    str(e), context=f"Main loop cycle {cycle}"
                )
                time.sleep(30)

        # Cleanup
        summary = self.risk.summary()
        log.info("Bot stopped. Final equity: $%.2f", self.risk.current_equity)
        log.info("Summary: %s", summary)
        self.notifier.notify_shutdown(
            equity=self.risk.current_equity,
            total_trades=summary.get("trades", 0),
        )
