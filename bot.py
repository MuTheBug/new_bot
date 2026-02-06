"""
Autonomous Trading Bot.

Connects to Binance Futures, monitors markets 24/7, generates signals,
executes trades, and manages positions with full autonomy.

Supports:
- Regular orders for small accounts
- TWAP/VP algo orders for larger positions (>1000 USDT notional)
- Dynamic leverage based on account size
- Automatic position sizing with fee awareness
- Trailing stops and take-profit management
- Circuit breakers for risk management
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from exchange import BinanceClient
from risk_manager import RiskManager
from strategy import (
    add_indicators,
    detect_regime,
    generate_signals,
    get_stop_loss,
    get_take_profit,
    get_trailing_stop,
)

logger = logging.getLogger(__name__)

LOG_DIR = config.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)


class Position:
    """Tracks an open position."""

    def __init__(self, symbol, side, entry_price, quantity, stop_loss,
                 take_profit, leverage, atr):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.atr = atr
        self.entry_time = datetime.now(timezone.utc)
        self.trailing_stop = None
        self.stop_order_id = None
        self.tp_order_id = None


class TradingBot:
    """Main autonomous trading bot."""

    def __init__(self):
        self.client = BinanceClient()
        self.risk_manager = RiskManager()
        self.positions = {}  # symbol -> Position
        self.last_signal_time = {}
        self.running = False
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_file = os.path.join(
            LOG_DIR,
            f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logger.info("Bot logging initialized: %s", log_file)

    def initialize(self):
        """Initialize bot: test connection, set up trading parameters."""
        logger.info("Initializing trading bot...")

        # Test connectivity
        if not self.client.ping():
            logger.error("Cannot connect to Binance API")
            return False

        logger.info("API connection successful")

        # Set position mode to one-way
        try:
            self.client.set_position_mode(dual_side=False)
        except Exception as e:
            logger.debug("Position mode setup: %s", e)

        # Configure each trading pair
        for symbol in config.TRADING_PAIRS:
            try:
                self.client.set_margin_type(symbol, "ISOLATED")
            except Exception as e:
                logger.debug("Margin type for %s: %s", symbol, e)

            # Set initial leverage (will be adjusted per trade)
            try:
                self.client.set_leverage(symbol, config.LEVERAGE["max_leverage"])
            except Exception as e:
                logger.warning("Leverage setup for %s: %s", symbol, e)

        # Get initial balance
        balance = self.client.get_balance()
        self.risk_manager.peak_balance = balance["total"]
        logger.info(
            "Initial balance: $%.2f (available: $%.2f)",
            balance["total"], balance["available"],
        )
        logger.info("Growth tier: %s", self.risk_manager.get_growth_tier(balance["total"]))

        # Sync existing positions
        self._sync_positions()

        logger.info("Bot initialization complete")
        return True

    def _sync_positions(self):
        """Sync with any existing open positions on the exchange."""
        try:
            exchange_positions = self.client.get_positions()
            for pos in exchange_positions:
                symbol = pos["symbol"]
                if symbol in config.TRADING_PAIRS:
                    logger.info(
                        "Found existing position: %s %s %.4f @ %.4f",
                        symbol, pos["side"], pos["size"], pos["entry_price"],
                    )
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=pos["side"],
                        entry_price=pos["entry_price"],
                        quantity=pos["size"],
                        stop_loss=0,  # Will be recalculated
                        take_profit=0,
                        leverage=pos["leverage"],
                        atr=0,
                    )
        except Exception as e:
            logger.warning("Could not sync positions: %s", e)

    def run(self):
        """Main bot loop. Runs continuously until stopped."""
        if not self.initialize():
            logger.error("Initialization failed. Exiting.")
            return

        self.running = True
        logger.info("Bot started. Monitoring %s", config.TRADING_PAIRS)

        cycle_count = 0
        last_performance_log = 0

        while self.running:
            try:
                cycle_count += 1
                now = time.time()

                # Process each trading pair
                for symbol in config.TRADING_PAIRS:
                    self._process_symbol(symbol)

                # Periodic performance logging
                if now - last_performance_log >= config.MONITOR["performance_log_interval"]:
                    self._log_performance()
                    last_performance_log = now

                # Wait for next candle (1hr timeframe = check every 5 min)
                time.sleep(300)

            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                self.running = False
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                time.sleep(60)  # Wait a minute before retrying

        self._shutdown()

    def _process_symbol(self, symbol):
        """Process a single trading pair: fetch data, check signals, manage positions."""
        try:
            # Fetch latest candles
            klines = self.client.get_klines(symbol, config.TIMEFRAME, config.CANDLE_LIMIT)
            df = self._klines_to_df(klines)

            if len(df) < config.STRATEGY["trend_ema_period"] + 10:
                logger.debug("Not enough data for %s", symbol)
                return

            # Generate signals
            df = generate_signals(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            regime = detect_regime(df)

            # Manage existing position
            if symbol in self.positions:
                self._manage_position(symbol, latest, df)
            # Check for new entry
            elif latest["signal"] != 0 and self.risk_manager.can_trade():
                if len(self.positions) < config.RISK["max_open_positions"]:
                    self._enter_position(symbol, latest, df)

        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e, exc_info=True)

    def _klines_to_df(self, klines):
        """Convert API kline data to DataFrame."""
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df

    def _enter_position(self, symbol, signal_row, df):
        """Enter a new position based on signal."""
        signal = signal_row["signal"]
        side = "LONG" if signal == 1 else "SHORT"
        order_side = "BUY" if side == "LONG" else "SELL"
        price = signal_row["close"]
        atr = signal_row["atr"]

        if pd.isna(atr) or atr <= 0:
            return

        # Get balance and calculate sizing
        balance_info = self.client.get_balance()
        balance = balance_info["available"]

        leverage = self.risk_manager.calculate_optimal_leverage(
            balance, atr, price, symbol
        )
        quantity = self.risk_manager.calculate_position_size(
            balance, leverage, price, atr, symbol
        )

        if quantity <= 0:
            logger.info("Cannot size position for %s (balance=$%.2f)", symbol, balance)
            return

        # Set leverage
        try:
            self.client.set_leverage(symbol, leverage)
        except Exception as e:
            logger.error("Failed to set leverage for %s: %s", symbol, e)
            return

        # Calculate stops
        stop_mult = config.STRATEGY["atr_stop_multiplier"]
        tp_mult = config.STRATEGY["atr_tp_multiplier"]

        if side == "LONG":
            stop_loss = round(price - (atr * stop_mult), 8)
            take_profit = round(price + (atr * tp_mult), 8)
        else:
            stop_loss = round(price + (atr * stop_mult), 8)
            take_profit = round(price - (atr * tp_mult), 8)

        # Determine order execution method
        notional = quantity * price
        use_algo = self.risk_manager.should_use_algo_order(quantity, price)

        logger.info(
            "ENTRY SIGNAL: %s %s qty=%.4f @ $%.4f, "
            "SL=$%.4f, TP=$%.4f, Leverage=%dx, Notional=$%.2f %s",
            symbol, side, quantity, price, stop_loss, take_profit,
            leverage, notional, "(TWAP)" if use_algo else "(MARKET)",
        )

        try:
            # Place entry order
            if use_algo:
                duration = min(
                    config.ALGO["twap_default_duration"],
                    config.ALGO["twap_max_duration"],
                )
                self.client.place_twap_order(
                    symbol, order_side, quantity, duration
                )
            else:
                self.client.place_market_order(symbol, order_side, quantity)

            # Place stop-loss
            sl_side = "SELL" if side == "LONG" else "BUY"
            self.client.place_stop_market(
                symbol, sl_side, quantity, stop_loss
            )

            # Place take-profit
            self.client.place_take_profit_market(
                symbol, sl_side, quantity, take_profit
            )

            # Track position
            pos = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                atr=atr,
            )
            self.positions[symbol] = pos

            self._log_trade("ENTRY", pos)
            logger.info(
                "Position opened: %s %s qty=%.4f @ $%.4f (lev=%dx)",
                symbol, side, quantity, price, leverage,
            )

        except Exception as e:
            logger.error("Failed to enter %s position: %s", symbol, e)
            # Cancel any partial orders
            try:
                self.client.cancel_all_orders(symbol)
            except Exception:
                pass

    def _manage_position(self, symbol, latest, df):
        """Manage an existing position: update trailing stop, check exit signals."""
        pos = self.positions[symbol]
        current_price = latest["close"]
        atr = latest["atr"] if not pd.isna(latest["atr"]) else pos.atr

        # Update trailing stop
        trailing = get_trailing_stop(
            pos.entry_price, current_price, atr, pos.side
        )

        if trailing is not None:
            should_update = False
            if pos.trailing_stop is None:
                should_update = True
            elif pos.side == "LONG" and trailing > pos.trailing_stop:
                should_update = True
            elif pos.side == "SHORT" and trailing < pos.trailing_stop:
                should_update = True

            if should_update:
                pos.trailing_stop = trailing
                # Update stop on exchange
                try:
                    self.client.cancel_all_orders(symbol)
                    sl_side = "SELL" if pos.side == "LONG" else "BUY"
                    self.client.place_stop_market(
                        symbol, sl_side, pos.quantity, round(trailing, 8)
                    )
                    # Re-place take profit
                    self.client.place_take_profit_market(
                        symbol, sl_side, pos.quantity, round(pos.take_profit, 8)
                    )
                    logger.info(
                        "Trailing stop updated for %s: $%.4f",
                        symbol, trailing,
                    )
                except Exception as e:
                    logger.error("Failed to update trailing stop: %s", e)

        # Check if position was closed by exchange (stop/TP hit)
        exchange_positions = self.client.get_positions(symbol)
        if not exchange_positions:
            # Position was closed
            if pos.side == "LONG":
                pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity

            notional = pos.quantity * pos.entry_price
            fees = notional * config.FEES["default"] * 2
            pnl -= fees

            balance_info = self.client.get_balance()
            self.risk_manager.record_trade(pnl, balance_info["total"])

            self._log_trade("EXIT", pos, pnl=pnl)
            logger.info(
                "Position closed: %s %s PnL=$%.4f",
                symbol, pos.side, pnl,
            )
            del self.positions[symbol]

        # Check for exit signal (trend reversal)
        elif latest["signal"] != 0:
            signal_side = "LONG" if latest["signal"] == 1 else "SHORT"
            if signal_side != pos.side:
                logger.info("Reverse signal for %s, closing position", symbol)
                self._close_position(symbol)

    def _close_position(self, symbol):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        close_side = "SELL" if pos.side == "LONG" else "BUY"

        try:
            # Cancel existing orders
            self.client.cancel_all_orders(symbol)
            # Market close
            self.client.place_market_order(symbol, close_side, pos.quantity)

            current_price = self.client.get_mark_price(symbol)
            if pos.side == "LONG":
                pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity

            notional = pos.quantity * pos.entry_price
            fees = notional * config.FEES["default"] * 2
            pnl -= fees

            balance_info = self.client.get_balance()
            self.risk_manager.record_trade(pnl, balance_info["total"])

            self._log_trade("EXIT", pos, pnl=pnl)
            logger.info(
                "Position closed: %s %s PnL=$%.4f",
                symbol, pos.side, pnl,
            )

        except Exception as e:
            logger.error("Failed to close position %s: %s", symbol, e)
        finally:
            if symbol in self.positions:
                del self.positions[symbol]

    def _log_trade(self, action, pos, pnl=None):
        """Log trade to JSON file for analysis."""
        trade_log = os.path.join(LOG_DIR, "trades.json")

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "leverage": pos.leverage,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
        }
        if pnl is not None:
            entry["pnl"] = pnl

        trades = []
        if os.path.exists(trade_log):
            try:
                with open(trade_log, "r") as f:
                    trades = json.load(f)
            except (json.JSONDecodeError, IOError):
                trades = []

        trades.append(entry)
        with open(trade_log, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    def _log_performance(self):
        """Log current performance metrics."""
        try:
            balance_info = self.client.get_balance()
            balance = balance_info["total"]
            status = self.risk_manager.get_status(balance)

            logger.info(
                "PERFORMANCE | Balance: $%.2f | Tier: %s | Drawdown: %.1%% | "
                "Daily PnL: $%.4f | Positions: %d | Halted: %s",
                balance, status["tier"], status["current_drawdown"] * 100,
                status["daily_pnl"], len(self.positions), status["is_halted"],
            )

            # Save performance snapshot
            perf_file = os.path.join(LOG_DIR, "performance.json")
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **status,
                "open_positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                    }
                    for p in self.positions.values()
                ],
            }

            snapshots = []
            if os.path.exists(perf_file):
                try:
                    with open(perf_file, "r") as f:
                        snapshots = json.load(f)
                except (json.JSONDecodeError, IOError):
                    snapshots = []

            snapshots.append(snapshot)
            # Keep last 1000 snapshots
            snapshots = snapshots[-1000:]
            with open(perf_file, "w") as f:
                json.dump(snapshots, f, indent=2, default=str)

        except Exception as e:
            logger.error("Failed to log performance: %s", e)

    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down bot...")

        # Close all positions if configured to do so
        # (Default: leave positions open with stops in place)
        logger.info(
            "Open positions at shutdown: %d",
            len(self.positions),
        )
        for symbol, pos in self.positions.items():
            logger.info(
                "  %s %s qty=%.4f @ $%.4f (SL=$%.4f, TP=$%.4f)",
                symbol, pos.side, pos.quantity, pos.entry_price,
                pos.stop_loss, pos.take_profit,
            )

        logger.info("Bot shutdown complete")


def main():
    """Entry point for the trading bot."""
    print("=" * 50)
    print("AUTONOMOUS TRADING BOT - ATM Strategy")
    print("=" * 50)

    if not config.API_KEY or not config.API_SECRET:
        print("\nWARNING: API keys not configured.")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        print("Running in monitoring-only mode.\n")

    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
