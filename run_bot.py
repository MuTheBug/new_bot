#!/usr/bin/env python3
"""Launch the autonomous trading bot.

Requires:
    - BINANCE_API_KEY and BINANCE_API_SECRET environment variables
    - A trained model (run run_backtest.py first, or provide a model path)

Optional:
    - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for Telegram notifications

Usage:
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"
    export TELEGRAM_BOT_TOKEN="your_bot_token"
    export TELEGRAM_CHAT_ID="your_chat_id"
    python run_bot.py [--symbol XRPUSDT] [--leverage 10] [--train-first]
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Autonomous Binance Futures Bot")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    parser.add_argument(
        "--train-first", action="store_true",
        help="Train model on historical data before starting live trading",
    )
    parser.add_argument(
        "--data", default="XRPUSDT_2022_2026.csv",
        help="CSV data path for --train-first mode",
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Disable Telegram notifications",
    )
    args = parser.parse_args()

    from bot.core.config import BotConfig
    from bot.core.bot import TradingBot
    from bot.utils.logger import get_logger

    log = get_logger("main", "INFO")

    # Validate API credentials
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        log.error("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        sys.exit(1)

    config = BotConfig()
    config.trading.symbols = [args.symbol]
    config.trading.leverage = args.leverage

    if args.no_telegram:
        config.telegram.enabled = False

    model = None

    if args.train_first:
        log.info("Training model on %s before going live...", args.data)
        if not os.path.exists(args.data):
            log.error("Data file not found: %s", args.data)
            sys.exit(1)

        from bot.backtest.engine import WalkForwardOptimiser
        wfo = WalkForwardOptimiser(config)
        result, model = wfo.run(args.data)
        log.info("Training complete. Backtest metrics: %s", result.metrics)

    bot = TradingBot(config, model=model)
    bot.initialise()
    bot.run()


if __name__ == "__main__":
    main()
