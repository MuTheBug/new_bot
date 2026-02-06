#!/usr/bin/env python3
"""Run walk-forward backtesting and cross-validation on historical data.

Usage:
    python run_backtest.py [--data XRPUSDT_2022_2026.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument(
        "--data", default="XRPUSDT_2022_2026.csv",
        help="Path to OHLCV CSV file",
    )
    parser.add_argument(
        "--balance", type=float, default=10.0,
        help="Initial balance in USDT",
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Leverage multiplier",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Also run purged K-Fold cross-validation",
    )
    args = parser.parse_args()

    from bot.core.config import BotConfig
    from bot.backtest.engine import WalkForwardOptimiser, purged_cv_score
    from bot.core.features import build_features, load_csv, make_labels
    from bot.utils.logger import get_logger

    log = get_logger("backtest", "INFO")

    config = BotConfig()
    config.backtest.initial_balance = args.balance
    config.trading.leverage = args.leverage

    data_path = args.data
    if not os.path.exists(data_path):
        log.error("Data file not found: %s", data_path)
        sys.exit(1)

    log.info("=" * 60)
    log.info("WALK-FORWARD BACKTEST")
    log.info("=" * 60)
    log.info("Data: %s", data_path)
    log.info("Initial balance: $%.2f", args.balance)
    log.info("Leverage: %dx", args.leverage)

    wfo = WalkForwardOptimiser(config)
    result, model = wfo.run(data_path)

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    for k, v in result.metrics.items():
        if isinstance(v, float):
            log.info("  %-20s: %.4f", k, v)
        else:
            log.info("  %-20s: %s", k, v)

    log.info("Equity curve: $%.2f -> $%.2f",
             result.equity_curve[0], result.equity_curve[-1])
    log.info("Total trades: %d", len(result.trades))

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2, default=str)

    equity_df_data = {"equity": result.equity_curve}
    import pandas as pd
    eq_df = pd.DataFrame(equity_df_data)
    eq_df.to_csv("results/equity_curve.csv", index=False)

    if result.trades:
        trades_df = pd.DataFrame(result.trades)
        trades_df.to_csv("results/trades.csv", index=False)

    log.info("Results saved to results/")

    # --- Optional: Purged CV ---
    if args.cv:
        log.info("=" * 60)
        log.info("PURGED K-FOLD CROSS-VALIDATION")
        log.info("=" * 60)

        df = load_csv(data_path)
        features = build_features(df, config.ml.lookback_periods)
        labels = make_labels(df["close"], config.ml.target_horizon)

        combined = pd.concat([features, labels.rename("label")], axis=1)
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined.dropna(inplace=True)

        features_clean = combined.drop(columns=["label"])
        labels_clean = combined["label"].values

        cv_scores = purged_cv_score(features_clean, labels_clean, config,
                                    n_splits=config.ml.n_purged_cv_splits)
        for k, v in cv_scores.items():
            log.info("  %-20s: %s", k, v)

    return result


if __name__ == "__main__":
    main()
