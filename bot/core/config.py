"""Central configuration for the trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class APIConfig:
    """Binance API credentials and endpoints."""

    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    base_url: str = "https://fapi.binance.com"
    ws_url: str = "wss://fstream.binance.com"
    recv_window: int = 5000
    rate_limit_per_min: int = 1200
    order_rate_limit_per_10s: int = 100


@dataclass
class TradingConfig:
    """Trading parameters optimised for a $10 starting balance."""

    symbols: List[str] = field(default_factory=lambda: ["XRPUSDT"])
    leverage: int = 10
    margin_type: str = "ISOLATED"
    # Position sizing — aggressive for scalping edge compounding
    risk_per_trade_pct: float = 3.0  # % of equity risked per trade
    max_position_pct: float = 90.0  # max % of equity in a single position
    min_notional_usd: float = 5.0  # Binance minimum notional
    # Timeframes
    primary_tf: str = "1h"
    # Signal thresholds — wider for more frequent scalping trades
    long_entry_threshold: float = 0.56
    short_entry_threshold: float = 0.44
    signal_exit_long: float = 0.44  # exit LONG if signal flips bearish
    signal_exit_short: float = 0.58  # exit SHORT if signal flips bullish
    # Time-based exit
    max_hold_candles: int = 18  # force exit after 3x prediction horizon


@dataclass
class RiskConfig:
    """Risk management parameters."""

    # ATR-based stops
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5  # R:R = 1.67:1 — more reward per risk unit
    # Equity-curve circuit breaker
    max_drawdown_pct: float = 30.0  # hard kill-switch
    soft_drawdown_pct: float = 15.0  # start scaling risk down at this level
    max_daily_loss_pct: float = 10.0
    max_consecutive_losses: int = 8
    # Flash-crash protection
    flash_crash_pct: float = 15.0  # single-candle move (hourly)
    cooldown_candles: int = 2
    # Trailing stop — let winners run before locking in
    trailing_activate_pct: float = 1.5
    trailing_callback_pct: float = 0.6


@dataclass
class MLConfig:
    """Machine-learning model parameters."""

    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    target_horizon: int = 6  # candles ahead for label
    label_threshold: float = 0.01  # for threshold-filtered training
    # Feature selection
    top_n_features: int = 30
    # LightGBM — stronger regularisation to prevent overfitting
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    })
    # Transformer
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_seq_len: int = 48
    transformer_epochs: int = 30
    transformer_lr: float = 1e-3
    transformer_batch_size: int = 64
    # Walk-forward
    wf_train_size: int = 5000
    wf_test_size: int = 1000
    wf_step: int = 1000
    # Cross-validation
    n_purged_cv_splits: int = 5
    embargo_pct: float = 0.01


@dataclass
class BacktestConfig:
    """Backtesting engine parameters."""

    initial_balance: float = 10.0
    maker_fee_pct: float = 0.02  # 0.02%
    taker_fee_pct: float = 0.04  # 0.04%
    slippage_pct: float = 0.03  # 0.03%
    funding_rate_pct: float = 0.01  # 8-hourly funding estimate


@dataclass
class TelegramConfig:
    """Telegram notification settings."""

    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    enabled: bool = True  # set False to disable even if token/chat_id are set


@dataclass
class BotConfig:
    """Root configuration aggregating all sub-configs."""

    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    log_level: str = "INFO"
    data_dir: str = "data"
