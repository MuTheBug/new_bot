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
    # Position sizing
    risk_per_trade_pct: float = 2.0  # % of equity risked per trade
    max_position_pct: float = 90.0  # max % of equity in a single position
    min_notional_usd: float = 5.0  # Binance minimum notional
    # Timeframes
    primary_tf: str = "1h"
    # Signal thresholds
    entry_confidence: float = 0.58
    exit_confidence: float = 0.38  # lower = less aggressive signal exits (let TP work)


@dataclass
class RiskConfig:
    """Risk management parameters."""

    # ATR-based stops
    atr_period: int = 14
    sl_atr_mult: float = 1.2
    tp_atr_mult: float = 3.0  # R:R = 2.5:1 (wider TP, tighter SL)
    # Equity-curve circuit breaker
    max_drawdown_pct: float = 25.0  # kill-switch threshold
    max_daily_loss_pct: float = 10.0
    max_consecutive_losses: int = 5
    # Flash-crash protection
    flash_crash_pct: float = 8.0  # single-candle move threshold
    cooldown_candles: int = 3  # pause after flash crash detected
    # Trailing stop
    trailing_activate_pct: float = 1.5
    trailing_callback_pct: float = 0.8


@dataclass
class MLConfig:
    """Machine-learning model parameters."""

    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    target_horizon: int = 3  # candles ahead for label
    # LightGBM
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    })
    # Transformer
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_seq_len: int = 48  # input sequence length
    transformer_epochs: int = 30
    transformer_lr: float = 1e-3
    transformer_batch_size: int = 64
    # Walk-forward
    wf_train_size: int = 5000
    wf_test_size: int = 500
    wf_step: int = 500
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
class BotConfig:
    """Root configuration aggregating all sub-configs."""

    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log_level: str = "INFO"
    data_dir: str = "data"
