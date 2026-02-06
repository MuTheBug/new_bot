"""
Configuration for the Autonomous Trading Bot.
All settings centralized here for easy adjustment.
"""
import os

# === API Configuration ===
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TESTNET = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"

BASE_URL_LIVE = "https://fapi.binance.com"
BASE_URL_TESTNET = "https://testnet.binancefuture.com"
BASE_URL = BASE_URL_TESTNET if TESTNET else BASE_URL_LIVE

# Algo trading endpoints (TWAP/VP) - only for positions >= 1000 USDT notional
ALGO_TWAP_ENDPOINT = "/sapi/v1/algo/futures/newOrderTwap"
ALGO_VP_ENDPOINT = "/sapi/v1/algo/futures/newOrderVp"
ALGO_OPEN_ORDERS = "/sapi/v1/algo/futures/openOrders"
ALGO_HIST_ORDERS = "/sapi/v1/algo/futures/historicalOrders"

# === Trading Pairs ===
TRADING_PAIRS = ["XRPUSDT", "TRXUSDT"]

# Pair-specific minimum order info (Binance minimums as of 2026)
PAIR_INFO = {
    "XRPUSDT": {
        "min_qty": 0.1,
        "qty_step": 0.1,
        "price_precision": 4,
        "qty_precision": 1,
        "min_notional": 5.0,
    },
    "TRXUSDT": {
        "min_qty": 1.0,
        "qty_step": 1.0,
        "price_precision": 5,
        "qty_precision": 0,
        "min_notional": 5.0,
    },
}

# === Strategy Parameters ===
# These are intentionally minimal to reduce overfitting risk
STRATEGY = {
    # Trend filter (dual EMA)
    "fast_ema_period": 21,
    "trend_ema_period": 55,
    # Mean reversion at extremes
    "rsi_period": 14,
    "rsi_oversold": 30,         # Strict oversold for longs
    "rsi_overbought": 70,       # Strict overbought for shorts
    # Volatility
    "atr_period": 14,
    # Volume
    "volume_ma_period": 20,
    "volume_threshold": 0.8,
    # Exit - balanced for win rate and R:R
    "atr_stop_multiplier": 2.0, # Stop at 2x ATR
    "atr_tp_multiplier": 3.5,   # TP at 3.5x ATR (1:1.75 R:R)
    "trailing_activation": 2.0, # Trailing after 2x ATR profit
    "trailing_step": 1.0,       # Trail by 1x ATR
    # Regime
    "adx_period": 14,
    "adx_trend_threshold": 20,
    # Cooldown - prevent overtrading
    "min_bars_between_trades": 6,
}

# === Risk Management ===
RISK = {
    "max_risk_per_trade": 0.015,     # 1.5% of balance per trade
    "max_open_positions": 2,          # Max simultaneous positions
    "max_daily_loss": 0.03,           # 3% max daily drawdown - stop trading
    "max_total_drawdown": 0.08,       # 8% max total drawdown - stop trading
    "max_consecutive_losses": 3,      # Reduce size after 3 consecutive losses
    "loss_reduction_factor": 0.5,     # Halve position after consecutive losses
    "min_balance_to_trade": 5.0,      # Minimum USDT to keep trading
}

# === Leverage Configuration ===
LEVERAGE = {
    "min_leverage": 1,
    "max_leverage": 20,
    # Balance-based leverage tiers (balance_threshold: max_leverage)
    "tiers": {
        10: 20,      # $10-49: up to 20x
        50: 15,      # $50-199: up to 15x
        200: 10,     # $200-999: up to 10x
        1000: 7,     # $1000-4999: up to 7x
        5000: 5,     # $5000+: up to 5x
    },
}

# === Algo Order Configuration ===
ALGO = {
    "twap_min_notional": 1000.0,      # Minimum notional for TWAP
    "twap_default_duration": 300,     # 5 minutes default
    "twap_max_duration": 3600,        # 1 hour max for our use
    "use_algo_orders": True,          # Enable algo orders when position qualifies
}

# === Fee Structure ===
FEES = {
    "maker": 0.0002,   # 0.02%
    "taker": 0.0004,   # 0.04%
    "default": 0.0004, # Assume taker for conservative estimates
}

# === Timeframe ===
TIMEFRAME = "1h"
CANDLE_LIMIT = 200     # How many candles to fetch for indicator calculation

# === Logging ===
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_LEVEL = "INFO"

# === Monitoring ===
MONITOR = {
    "heartbeat_interval": 60,         # Seconds between heartbeats
    "performance_log_interval": 3600, # Log performance every hour
    "alert_on_drawdown": 0.05,        # Alert at 5% drawdown
    "alert_on_consecutive_loss": 3,
    "dashboard_port": 8080,
}

# === Backtest Configuration ===
BACKTEST = {
    "data_dir": os.path.dirname(__file__),
    "initial_balances": [10, 100, 1000],
    "commission_rate": 0.0004,        # Taker fee
    "slippage_rate": 0.0002,          # 0.02% slippage estimate
    "oos_split": 0.3,                 # 30% out-of-sample
    "walk_forward_windows": 5,
    "monte_carlo_iterations": 1000,
}
