# Binance USDT-M Futures Trading Bot

Autonomous ML-powered trading bot for Binance USDT-Margined Futures, optimised for micro-balance accounts ($10+).

Uses LightGBM gradient boosting with walk-forward optimisation to generate directional signals on XRP/USDT 1-hour candles, with ATR-based dynamic risk management and Telegram notifications.

## Walk-Forward Backtest Results

| Metric | Value |
|---|---|
| Starting balance | $10.00 |
| Final equity | $25.22 |
| Peak equity | $36.12 |
| Return | +152% |
| Profit factor | 1.28 |
| Sharpe ratio | 9.53 |
| Win rate | 52.3% |
| Total trades | 344 |
| Max drawdown | 30.2% |
| Avg hold time | 7.3 hours |

## Project Structure

```
new_bot/
├── bot/
│   ├── core/
│   │   ├── bot.py              # Main autonomous trading loop
│   │   ├── config.py           # All configuration (trading, risk, ML, Telegram)
│   │   ├── features.py         # 54-feature engineering pipeline
│   │   ├── position_sizer.py   # Binance-compliant position sizing
│   │   └── risk.py             # Risk manager (SL/TP, kill-switch, drawdown scaling)
│   ├── ml/
│   │   └── models.py           # LightGBM + Transformer ensemble
│   ├── exchange/
│   │   └── client.py           # Binance Futures REST API client
│   ├── backtest/
│   │   └── engine.py           # Walk-forward backtest engine
│   └── utils/
│       ├── logger.py           # Structured logging
│       └── telegram.py         # Telegram notification service
├── run_backtest.py             # Run walk-forward backtest on historical data
├── run_bot.py                  # Launch live trading bot
├── requirements.txt
├── .gitignore
└── XRPUSDT_2022_2026.csv      # Historical data (1h OHLCV)
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

**Note:** PyTorch (`torch`) is optional. Without it, the bot uses LightGBM only (which performed best in backtests). Install it only if you want the Transformer ensemble component.

### 2. Run Backtest (Train & Validate Model)

```bash
python run_backtest.py --data XRPUSDT_2022_2026.csv --balance 10 --leverage 10
```

This runs walk-forward optimisation across the full dataset:
- Trains LightGBM on expanding windows (5000+ candles)
- Tests on out-of-sample 1000-candle segments
- Selects top 30 features via importance ranking
- Outputs results to `results/` directory

**Optional:** Add `--cv` for purged K-Fold cross-validation:

```bash
python run_backtest.py --data XRPUSDT_2022_2026.csv --cv
```

Results are saved to:
- `results/backtest_metrics.json` — performance metrics
- `results/equity_curve.csv` — equity over time
- `results/trades.csv` — individual trade log

### 3. Set Up Telegram Notifications (Optional)

1. **Create a Telegram bot:**
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` and follow the prompts
   - Copy the **bot token** (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

2. **Get your chat ID:**
   - Start a chat with your new bot (send any message)
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Find `"chat":{"id":123456789}` in the response — that number is your chat ID

3. **Set environment variables:**

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export TELEGRAM_CHAT_ID="123456789"
```

The bot sends notifications for:
- Position opened (side, price, SL/TP, signal strength, equity)
- Position closed (reason, equity)
- Kill-switch activated (drawdown threshold breached)
- Errors (API failures, unexpected exceptions)
- Daily summary (PnL, trade count, drawdown)
- Bot startup/shutdown

### 4. Run Live Trading

```bash
# Set Binance API credentials
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# Optional: Telegram notifications
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Train model on historical data and start trading
python run_bot.py --train-first --data XRPUSDT_2022_2026.csv

# Or start with a pre-trained model (advanced)
python run_bot.py --symbol XRPUSDT --leverage 10
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--symbol` | `XRPUSDT` | Trading pair |
| `--leverage` | `10` | Leverage multiplier |
| `--train-first` | off | Train model before going live |
| `--data` | `XRPUSDT_2022_2026.csv` | CSV for training |
| `--no-telegram` | off | Disable Telegram notifications |

### 5. Run as a Background Service

```bash
# Using nohup
nohup python run_bot.py --train-first --data XRPUSDT_2022_2026.csv > bot.log 2>&1 &

# Using screen
screen -S trading_bot
python run_bot.py --train-first --data XRPUSDT_2022_2026.csv
# Detach: Ctrl+A, then D
# Reattach: screen -r trading_bot

# Using systemd (create /etc/systemd/system/trading-bot.service)
# [Unit]
# Description=Binance Trading Bot
# After=network.target
#
# [Service]
# Type=simple
# User=your_user
# WorkingDirectory=/path/to/new_bot
# EnvironmentFile=/path/to/new_bot/.env
# ExecStart=/path/to/venv/bin/python run_bot.py --train-first
# Restart=on-failure
# RestartSec=30
#
# [Install]
# WantedBy=multi-user.target
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `BINANCE_API_KEY` | Yes (live) | Binance Futures API key |
| `BINANCE_API_SECRET` | Yes (live) | Binance Futures API secret |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | No | Your Telegram chat ID |

You can put these in a `.env` file (already in `.gitignore`):

```bash
# .env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Then load them before running:

```bash
export $(cat .env | xargs) && python run_bot.py --train-first
```

## How It Works

### ML Pipeline

1. **Feature Engineering** (`bot/core/features.py`) — 54 features:
   - Price action: RSI, MACD, Bollinger Bands, ADX, Stochastic
   - Order flow proxies: taker buy ratio, volume delta, trade intensity, VWAP
   - Volatility: Garman-Klass, realised vol ratio, ATR
   - Momentum: multi-horizon log returns, Z-scores
   - Cyclical time encoding (hour-of-day, day-of-week)

2. **Label Generation** — Threshold-filtered binary labels:
   - Looks 6 candles ahead (6 hours)
   - Only labels candles where the move exceeds 1% (filters noise)
   - Neutral moves are excluded from training

3. **Model Training** — LightGBM gradient boosting:
   - Walk-forward expanding windows (no look-ahead bias)
   - Feature selection: top 30 by importance
   - Strong regularisation: L1/L2 penalties, min samples, feature/bagging fractions
   - Early stopping on validation AUC

4. **Signal Generation** — Model predicts probability of upward move:
   - `signal >= 0.56` → LONG entry
   - `signal <= 0.44` → SHORT entry
   - Asymmetric: model has stronger edge on SHORT side

### Risk Management

| Parameter | Value | Description |
|---|---|---|
| Risk per trade | 3% | Max loss per position at stop-loss |
| SL distance | 1.5x ATR | Dynamic stop-loss |
| TP distance | 2.5x ATR | Dynamic take-profit (R:R = 1.67) |
| Max hold | 18 candles | Force exit after 18 hours |
| Kill-switch | 30% DD | Hard stop on all trading |
| Soft scaling | 15% DD | Start reducing position sizes |
| Daily loss limit | 10% | Max daily drawdown |
| Consecutive losses | 8 | Pause after 8 losses in a row |
| Flash crash | 15% move | Cooldown after extreme candle |
| Trailing stop | 1.5% activate, 0.6% callback | Lock in profits on winners |

### Position Sizing

Risk-based sizing ensures each stop-loss hit costs exactly `risk_per_trade_pct` of equity:

```
notional = risk_amount / (stop_distance / entry_price)
```

Additional constraints:
- Max position capped at 90% of equity x leverage
- Minimum notional $5 (Binance requirement)
- Soft drawdown scaling reduces position sizes from 100% to 25% during 15-30% drawdown

## Configuration

All parameters are in `bot/core/config.py` as dataclasses. Key sections:

- `TradingConfig` — symbols, leverage, entry/exit thresholds, risk per trade
- `RiskConfig` — ATR stops, drawdown limits, trailing stop, flash crash
- `MLConfig` — LightGBM hyperparameters, walk-forward windows, feature selection
- `BacktestConfig` — fees, slippage, funding rate estimates
- `TelegramConfig` — bot token, chat ID, enable/disable

To modify parameters, edit the defaults in `config.py` or override them programmatically before calling `run()`.

## Troubleshooting

**"Telegram notifications disabled" warning:**
Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` environment variables, or pass `--no-telegram` to suppress.

**"PyTorch unavailable — using LightGBM only":**
This is fine. The Transformer component is optional and the bot works well with LightGBM alone.

**Backtest shows few trades:**
The model is selective by design. Walk-forward training needs at least 5000 candles before generating predictions. First trades appear after index 5000 (~7 months into the data).

**Kill-switch fires early:**
The 30% drawdown threshold protects capital. If the strategy loses 30% from peak equity, it stops trading. This is intentional — it preserved $25.22 from a $36 peak in backtests.

**Minimum notional errors:**
Binance requires at least $5 notional per trade. With a $10 account and 3% risk, the notional is ~$15-20 which is well above minimum. If equity drops below ~$3, trades become infeasible.
