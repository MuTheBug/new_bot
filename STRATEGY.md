# Adaptive Trend-Momentum (ATM) Trading Strategy

## Strategy Overview

The ATM strategy exploits mean-reversion within established trends on Binance USDT-M Perpetual Futures. It identifies trend direction using dual EMA alignment, then enters positions when RSI reaches extreme levels within that trend, betting on the continuation of the trend after temporary pullbacks.

## Core Edge

Cryptocurrency markets exhibit two well-documented properties:
1. **Trend persistence** - prices tend to continue in the direction of an established trend
2. **Mean reversion within trends** - oversold conditions in uptrends and overbought conditions in downtrends tend to resolve in the trend's direction

The strategy captures this by:
- **Only trading with the trend** (EMA 21 > EMA 55 = uptrend, and vice versa)
- **Entering at extremes** (RSI <= 30 in uptrend = oversold bounce opportunity)
- **Confirming with price action** (bullish/bearish candle confirmation)
- **Using volume as validation** (moves with volume are more reliable)

## Signal Logic

### Long Entry (all conditions required):
1. **Uptrend**: EMA(21) > EMA(55)
2. **Oversold**: RSI(14) was at or below 30 within last 2 bars
3. **Recovery**: RSI is now rising (current > previous)
4. **Confirmation**: Current candle is bullish (close > open)
5. **Volume**: Above 0.8x 20-period average

### Short Entry (mirror):
1. **Downtrend**: EMA(21) < EMA(55)
2. **Overbought**: RSI(14) was at or above 70 within last 2 bars
3. **Decline**: RSI is now falling
4. **Confirmation**: Current candle is bearish
5. **Volume**: Above 0.8x average

### Exit Logic:
- **Stop Loss**: 2.0x ATR(14) from entry
- **Take Profit**: 3.5x ATR(14) from entry (1:1.75 Risk/Reward)
- **Trailing Stop**: Activates after 2.0x ATR profit, trails by 1.0x ATR
- **Cooldown**: Minimum 6 bars between trades per symbol

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Fast EMA | 21 | Standard short-term trend measure |
| Trend EMA | 55 | Intermediate trend direction |
| RSI Period | 14 | Standard momentum oscillator |
| RSI Oversold | 30 | Classic oversold threshold |
| RSI Overbought | 70 | Classic overbought threshold |
| ATR Period | 14 | Volatility-based stop/TP sizing |
| Stop Loss | 2.0x ATR | Room for normal volatility |
| Take Profit | 3.5x ATR | Positive expectancy target |
| Trailing Activation | 2.0x ATR | Lock in profits on runners |
| Volume Filter | 0.8x avg | Minimal filter, avoids dead markets |
| Min Bar Cooldown | 6 | Prevents overtrading |

**Total tunable parameters: 11** - intentionally minimal to reduce overfitting risk.

## Backtest Results (2022-2026, 4 years, 1H data)

### Combined (XRPUSDT + TRXUSDT)

| Metric | $10 Start | $100 Start | $1000 Start |
|--------|-----------|------------|-------------|
| Final Balance | $12.01 | $115.93 | $1,159.63 |
| Net Return | 20.1% | 15.9% | 16.0% |
| Total Trades | 68 | 68 | 68 |
| Win Rate | 55.9% | 55.9% | 55.9% |
| Profit Factor | 1.42 | 1.35 | 1.35 |
| Max Drawdown | 11.0% | 10.2% | 10.2% |
| Sharpe Ratio | 0.66 | 0.59 | 0.59 |
| Max Consec. Losses | 5 | 5 | 5 |
| Avg Trade Duration | 19h 7m | 19h 7m | 19h 7m |

### Per-Symbol ($100 start)

| Metric | TRXUSDT | XRPUSDT |
|--------|---------|---------|
| Net Return | **18.3%** | -1.4% |
| Win Rate | **59.1%** | 50.0% |
| Profit Factor | **1.68** | 0.92 |
| Max Drawdown | **6.1%** | 8.2% |
| Sharpe Ratio | **0.88** | -0.07 |

### In-Sample vs Out-of-Sample Validation

| Metric | In-Sample | Out-of-Sample | Divergence |
|--------|-----------|---------------|------------|
| Win Rate | 55.8% | 56.2% | 0.9% PASS |
| Profit Factor | 1.36 | 1.33 | 2.4% PASS |
| Max DD | 10.2% | 6.2% | - |

### Walk-Forward Analysis (5 windows)

| Window | Period | Trades | Win Rate | PF | Return |
|--------|--------|--------|----------|-----|--------|
| 1 | Oct 2022 - Jun 2023 | 15 | 53.3% | 1.37 | +3.9% |
| 2 | Jun 2023 - Feb 2024 | 11 | 72.7% | 1.94 | +3.9% |
| 3 | Feb 2024 - Oct 2024 | 11 | 45.5% | 1.34 | +2.4% |
| 4 | Oct 2024 - Jun 2025 | 14 | 64.3% | 2.14 | +7.9% |
| 5 | Jun 2025 - Jan 2026 | 6 | 33.3% | 0.44 | -3.5% |

**4 of 5 walk-forward windows profitable.**

### Monte Carlo Simulation (1000 iterations, bootstrap)

| Starting Balance | Median Final | 5th Pctile | 95th Pctile | P(Profit) |
|-----------------|--------------|------------|-------------|-----------|
| $10 | $13.65 | $9.52 | $19.53 | 92.1% |
| $100 | $133.76 | $95.00 | $192.29 | 91.7% |
| $1000 | $1,354.47 | $946.72 | $1,996.13 | 91.9% |

## Risk Management

### Position Sizing
- Maximum 1.5% of balance risked per trade
- Position size = (risk_budget / stop_distance) capped by margin
- Respects Binance minimum notional and quantity requirements
- After 3 consecutive losses, position size halved

### Dynamic Leverage
| Balance Range | Max Leverage |
|--------------|-------------|
| $10 - $49 | 20x |
| $50 - $199 | 15x |
| $200 - $999 | 10x |
| $1,000 - $4,999 | 7x |
| $5,000+ | 5x |

### Circuit Breakers
- **Daily loss limit**: 3% - stops trading until next day
- **Minimum balance**: $5 USDT - full halt

### Algo Order Integration
- Positions with >$1,000 notional use TWAP execution (5-minute duration)
- Smaller positions use standard market orders
- TWAP reduces slippage on larger entries

## File Structure

```
new_bot/
  config.py         - All configuration parameters
  strategy.py       - Signal generation and indicators
  risk_manager.py   - Position sizing, leverage, circuit breakers
  exchange.py       - Binance API client (regular + algo orders)
  backtester.py     - Backtesting engine with walk-forward and Monte Carlo
  bot.py            - Autonomous trading bot (24/7 operation)
  monitor.py        - Performance dashboard and alerting
  requirements.txt  - Python dependencies
```

## Running

### Backtest
```bash
python backtester.py
```

### Live Bot
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export BINANCE_TESTNET="true"  # Use testnet first
python bot.py
```

### Monitor
```bash
python monitor.py          # One-time dashboard
python monitor.py --live   # Continuous refresh
```

## Honest Assessment

The strategy demonstrates a modest but consistent edge:
- ~56% win rate with ~1:1.1 risk/reward ratio yields PF ~1.35
- Performance is driven primarily by TRXUSDT (PF 1.68) while XRPUSDT is near breakeven
- The 4-year 16% return (~4%/year) with 10% max drawdown is a real but modest edge
- Walk-forward validation shows the edge persists out-of-sample
- Monte Carlo confirms ~92% probability of profit over a similar trade sequence

This is not a "get rich quick" system. It is a disciplined, conservative strategy that:
- Preserves capital through strict risk management
- Maintains edge through selective entry criteria
- Avoids overfitting through minimal parameters and robust validation

**Paper trade extensively before risking real capital. Past performance does not guarantee future results.**
