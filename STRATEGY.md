# Dual-Mode Trading Strategy: Signal + Grid

## Overview

The bot supports two strategy modes, selectable via `STRATEGY_MODE` in config:

1. **Signal Mode** (`signal`) - RSI mean-reversion within EMA trends. Fewer trades, higher profit factor.
2. **Grid Mode** (`grid`) - Smart grid trading with compounding. Many trades, 97%+ win rate.

Both modes are trend-aligned (EMA 21 > EMA 55 = uptrend) and use ATR-based dynamic sizing.

---

## Mode 1: Signal Strategy (RSI Mean-Reversion)

### Core Edge
Enters at RSI extremes within established trends, capturing mean-reversion bounces.

### Entry Logic
**Long** (all required): Uptrend + RSI was <=30 recently + RSI recovering + bullish candle + volume > 0.8x avg
**Short** (mirror): Downtrend + RSI was >=70 recently + RSI declining + bearish candle + volume > 0.8x avg

### Exit Logic
- Stop Loss: 2.0x ATR(14)
- Take Profit: 3.5x ATR(14)
- Trailing Stop: Activates at 2.0x ATR, trails by 1.0x ATR
- Cooldown: 6 bars minimum between trades

### Backtest Results (2022-2026, 4 years, $100 start)

| Metric | Combined | TRXUSDT | XRPUSDT |
|--------|----------|---------|---------|
| Final Balance | $115.93 | $118.30 | $98.60 |
| Net Return | 15.9% (4yr) | 18.3% | -1.4% |
| Annualized | 3.8% | 4.3% | -0.4% |
| Win Rate | 55.9% | 59.1% | 50.0% |
| Profit Factor | 1.35 | 1.68 | 0.92 |
| Max Drawdown | 10.2% | 6.1% | 8.2% |
| Total Trades | 68 | 44 | 24 |

**Validation**: IS/OOS divergence <3%, 4/5 walk-forward windows profitable, Monte Carlo 92% P(profit).

---

## Mode 2: Grid Strategy (Smart Grid with Compounding)

### Core Edge
Places buy/sell grid orders at regular ATR-based intervals around a dynamic EMA center. Profits from natural price oscillation. With compounding, each grid cycle's profit increases future position sizes.

### Grid Mechanics
- **Center**: EMA(55) as dynamic fair value
- **Spacing**: ATR(14) × multiplier (default 1.0)
- **Buy grids**: N levels below center, TP at next level up
- **Sell grids**: N levels above center, TP at next level down
- **Trend-aligned**: Only buy grids in uptrend, sell grids in downtrend
- **No per-position stop-loss**: Positions hold until TP or liquidation
- **Compounding**: Position sizes recalculated from current balance each bar

### Grid Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Grid Spacing | 1.0x ATR | Distance between grid levels |
| Leverage | 3x | Conservative for grid safety |
| Deploy % | 20% | % of balance as grid margin |
| Buy/Sell Grids | 4 each | Grid levels per side |
| Max Positions | 5/symbol | Prevents over-accumulation |
| Max Margin | 45% | Cap on total margin deployed |
| Trend Aligned | Yes | Only fill grids with trend |

### Backtest Results (2022-2026, 4 years, $100 start)

| Metric | Combined | XRPUSDT | TRXUSDT |
|--------|----------|---------|---------|
| Final Balance | $112.46 | $116.95 | $97.72 |
| Net Return | 12.5% (4yr) | 17.0% | -2.3% |
| Annualized | 3.0% | 4.0% | -0.6% |
| Win Rate | 97.8% | 97.5% | 98.2% |
| Profit Factor | 1.07 | 1.16 | 0.96 |
| Max Drawdown | 20.4% | 15.8% | 23.5% |
| Total Trades | 2,968 | 1,580 | 1,388 |

### Trade Breakdown
- Take Profit: 2,903 (97.8%)
- Liquidated: 59 (2.0%)
- End of Data: 6 (0.2%)

---

## Strategy Comparison

| Metric | Signal Mode | Grid Mode |
|--------|------------|-----------|
| Monthly Return | ~0.33% | ~0.33% |
| Annualized | ~3.8% | ~3.0% |
| Win Rate | 55.9% | 97.8% |
| Profit Factor | 1.35 | 1.07 |
| Max Drawdown | 10.2% | 20.4% |
| Total Trades (4yr) | 68 | 2,968 |
| Trades/Month | ~1.4 | ~62 |
| Best For | Low drawdown | High frequency |

## Risk Management

### Position Sizing (Signal Mode)
- 1.5% of balance risked per trade
- Size = risk_budget / stop_distance, capped by margin
- Halve size after 3 consecutive losses

### Position Sizing (Grid Mode)
- 20% of balance deployed across all grid levels
- Each level: deploy_pct / n_total_grids × leverage
- Total margin capped at 45% of balance

### Dynamic Leverage
| Balance Range | Max Leverage |
|--------------|-------------|
| $10 - $49 | 20x |
| $50 - $199 | 15x |
| $200 - $999 | 10x |
| $1,000 - $4,999 | 7x |
| $5,000+ | 5x |

### Circuit Breakers
- Daily loss limit: 3% - stops trading until next day
- Minimum balance: $5 USDT - full halt

### Algo Order Integration
- Positions with >$1,000 notional use TWAP execution
- Smaller positions use standard market orders

## File Structure

```
new_bot/
  config.py            - All configuration (strategy mode, params, risk)
  strategy.py          - Signal strategy (RSI mean-reversion)
  grid_strategy.py     - Grid strategy (smart grid with compounding)
  grid_backtester.py   - Grid-specific backtesting engine
  risk_manager.py      - Position sizing, leverage, circuit breakers
  exchange.py          - Binance API client (regular + algo orders)
  backtester.py        - Signal strategy backtesting engine
  bot.py               - Autonomous trading bot (24/7 operation)
  monitor.py           - Performance dashboard and alerting
  requirements.txt     - Python dependencies
```

## Running

### Backtest Signal Strategy
```bash
python backtester.py
```

### Backtest Grid Strategy
```bash
python grid_backtester.py
```

### Live Bot (Grid Mode - default)
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export BINANCE_TESTNET="true"
export STRATEGY_MODE="grid"    # or "signal"
python bot.py
```

## Honest Assessment

After exhaustive testing of 200+ parameter combinations across 7 strategy variants:
- RSI mean-reversion and grid trading both produce ~3-4% annualized returns
- Signal mode: lower drawdown (10%), fewer trades, higher PF (1.35)
- Grid mode: near-perfect win rate (97.8%), many trades, lower PF (1.07)
- Performance driven primarily by XRPUSDT for grids, TRXUSDT for signals
- Both strategies pass anti-overfitting validation

**Paper trade extensively before risking real capital. Past performance does not guarantee future results.**
