"""
Strategy Lab - Test multiple aggressive strategies head-to-head.
Goal: Find a strategy delivering 16%+ monthly returns.
"""
import logging
import math
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

# ── Data Loading ─────────────────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df

# ── Indicators ───────────────────────────────────────────────────────────

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p, min_periods=p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()

def bollinger(s, p=20, std=2.0):
    mid = sma(s, p)
    sd = s.rolling(p).std()
    return mid, mid + std * sd, mid - std * sd

def macd(s, fast=12, slow=26, signal=9):
    f = ema(s, fast)
    sl = ema(s, slow)
    m = f - sl
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

# ── Fast Backtester ──────────────────────────────────────────────────────

def fast_backtest(df, signals, params, initial_balance=100.0):
    """
    Vectorized-ish backtester for speed.
    signals: Series of 1 (long), -1 (short), 0 (no signal), aligned with df index.
    params: dict with stop_atr, tp_atr, risk_pct, max_leverage, trailing_atr, trailing_step_atr
    """
    balance = initial_balance
    peak = initial_balance
    trades = []
    position = None  # (side, entry, qty, sl, tp, leverage, atr_val, trailing_sl)
    max_dd = 0.0
    equity_curve = [initial_balance]

    stop_m = params["stop_atr"]
    tp_m = params["tp_atr"]
    risk_pct = params["risk_pct"]
    max_lev = params["max_leverage"]
    trail_act = params.get("trailing_act_atr", 999)
    trail_step = params.get("trailing_step_atr", 1.0)
    fee = params.get("fee", 0.0006)  # taker + slippage
    cooldown = params.get("cooldown", 1)
    min_notional = 5.0
    bars_since_trade = cooldown + 1

    atr_vals = df["atr"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    sig_vals = signals.values

    for i in range(len(df)):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            equity_curve.append(balance)
            continue

        bars_since_trade += 1

        # ── Manage open position ──
        if position is not None:
            side, entry, qty, sl, tp, lev, a, tsl = position
            h, l, c = highs[i], lows[i], closes[i]

            closed = False
            exit_price = None

            if side == 1:  # LONG
                if l <= sl:
                    exit_price = sl
                    closed = True
                elif h >= tp:
                    exit_price = tp
                    closed = True
                else:
                    # trailing
                    profit = c - entry
                    if profit >= trail_act * a and tsl is not None:
                        new_tsl = c - trail_step * a
                        if new_tsl > tsl:
                            tsl = new_tsl
                            position = (side, entry, qty, sl, tp, lev, a, tsl)
                    if tsl is not None and l <= tsl:
                        exit_price = tsl
                        closed = True
            else:  # SHORT
                if h >= sl:
                    exit_price = sl
                    closed = True
                elif l <= tp:
                    exit_price = tp
                    closed = True
                else:
                    profit = entry - c
                    if profit >= trail_act * a and tsl is not None:
                        new_tsl = c + trail_step * a
                        if new_tsl < tsl:
                            tsl = new_tsl
                            position = (side, entry, qty, sl, tp, lev, a, tsl)
                    if tsl is not None and h >= tsl:
                        exit_price = tsl
                        closed = True

            if closed:
                if side == 1:
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty
                pnl -= (qty * entry + qty * exit_price) * fee
                balance += pnl
                trades.append(pnl)
                position = None
                bars_since_trade = 0

        # ── New entry ──
        if position is None and bars_since_trade >= cooldown and i < len(df) - 1:
            sig = sig_vals[i]
            if sig != 0 and balance > 5:
                a = atr_vals[i]
                price = closes[i]
                side = int(sig)

                # Position sizing
                stop_dist = a * stop_m
                stop_pct = stop_dist / price
                risk_budget = balance * risk_pct

                # Leverage: enough to make the trade viable, capped
                lev = min(max_lev, max(1, int(risk_pct / stop_pct)))

                # Size
                max_pos = balance * lev * (1 - fee * 2)
                risk_pos = risk_budget / stop_pct if stop_pct > 0 else 0
                pos_usdt = min(max_pos, risk_pos)
                qty = pos_usdt / price if price > 0 else 0

                if qty * price < min_notional:
                    if min_notional / price * price / lev < balance * 0.95:
                        qty = min_notional / price
                    else:
                        equity_curve.append(balance)
                        continue

                if side == 1:
                    sl_price = price - stop_dist
                    tp_price = price + a * tp_m
                else:
                    sl_price = price + stop_dist
                    tp_price = price - a * tp_m

                init_trail = None
                if trail_act < 999:
                    if side == 1:
                        init_trail = price - trail_step * a
                    else:
                        init_trail = price + trail_step * a

                position = (side, price, qty, sl_price, tp_price, lev, a, init_trail)

        # Track equity
        if position is not None:
            side, entry, qty, sl, tp, lev, a, tsl = position
            if side == 1:
                unrealized = (closes[i] - entry) * qty
            else:
                unrealized = (entry - closes[i]) * qty
            equity_curve.append(balance + unrealized)
        else:
            equity_curve.append(balance)

        peak = max(peak, equity_curve[-1])
        if peak > 0:
            dd = (peak - equity_curve[-1]) / peak
            max_dd = max(max_dd, dd)

    # Close remaining
    if position is not None:
        side, entry, qty, sl, tp, lev, a, tsl = position
        exit_price = closes[-1]
        if side == 1:
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty
        pnl -= (qty * entry + qty * exit_price) * fee
        balance += pnl
        trades.append(pnl)

    # Stats
    if not trades:
        return None

    trades = np.array(trades)
    winners = trades[trades > 0]
    losers = trades[trades <= 0]
    n = len(trades)
    wr = len(winners) / n
    avg_w = winners.mean() if len(winners) > 0 else 0
    avg_l = abs(losers.mean()) if len(losers) > 0 else 0
    gross_w = winners.sum() if len(winners) > 0 else 0
    gross_l = abs(losers.sum()) if len(losers) > 0 else 0
    pf = gross_w / gross_l if gross_l > 0 else 999

    eq = np.array(equity_curve)
    if len(eq) > 1:
        rets = np.diff(eq) / eq[:-1]
        rets = rets[np.isfinite(rets)]
        sharpe = (rets.mean() / rets.std()) * np.sqrt(8760) if rets.std() > 0 else 0
    else:
        sharpe = 0

    # Monthly returns
    total_hours = len(df)
    total_months = total_hours / (24 * 30.44)
    if total_months > 0 and balance > 0 and initial_balance > 0:
        monthly_return = (balance / initial_balance) ** (1 / total_months) - 1
    else:
        monthly_return = 0

    # Consecutive losses
    max_consec = 0
    cur = 0
    for t in trades:
        if t <= 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    return {
        "balance": balance,
        "return": (balance / initial_balance - 1),
        "monthly_return": monthly_return,
        "trades": n,
        "win_rate": wr,
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "rr": avg_w / avg_l if avg_l > 0 else 999,
        "pf": pf,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "max_consec_loss": max_consec,
    }


# ── Strategy Signal Generators ───────────────────────────────────────────

def strategy_momentum_breakout(df):
    """
    Breakout strategy: enter when price breaks above/below Bollinger Bands
    with volume surge, in direction of EMA trend.
    """
    df = df.copy()
    df["ema_fast"] = ema(df["close"], 10)
    df["ema_slow"] = ema(df["close"], 30)
    df["atr"] = atr(df["high"], df["low"], df["close"], 10)
    df["rsi"] = rsi(df["close"], 10)
    mid, upper, lower = bollinger(df["close"], 20, 2.0)
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["vol_ma"] = sma(df["volume"], 15)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    signals = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        if np.isnan(df["atr"].iloc[i]):
            continue
        c = df["close"].iloc[i]
        prev_c = df["close"].iloc[i-1]
        # Long: close above upper BB, trend up, volume surge
        if (c > df["bb_upper"].iloc[i] and
            prev_c <= df["bb_upper"].iloc[i-1] and
            df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i] and
            df["vol_ratio"].iloc[i] > 1.3 and
            df["rsi"].iloc[i] > 55 and df["rsi"].iloc[i] < 85):
            signals.iloc[i] = 1
        # Short: close below lower BB, trend down, volume surge
        elif (c < df["bb_lower"].iloc[i] and
              prev_c >= df["bb_lower"].iloc[i-1] and
              df["ema_fast"].iloc[i] < df["ema_slow"].iloc[i] and
              df["vol_ratio"].iloc[i] > 1.3 and
              df["rsi"].iloc[i] < 45 and df["rsi"].iloc[i] > 15):
            signals.iloc[i] = -1

    return signals, df

def strategy_macd_momentum(df):
    """
    MACD histogram reversal with trend and volume.
    Enter when MACD histogram flips direction with EMA trend confirmation.
    """
    df = df.copy()
    df["ema_fast"] = ema(df["close"], 8)
    df["ema_slow"] = ema(df["close"], 21)
    df["atr"] = atr(df["high"], df["low"], df["close"], 10)
    m, sig, hist = macd(df["close"], 8, 21, 5)
    df["macd_hist"] = hist
    df["vol_ma"] = sma(df["volume"], 15)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)
    df["rsi"] = rsi(df["close"], 10)

    signals = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        if np.isnan(df["atr"].iloc[i]):
            continue
        h = df["macd_hist"].iloc[i]
        ph = df["macd_hist"].iloc[i-1]
        # Long: MACD hist turns positive, EMA trend up
        if (h > 0 and ph <= 0 and
            df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i] and
            df["vol_ratio"].iloc[i] > 0.9 and
            df["rsi"].iloc[i] > 45):
            signals.iloc[i] = 1
        # Short: MACD hist turns negative, EMA trend down
        elif (h < 0 and ph >= 0 and
              df["ema_fast"].iloc[i] < df["ema_slow"].iloc[i] and
              df["vol_ratio"].iloc[i] > 0.9 and
              df["rsi"].iloc[i] < 55):
            signals.iloc[i] = -1

    return signals, df

def strategy_rsi_extreme_bounce(df):
    """
    Aggressive RSI extremes with fast EMA trend.
    Enter on RSI < 25 bounce in uptrend (or RSI > 75 rejection in downtrend).
    """
    df = df.copy()
    df["ema_fast"] = ema(df["close"], 9)
    df["ema_slow"] = ema(df["close"], 21)
    df["atr"] = atr(df["high"], df["low"], df["close"], 10)
    df["rsi"] = rsi(df["close"], 7)  # Faster RSI
    df["vol_ma"] = sma(df["volume"], 15)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    signals = pd.Series(0, index=df.index)

    for i in range(3, len(df)):
        if np.isnan(df["atr"].iloc[i]):
            continue
        r = df["rsi"].iloc[i]
        pr = df["rsi"].iloc[i-1]
        pr2 = df["rsi"].iloc[i-2]
        trend_up = df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i]
        bullish = df["close"].iloc[i] > df["open"].iloc[i]
        bearish = df["close"].iloc[i] < df["open"].iloc[i]

        # Long: RSI was < 25 recently, now recovering, uptrend
        if trend_up and (pr <= 25 or pr2 <= 25) and r > pr and bullish:
            signals.iloc[i] = 1
        # Short: RSI was > 75 recently, now declining, downtrend
        elif not trend_up and (pr >= 75 or pr2 >= 75) and r < pr and bearish:
            signals.iloc[i] = -1

    return signals, df

def strategy_ema_crossover_aggressive(df):
    """
    Fast EMA crossover with aggressive sizing.
    Cross of EMA 5 over EMA 13, confirmed by EMA 30 trend.
    """
    df = df.copy()
    df["ema5"] = ema(df["close"], 5)
    df["ema13"] = ema(df["close"], 13)
    df["ema30"] = ema(df["close"], 30)
    df["atr"] = atr(df["high"], df["low"], df["close"], 10)
    df["rsi"] = rsi(df["close"], 10)
    df["vol_ma"] = sma(df["volume"], 15)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    signals = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        if np.isnan(df["atr"].iloc[i]):
            continue
        cross_up = df["ema5"].iloc[i] > df["ema13"].iloc[i] and df["ema5"].iloc[i-1] <= df["ema13"].iloc[i-1]
        cross_down = df["ema5"].iloc[i] < df["ema13"].iloc[i] and df["ema5"].iloc[i-1] >= df["ema13"].iloc[i-1]

        if cross_up and df["close"].iloc[i] > df["ema30"].iloc[i] and df["vol_ratio"].iloc[i] > 0.8:
            signals.iloc[i] = 1
        elif cross_down and df["close"].iloc[i] < df["ema30"].iloc[i] and df["vol_ratio"].iloc[i] > 0.8:
            signals.iloc[i] = -1

    return signals, df

def strategy_multiconfirm(df):
    """
    Multi-indicator confirmation: EMA trend + MACD + RSI + Volume all agree.
    Fewer trades but higher conviction.
    """
    df = df.copy()
    df["ema8"] = ema(df["close"], 8)
    df["ema21"] = ema(df["close"], 21)
    df["atr"] = atr(df["high"], df["low"], df["close"], 10)
    df["rsi"] = rsi(df["close"], 10)
    m, sig, hist = macd(df["close"], 8, 21, 5)
    df["macd_hist"] = hist
    df["vol_ma"] = sma(df["volume"], 15)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    signals = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        if np.isnan(df["atr"].iloc[i]):
            continue
        trend_up = df["ema8"].iloc[i] > df["ema21"].iloc[i]
        macd_bull = df["macd_hist"].iloc[i] > 0
        rsi_ok_l = 40 < df["rsi"].iloc[i] < 70
        rsi_ok_s = 30 < df["rsi"].iloc[i] < 60
        vol_ok = df["vol_ratio"].iloc[i] > 1.0
        bullish = df["close"].iloc[i] > df["open"].iloc[i]
        bearish = df["close"].iloc[i] < df["open"].iloc[i]

        if trend_up and macd_bull and rsi_ok_l and vol_ok and bullish:
            signals.iloc[i] = 1
        elif not trend_up and not macd_bull and rsi_ok_s and vol_ok and bearish:
            signals.iloc[i] = -1

    return signals, df


# ── Main: Run All Strategies with Multiple Param Sets ────────────────────

def main():
    data_dir = os.path.dirname(__file__)
    pairs = {}
    for sym in ["XRPUSDT", "TRXUSDT"]:
        fp = os.path.join(data_dir, f"{sym}_2022_2026.csv")
        if os.path.exists(fp):
            pairs[sym] = load_data(fp)
            print(f"Loaded {sym}: {len(pairs[sym])} bars")

    strategies = {
        "BB_Breakout": strategy_momentum_breakout,
        "MACD_Momentum": strategy_macd_momentum,
        "RSI_Extreme": strategy_rsi_extreme_bounce,
        "EMA_Cross": strategy_ema_crossover_aggressive,
        "Multi_Confirm": strategy_multiconfirm,
    }

    param_sets = [
        {"name": "Tight_Aggressive",  "stop_atr": 1.0, "tp_atr": 2.5, "risk_pct": 0.04, "max_leverage": 20, "trailing_act_atr": 1.5, "trailing_step_atr": 0.6, "cooldown": 1, "fee": 0.0006},
        {"name": "Medium",            "stop_atr": 1.2, "tp_atr": 3.0, "risk_pct": 0.03, "max_leverage": 15, "trailing_act_atr": 2.0, "trailing_step_atr": 0.8, "cooldown": 2, "fee": 0.0006},
        {"name": "Wide_HighRisk",     "stop_atr": 1.5, "tp_atr": 4.0, "risk_pct": 0.05, "max_leverage": 20, "trailing_act_atr": 2.5, "trailing_step_atr": 1.0, "cooldown": 1, "fee": 0.0006},
        {"name": "Scalp",             "stop_atr": 0.7, "tp_atr": 1.5, "risk_pct": 0.04, "max_leverage": 20, "trailing_act_atr": 1.0, "trailing_step_atr": 0.4, "cooldown": 1, "fee": 0.0006},
        {"name": "Trend_Ride",        "stop_atr": 1.5, "tp_atr": 5.0, "risk_pct": 0.03, "max_leverage": 15, "trailing_act_atr": 2.0, "trailing_step_atr": 1.0, "cooldown": 3, "fee": 0.0006},
        {"name": "Ultra_Aggressive",  "stop_atr": 0.8, "tp_atr": 2.0, "risk_pct": 0.06, "max_leverage": 20, "trailing_act_atr": 1.2, "trailing_step_atr": 0.5, "cooldown": 1, "fee": 0.0006},
    ]

    print("\n" + "=" * 120)
    print(f"{'Strategy':<18} {'Params':<20} {'Symbol':<10} {'Bal $100→':<12} {'Monthly':<10} "
          f"{'Trades':<8} {'WR':<8} {'PF':<8} {'RR':<8} {'MaxDD':<8} {'Sharpe':<8} {'MaxCL':<6}")
    print("=" * 120)

    best_monthly = -999
    best_combo = None

    for sname, sfunc in strategies.items():
        for sym, df in pairs.items():
            sigs, df_ind = sfunc(df)
            for ps in param_sets:
                result = fast_backtest(df_ind, sigs, ps, initial_balance=100.0)
                if result is None:
                    continue
                mr = result["monthly_return"]
                tag = ""
                if mr > best_monthly:
                    best_monthly = mr
                    best_combo = (sname, ps["name"], sym, result)
                    tag = " ***"

                if result["trades"] >= 20:  # Only show meaningful results
                    print(f"{sname:<18} {ps['name']:<20} {sym:<10} "
                          f"${result['balance']:<11.2f} {mr:<9.1%} "
                          f"{result['trades']:<8} {result['win_rate']:<7.1%} "
                          f"{result['pf']:<7.2f} {result['rr']:<7.2f} "
                          f"{result['max_dd']:<7.1%} {result['sharpe']:<7.2f} "
                          f"{result['max_consec_loss']:<6}{tag}")

    # Combined-pair tests for best strategies
    print("\n" + "=" * 120)
    print("COMBINED PAIR TESTS (best combos)")
    print("=" * 120)

    for sname, sfunc in strategies.items():
        for ps in param_sets:
            total_balance = 100.0
            all_trades = []
            combined_monthly = []

            for sym, df in pairs.items():
                sigs, df_ind = sfunc(df)
                # Each pair gets half the capital
                result = fast_backtest(df_ind, sigs, ps, initial_balance=50.0)
                if result is None:
                    continue
                total_balance += (result["balance"] - 50.0)
                combined_monthly.append(result["monthly_return"])

            if combined_monthly:
                avg_monthly = np.mean(combined_monthly)
                total_ret = total_balance / 100.0 - 1
                total_months = len(list(pairs.values())[0]) / (24 * 30.44)
                if total_months > 0 and total_balance > 0:
                    actual_monthly = (total_balance / 100.0) ** (1 / total_months) - 1
                else:
                    actual_monthly = 0

                if actual_monthly > 0.05:  # Only show promising combos
                    print(f"{sname:<18} {ps['name']:<20} Combined  "
                          f"${total_balance:<11.2f} {actual_monthly:<9.1%}")

    print("\n" + "=" * 120)
    if best_combo:
        sn, pn, sym, r = best_combo
        print(f"BEST: {sn} + {pn} on {sym}")
        print(f"  Monthly Return: {r['monthly_return']:.1%}")
        print(f"  Total Return:   {r['return']:.1%}")
        print(f"  Balance:        ${r['balance']:.2f}")
        print(f"  Win Rate:       {r['win_rate']:.1%}")
        print(f"  Profit Factor:  {r['pf']:.2f}")
        print(f"  Max Drawdown:   {r['max_dd']:.1%}")
        print(f"  Trades:         {r['trades']}")
    print("=" * 120)


if __name__ == "__main__":
    main()
