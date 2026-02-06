"""
Optimize the proven RSI mean-reversion strategy.
The edge is real (55.9% WR, PF 1.35). Now we amplify it.

Test matrix:
- Risk per trade: 3%, 5%, 8%, 10%
- Leverage: 10x, 15x, 20x
- Stop/TP: multiple combos
- Signal looseness: strict, moderate, loose
- On both symbols individually and combined
"""
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df


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

def atr_calc(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()


def generate_signals(df, rsi_period=14, rsi_os=30, rsi_ob=70,
                     ema_fast=21, ema_trend=55, cooldown=6,
                     vol_thresh=0.8, atr_period=10, loose=False):
    """Generate RSI mean-reversion signals with adjustable strictness."""
    df = df.copy()
    df["ema_f"] = ema(df["close"], ema_fast)
    df["ema_t"] = ema(df["close"], ema_trend)
    df["rsi"] = rsi(df["close"], rsi_period)
    df["atr"] = atr_calc(df["high"], df["low"], df["close"], atr_period)
    df["vol_ma"] = sma(df["volume"], 20)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)
    df["uptrend"] = df["ema_f"] > df["ema_t"]
    df["downtrend"] = df["ema_f"] < df["ema_t"]
    df["bullish"] = df["close"] > df["open"]
    df["bearish"] = df["close"] < df["open"]

    signals = pd.Series(0, index=df.index)
    last_sig = -cooldown - 1

    rsi_vals = df["rsi"].values
    up = df["uptrend"].values
    dn = df["downtrend"].values
    bull = df["bullish"].values
    bear = df["bearish"].values
    vr = df["vol_ratio"].values
    atr_v = df["atr"].values

    # Looser thresholds for more signals
    os_thresh = rsi_os + (5 if loose else 0)
    ob_thresh = rsi_ob - (5 if loose else 0)

    for i in range(3, len(df)):
        if np.isnan(atr_v[i]) or np.isnan(rsi_vals[i]):
            continue
        if (i - last_sig) < cooldown:
            continue

        vol_ok = vr[i] > vol_thresh

        # LONG
        if up[i]:
            was_os = rsi_vals[i-1] <= os_thresh or rsi_vals[i-2] <= os_thresh
            recovering = rsi_vals[i] > rsi_vals[i-1]
            if loose:
                # Also allow near-oversold entries
                was_os = was_os or rsi_vals[i] <= os_thresh + 3
            if was_os and recovering and bull[i] and vol_ok:
                signals.iloc[i] = 1
                last_sig = i
                continue
            # EMA alignment just formed
            if up[i] and not up[i-1]:
                if 40 < rsi_vals[i] < 60 and vol_ok:
                    signals.iloc[i] = 1
                    last_sig = i
                    continue

        # SHORT
        if dn[i]:
            was_ob = rsi_vals[i-1] >= ob_thresh or rsi_vals[i-2] >= ob_thresh
            declining = rsi_vals[i] < rsi_vals[i-1]
            if loose:
                was_ob = was_ob or rsi_vals[i] >= ob_thresh - 3
            if was_ob and declining and bear[i] and vol_ok:
                signals.iloc[i] = -1
                last_sig = i
                continue
            if dn[i] and not dn[i-1]:
                if 40 < rsi_vals[i] < 60 and vol_ok:
                    signals.iloc[i] = -1
                    last_sig = i
                    continue

    return signals, df


def fast_backtest(df, signals, stop_atr, tp_atr, risk_pct, max_lev,
                  trail_act, trail_step, fee=0.0006, initial_balance=100.0):
    """Fast backtest with compounding."""
    balance = initial_balance
    peak_bal = initial_balance
    trades = []
    position = None
    max_dd = 0.0

    atr_v = df["atr"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(len(df)):
        if np.isnan(atr_v[i]) or atr_v[i] <= 0:
            continue

        # Manage position
        if position is not None:
            side, entry, qty, sl, tp, tsl, a = position
            h, l, c = highs[i], lows[i], closes[i]

            exit_price = None
            if side == 1:
                if l <= sl:
                    exit_price = sl
                elif tsl is not None and l <= tsl:
                    exit_price = tsl
                elif h >= tp:
                    exit_price = tp
                else:
                    # Update trailing
                    if (c - entry) >= trail_act * a and tsl is not None:
                        new_tsl = c - trail_step * a
                        if new_tsl > tsl:
                            position = (side, entry, qty, sl, tp, new_tsl, a)
            else:
                if h >= sl:
                    exit_price = sl
                elif tsl is not None and h >= tsl:
                    exit_price = tsl
                elif l <= tp:
                    exit_price = tp
                else:
                    if (entry - c) >= trail_act * a and tsl is not None:
                        new_tsl = c + trail_step * a
                        if new_tsl < tsl:
                            position = (side, entry, qty, sl, tp, new_tsl, a)

            if exit_price is not None:
                if side == 1:
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty
                pnl -= (qty * entry + qty * exit_price) * fee
                balance += pnl
                trades.append(pnl)
                position = None
                if balance <= 5:
                    break

        # New entry
        if position is None and signals.values[i] != 0 and balance > 5:
            sig = int(signals.values[i])
            a = atr_v[i]
            price = closes[i]

            stop_dist = a * stop_atr
            stop_pct = stop_dist / price if price > 0 else 0.01
            risk_budget = balance * risk_pct

            lev = min(max_lev, max(1, int(risk_pct / stop_pct)))
            max_pos = balance * lev * (1 - fee * 2)
            risk_pos = risk_budget / stop_pct if stop_pct > 0 else 0
            pos_usdt = min(max_pos, risk_pos)
            qty = pos_usdt / price if price > 0 else 0

            if qty * price < 5.0:
                if 5.0 / price * price / lev < balance * 0.95:
                    qty = 5.0 / price
                else:
                    continue

            if sig == 1:
                sl_p = price - stop_dist
                tp_p = price + a * tp_atr
                init_tsl = price - trail_step * a if trail_act < 100 else None
            else:
                sl_p = price + stop_dist
                tp_p = price - a * tp_atr
                init_tsl = price + trail_step * a if trail_act < 100 else None

            position = (sig, price, qty, sl_p, tp_p, init_tsl, a)

        peak_bal = max(peak_bal, balance)
        if peak_bal > 0:
            dd = (peak_bal - balance) / peak_bal
            max_dd = max(max_dd, dd)

    # Close remaining
    if position is not None:
        side, entry, qty, sl, tp, tsl, a = position
        c = closes[-1]
        pnl = ((c - entry) if side == 1 else (entry - c)) * qty
        pnl -= (qty * entry + qty * c) * fee
        balance += pnl
        trades.append(pnl)

    if not trades:
        return None

    trades = np.array(trades)
    winners = trades[trades > 0]
    losers = trades[trades <= 0]
    n = len(trades)
    wr = len(winners) / n if n > 0 else 0
    pf = winners.sum() / abs(losers.sum()) if len(losers) > 0 and losers.sum() != 0 else 999

    total_hours = len(df)
    total_months = total_hours / (24 * 30.44)
    total_years = total_hours / (24 * 365.25)
    if total_months > 0 and balance > 0 and initial_balance > 0:
        monthly_ret = (balance / initial_balance) ** (1 / total_months) - 1
    else:
        monthly_ret = -1
    if total_years > 0 and balance > 0 and initial_balance > 0:
        annual_ret = (balance / initial_balance) ** (1 / total_years) - 1
    else:
        annual_ret = -1

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
        "total_return": balance / initial_balance - 1,
        "monthly_return": monthly_ret,
        "annual_return": annual_ret,
        "trades": n,
        "win_rate": wr,
        "pf": pf,
        "max_dd": max_dd,
        "max_consec_loss": max_consec,
    }


def main():
    data_dir = os.path.dirname(__file__)
    pairs = {}
    for sym in ["XRPUSDT", "TRXUSDT"]:
        fp = os.path.join(data_dir, f"{sym}_2022_2026.csv")
        if os.path.exists(fp):
            pairs[sym] = load_data(fp)

    # Signal generation configs
    sig_configs = [
        {"name": "Strict_RSI7",  "rsi_period": 7,  "rsi_os": 25, "rsi_ob": 75, "ema_fast": 21, "ema_trend": 55, "cooldown": 4, "vol_thresh": 0.8, "loose": False},
        {"name": "Strict_RSI10", "rsi_period": 10, "rsi_os": 28, "rsi_ob": 72, "ema_fast": 15, "ema_trend": 40, "cooldown": 3, "vol_thresh": 0.7, "loose": False},
        {"name": "Loose_RSI7",   "rsi_period": 7,  "rsi_os": 28, "rsi_ob": 72, "ema_fast": 21, "ema_trend": 55, "cooldown": 2, "vol_thresh": 0.7, "loose": True},
        {"name": "Loose_RSI10",  "rsi_period": 10, "rsi_os": 30, "rsi_ob": 70, "ema_fast": 15, "ema_trend": 40, "cooldown": 2, "vol_thresh": 0.7, "loose": True},
        {"name": "Fast_EMA",     "rsi_period": 7,  "rsi_os": 28, "rsi_ob": 72, "ema_fast": 10, "ema_trend": 30, "cooldown": 2, "vol_thresh": 0.6, "loose": True},
    ]

    # Risk/exit parameter configs
    risk_configs = [
        {"name": "R5_S1.0_T2.5", "risk_pct": 0.05, "max_lev": 20, "stop_atr": 1.0, "tp_atr": 2.5, "trail_act": 1.5, "trail_step": 0.6},
        {"name": "R5_S1.2_T3.0", "risk_pct": 0.05, "max_lev": 20, "stop_atr": 1.2, "tp_atr": 3.0, "trail_act": 2.0, "trail_step": 0.8},
        {"name": "R8_S1.0_T2.0", "risk_pct": 0.08, "max_lev": 20, "stop_atr": 1.0, "tp_atr": 2.0, "trail_act": 1.5, "trail_step": 0.6},
        {"name": "R8_S1.2_T2.5", "risk_pct": 0.08, "max_lev": 20, "stop_atr": 1.2, "tp_atr": 2.5, "trail_act": 1.5, "trail_step": 0.7},
        {"name": "R10_S1.0_T2.0","risk_pct": 0.10, "max_lev": 20, "stop_atr": 1.0, "tp_atr": 2.0, "trail_act": 1.2, "trail_step": 0.5},
        {"name": "R10_S1.5_T3.0","risk_pct": 0.10, "max_lev": 20, "stop_atr": 1.5, "tp_atr": 3.0, "trail_act": 2.0, "trail_step": 0.8},
        {"name": "R3_S2.0_T3.5", "risk_pct": 0.03, "max_lev": 15, "stop_atr": 2.0, "tp_atr": 3.5, "trail_act": 2.0, "trail_step": 1.0},
    ]

    print(f"{'Signals':<16} {'Risk/Exit':<18} {'Symbol':<10} {'$100→':<12} {'Monthly':<10} "
          f"{'Annual':<10} {'Trades':<8} {'WR':<8} {'PF':<8} {'MaxDD':<8} {'MCL':<6}")
    print("=" * 130)

    results = []

    for sc in sig_configs:
        for sym, df in pairs.items():
            sigs, df_ind = generate_signals(
                df, rsi_period=sc["rsi_period"], rsi_os=sc["rsi_os"],
                rsi_ob=sc["rsi_ob"], ema_fast=sc["ema_fast"],
                ema_trend=sc["ema_trend"], cooldown=sc["cooldown"],
                vol_thresh=sc["vol_thresh"], loose=sc["loose"]
            )
            for rc in risk_configs:
                r = fast_backtest(
                    df_ind, sigs, rc["stop_atr"], rc["tp_atr"],
                    rc["risk_pct"], rc["max_lev"],
                    rc["trail_act"], rc["trail_step"]
                )
                if r is None or r["trades"] < 20:
                    continue

                results.append((sc["name"], rc["name"], sym, r))

                flag = ""
                if r["monthly_return"] >= 0.10:
                    flag = " !!!"
                elif r["monthly_return"] >= 0.05:
                    flag = " **"
                elif r["monthly_return"] >= 0.02:
                    flag = " *"

                print(f"{sc['name']:<16} {rc['name']:<18} {sym:<10} "
                      f"${r['balance']:<11.2f} {r['monthly_return']:<9.1%} "
                      f"{r['annual_return']:<9.1%} "
                      f"{r['trades']:<8} {r['win_rate']:<7.1%} "
                      f"{r['pf']:<7.2f} {r['max_dd']:<7.1%} "
                      f"{r['max_consec_loss']:<6}{flag}")

    # Sort by monthly return
    results.sort(key=lambda x: x[3]["monthly_return"], reverse=True)
    print("\n" + "=" * 100)
    print("TOP 15 RESULTS BY MONTHLY RETURN:")
    print("=" * 100)
    for i, (sn, rn, sym, r) in enumerate(results[:15]):
        print(f"  {i+1:>2}. {sn:<16} {rn:<18} {sym:<10} "
              f"Monthly: {r['monthly_return']:.1%}  Annual: {r['annual_return']:.1%}  "
              f"WR: {r['win_rate']:.1%}  PF: {r['pf']:.2f}  DD: {r['max_dd']:.1%}  "
              f"Trades: {r['trades']}")

    # Also sort by Sharpe-like metric (monthly_return / max_dd)
    results_with_ratio = [(s, rn, sym, r, r["monthly_return"] / max(r["max_dd"], 0.01))
                           for s, rn, sym, r in results if r["monthly_return"] > 0]
    results_with_ratio.sort(key=lambda x: x[4], reverse=True)
    print("\n" + "=" * 100)
    print("TOP 15 BY RETURN/RISK RATIO:")
    print("=" * 100)
    for i, (sn, rn, sym, r, ratio) in enumerate(results_with_ratio[:15]):
        print(f"  {i+1:>2}. {sn:<16} {rn:<18} {sym:<10} "
              f"Monthly: {r['monthly_return']:.1%}  DD: {r['max_dd']:.1%}  "
              f"Ratio: {ratio:.2f}  WR: {r['win_rate']:.1%}  Trades: {r['trades']}")


if __name__ == "__main__":
    main()
