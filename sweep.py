"""
Sweep risk parameters on the proven strategy using the real backtester.
Find the maximum sustainable aggression level.
"""
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING)

import config
from backtester import Backtester, load_data

data_dir = config.BACKTEST["data_dir"]
data_dict = {}
for pair in config.TRADING_PAIRS:
    fp = os.path.join(data_dir, f"{pair}_2022_2026.csv")
    if os.path.exists(fp):
        data_dict[pair] = load_data(fp)

# Test matrix
risk_levels = [0.015, 0.025, 0.035, 0.05, 0.06, 0.08]
stop_tp_combos = [
    (2.0, 3.5, "2.0/3.5"),
    (1.5, 3.0, "1.5/3.0"),
    (1.2, 2.5, "1.2/2.5"),
    (1.0, 2.5, "1.0/2.5"),
    (1.5, 4.0, "1.5/4.0"),
]
leverage_caps = [15, 20]

print(f"{'Risk%':<8} {'SL/TP':<10} {'MaxLev':<8} {'$100→':<12} {'Monthly':<10} "
      f"{'Annual':<10} {'Trades':<8} {'WR':<8} {'PF':<8} {'MaxDD':<8}")
print("=" * 100)

best = None
best_monthly = -999

for risk in risk_levels:
    for sl_m, tp_m, st_name in stop_tp_combos:
        for max_lev in leverage_caps:
            # Temporarily override config
            orig_risk = config.RISK["max_risk_per_trade"]
            orig_sl = config.STRATEGY["atr_stop_multiplier"]
            orig_tp = config.STRATEGY["atr_tp_multiplier"]
            orig_lev = config.LEVERAGE["max_leverage"]

            config.RISK["max_risk_per_trade"] = risk
            config.STRATEGY["atr_stop_multiplier"] = sl_m
            config.STRATEGY["atr_tp_multiplier"] = tp_m
            config.LEVERAGE["max_leverage"] = max_lev
            # Update all tier caps too
            for tier in config.LEVERAGE["tiers"]:
                config.LEVERAGE["tiers"][tier] = min(
                    config.LEVERAGE["tiers"][tier], max_lev
                )

            bt = Backtester(initial_balance=100.0)
            results = bt.run(data_dict)

            if "error" not in results:
                mr = results.get("annualized_return", 0)
                monthly = results.get("monthly_return", 0)
                if monthly == 0 and results.get("test_duration_years", 0) > 0:
                    yrs = results["test_duration_years"]
                    if yrs > 0:
                        monthly = (results["final_balance"] / 100.0) ** (1/(yrs*12)) - 1

                flag = ""
                if monthly > best_monthly:
                    best_monthly = monthly
                    best = (risk, st_name, max_lev, results, monthly)
                    flag = " <-- BEST"

                print(f"{risk:<7.1%} {st_name:<10} {max_lev:<8} "
                      f"${results['final_balance']:<11.2f} "
                      f"{monthly:<9.2%} {mr:<9.1%} "
                      f"{results['total_trades']:<8} "
                      f"{results['win_rate']:<7.1%} "
                      f"{results['profit_factor']:<7.2f} "
                      f"{results['max_drawdown']:<7.1%}{flag}")

            # Restore
            config.RISK["max_risk_per_trade"] = orig_risk
            config.STRATEGY["atr_stop_multiplier"] = orig_sl
            config.STRATEGY["atr_tp_multiplier"] = orig_tp
            config.LEVERAGE["max_leverage"] = orig_lev
            config.LEVERAGE["tiers"] = {10: 20, 50: 15, 200: 10, 1000: 7, 5000: 5}

print("\n" + "=" * 100)
if best:
    risk, st, lev, r, mr = best
    print(f"BEST: Risk={risk:.1%}, SL/TP={st}, Lev={lev}")
    print(f"  Monthly: {mr:.2%}")
    print(f"  Annual:  {r.get('annualized_return',0):.1%}")
    print(f"  Balance: ${r['final_balance']:.2f}")
    print(f"  WR:      {r['win_rate']:.1%}")
    print(f"  PF:      {r['profit_factor']:.2f}")
    print(f"  MaxDD:   {r['max_drawdown']:.1%}")
    print(f"  Trades:  {r['total_trades']}")
