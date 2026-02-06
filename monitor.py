"""
Monitoring Dashboard and Alerting System.

Provides:
- Real-time performance tracking
- Drawdown alerts
- Strategy degradation detection
- Console-based dashboard
- JSON log analysis
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors and analyzes trading bot performance."""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.LOG_DIR
        self.alerts = []

    def load_trades(self):
        """Load trade history from log file."""
        trade_file = os.path.join(self.log_dir, "trades.json")
        if not os.path.exists(trade_file):
            return []
        try:
            with open(trade_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def load_performance(self):
        """Load performance snapshots."""
        perf_file = os.path.join(self.log_dir, "performance.json")
        if not os.path.exists(perf_file):
            return []
        try:
            with open(perf_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def analyze_trades(self, trades=None):
        """Analyze trade history and return statistics."""
        if trades is None:
            trades = self.load_trades()

        exits = [t for t in trades if t.get("action") == "EXIT" and "pnl" in t]
        if not exits:
            return {"status": "No completed trades yet"}

        pnls = [t["pnl"] for t in exits]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total = len(pnls)
        win_rate = len(winners) / total if total > 0 else 0
        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0
        profit_factor = (
            sum(winners) / abs(sum(losers))
            if losers and sum(losers) != 0
            else float("inf")
        )

        # Consecutive losses
        max_consec = 0
        current = 0
        for p in pnls:
            if p <= 0:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0

        # Equity curve from PnL
        equity = [0]
        for p in pnls:
            equity.append(equity[-1] + p)

        peak = 0
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            if peak > 0:
                dd = (peak - e) / peak
                max_dd = max(max_dd, dd)

        return {
            "total_trades": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": sum(pnls),
            "max_drawdown": max_dd,
            "max_consecutive_losses": max_consec,
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "sharpe_estimate": (
                np.mean(pnls) / np.std(pnls) * np.sqrt(365 * 24)
                if np.std(pnls) > 0
                else 0
            ),
        }

    def check_alerts(self, stats):
        """Check for alert conditions and return active alerts."""
        alerts = []

        if stats.get("max_drawdown", 0) >= config.MONITOR["alert_on_drawdown"]:
            alerts.append({
                "level": "WARNING",
                "message": f"Drawdown at {stats['max_drawdown']:.1%} "
                           f"(threshold: {config.MONITOR['alert_on_drawdown']:.1%})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if (
            stats.get("max_consecutive_losses", 0)
            >= config.MONITOR["alert_on_consecutive_loss"]
        ):
            alerts.append({
                "level": "WARNING",
                "message": f"Consecutive losses: {stats['max_consecutive_losses']} "
                           f"(threshold: {config.MONITOR['alert_on_consecutive_loss']})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Strategy degradation: win rate dropping significantly
        if stats.get("total_trades", 0) >= 20:
            if stats.get("win_rate", 0) < 0.40:
                alerts.append({
                    "level": "CRITICAL",
                    "message": f"Win rate degraded to {stats['win_rate']:.1%} "
                               f"(expected >50%)",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            if stats.get("profit_factor", 0) < 1.0:
                alerts.append({
                    "level": "CRITICAL",
                    "message": f"Profit factor below 1.0: {stats['profit_factor']:.2f}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        self.alerts = alerts
        return alerts

    def print_dashboard(self):
        """Print a console-based performance dashboard."""
        stats = self.analyze_trades()
        perf = self.load_performance()

        print("\n" + "=" * 60)
        print("       TRADING BOT PERFORMANCE DASHBOARD")
        print("=" * 60)

        if "status" in stats:
            print(f"\n  {stats['status']}")
        else:
            print(f"\n  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate:     {stats['win_rate']:.1%} "
                  f"({stats['winners']}W / {stats['losers']}L)")
            print(f"  Total PnL:    ${stats['total_pnl']:.4f}")
            print(f"  Avg Win:      ${stats['avg_win']:.4f}")
            print(f"  Avg Loss:     ${stats['avg_loss']:.4f}")
            print(f"  Profit Factor:{stats['profit_factor']:.2f}")
            print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
            print(f"  Sharpe Est:   {stats['sharpe_estimate']:.2f}")
            print(f"  Max Consec L: {stats['max_consecutive_losses']}")
            print(f"  Best Trade:   ${stats['best_trade']:.4f}")
            print(f"  Worst Trade:  ${stats['worst_trade']:.4f}")

        # Latest performance snapshot
        if perf:
            latest = perf[-1]
            print(f"\n  --- Latest Snapshot ---")
            print(f"  Balance:    ${latest.get('balance', 0):.2f}")
            print(f"  Tier:       {latest.get('tier', 'N/A')}")
            print(f"  Drawdown:   {latest.get('current_drawdown', 0):.1%}")
            print(f"  Daily PnL:  ${latest.get('daily_pnl', 0):.4f}")
            print(f"  Halted:     {latest.get('is_halted', False)}")
            if latest.get("open_positions"):
                print(f"  Positions:  {len(latest['open_positions'])}")
                for p in latest["open_positions"]:
                    print(f"    {p['symbol']} {p['side']} "
                          f"qty={p['quantity']} @ ${p['entry_price']}")

        # Alerts
        if "status" not in stats:
            alerts = self.check_alerts(stats)
            if alerts:
                print(f"\n  --- ALERTS ---")
                for alert in alerts:
                    print(f"  [{alert['level']}] {alert['message']}")
            else:
                print(f"\n  No active alerts.")

        print("\n" + "=" * 60)

    def run_live_dashboard(self, refresh_interval=60):
        """Run continuously refreshing dashboard."""
        print("Starting live dashboard (Ctrl+C to stop)...")
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                self.print_dashboard()
                print(f"\n  Next refresh in {refresh_interval}s...")
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


def main():
    """Run the monitoring dashboard."""
    monitor = PerformanceMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        monitor.run_live_dashboard(interval)
    else:
        monitor.print_dashboard()


if __name__ == "__main__":
    main()
