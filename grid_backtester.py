"""
Smart Grid V4 - Realistic Grid with Liquidation + Trend-Flip.

Fixes from V3:
- Proper liquidation (position closed at loss when unrealized loss >= margin)
- Trend-flip mode: close all positions when trend reverses (prevents accumulation)
- Better position sizing with proper margin accounting
- Tracks margin locked in open positions to prevent over-leveraging

Two modes tested:
1. Hold-until-TP: positions only close at TP or liquidation
2. Trend-flip: close all opposite-side positions when EMA trend changes
"""
import logging
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df


def add_grid_indicators(df, ema_period=55, atr_period=14):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    df["fast_ema"] = df["close"].ewm(span=21, adjust=False).mean()
    df["uptrend"] = df["fast_ema"] > df["ema"]
    df["downtrend"] = df["fast_ema"] < df["ema"]
    return df


class GridPosition:
    def __init__(self, symbol, side, entry_price, quantity, take_profit,
                 leverage, entry_time, grid_idx, margin):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.take_profit = take_profit
        self.leverage = leverage
        self.entry_time = entry_time
        self.margin = margin  # Actual margin locked
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.exit_reason = ""
        self.grid_idx = grid_idx
        self.bars_held = 0

        # Calculate liquidation price
        if side == "LONG":
            # Liquidation when loss = margin (simplified, ignoring maintenance margin)
            self.liquidation_price = entry_price * (1 - 0.95 / leverage)
        else:
            self.liquidation_price = entry_price * (1 + 0.95 / leverage)

    def get_unrealized_pnl(self, current_price):
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def check_liquidation(self, row):
        """Check if position is liquidated."""
        self.bars_held += 1
        if self.side == "LONG":
            if row["low"] <= self.liquidation_price:
                self._close(self.liquidation_price, row.name, "LIQUIDATED")
                return True
        else:
            if row["high"] >= self.liquidation_price:
                self._close(self.liquidation_price, row.name, "LIQUIDATED")
                return True
        return False

    def check_tp(self, row):
        if self.side == "LONG" and row["high"] >= self.take_profit:
            self._close(self.take_profit, row.name, "TAKE_PROFIT")
            return True
        if self.side == "SHORT" and row["low"] <= self.take_profit:
            self._close(self.take_profit, row.name, "TAKE_PROFIT")
            return True
        return False

    def force_close(self, price, time, reason="FORCE"):
        self._close(price, time, reason)

    def _close(self, exit_price, exit_time, reason):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        if self.side == "LONG":
            raw_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            raw_pnl = (self.entry_price - self.exit_price) * self.quantity
        fee_rate = 0.0004 + 0.0002
        entry_notional = self.quantity * self.entry_price
        exit_notional = self.quantity * self.exit_price
        fees = (entry_notional + exit_notional) * fee_rate
        self.pnl = raw_pnl - fees
        # Cap loss at margin (liquidation)
        if self.pnl < -self.margin:
            self.pnl = -self.margin


class SmartGridV4:
    """Realistic grid with liquidation and trend-flip."""

    def __init__(self, initial_balance=100.0, params=None):
        self.initial_balance = initial_balance
        self.balance = initial_balance

        p = params or {}
        self.n_buy_grids = p.get("n_buy_grids", 4)
        self.n_sell_grids = p.get("n_sell_grids", 4)
        self.spacing_mult = p.get("spacing_mult", 1.0)
        self.leverage = p.get("leverage", 3)
        self.deploy_pct = p.get("deploy_pct", 0.15)
        self.max_pos_per_symbol = p.get("max_positions", 5)
        self.ema_period = p.get("ema_period", 55)
        self.atr_period = p.get("atr_period", 14)
        self.min_balance = p.get("min_balance", 3.0)
        self.max_margin_pct = p.get("max_margin", 0.50)
        self.tp_mult = p.get("tp_mult", 1.0)
        # Trend-aligned: only open positions in trend direction
        self.trend_aligned = p.get("trend_aligned", False)
        # Trend-flip: close all counter-trend positions when trend changes
        self.trend_flip = p.get("trend_flip", False)

        self.positions = defaultdict(list)
        self.closed_trades = []
        self.equity_curve = []
        self.total_fills = 0
        self.margin_locked = 0.0  # Total margin in open positions
        self.last_trends = {}  # symbol -> bool (was uptrend)

    def run(self, data_dict):
        prepared = {}
        for symbol, df in data_dict.items():
            prepared[symbol] = add_grid_indicators(
                df, self.ema_period, self.atr_period
            )

        all_ts = set()
        for df in prepared.values():
            all_ts.update(df.index)
        all_ts = sorted(all_ts)

        for ts in all_ts:
            if self.balance < self.min_balance:
                break

            for symbol in prepared:
                df = prepared[symbol]
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                if pd.isna(row["ema"]) or pd.isna(row["atr"]) or row["atr"] <= 0:
                    continue
                self._process_bar(symbol, row)

            # Record equity
            unrealized = 0
            for symbol, positions in self.positions.items():
                for pos in positions:
                    if ts in prepared[symbol].index:
                        price = prepared[symbol].loc[ts, "close"]
                        unrealized += pos.get_unrealized_pnl(price)

            equity = max(self.balance + unrealized, 0)
            self.equity_curve.append({
                "timestamp": ts,
                "balance": self.balance,
                "equity": equity,
                "open_positions": sum(len(p) for p in self.positions.values()),
            })

        # Close remaining at end
        for symbol, positions in list(self.positions.items()):
            if not positions:
                continue
            last_row = prepared[symbol].iloc[-1]
            for pos in positions:
                pos.force_close(last_row["close"], last_row.name, "END_OF_DATA")
                self.balance += pos.pnl
                self.margin_locked -= pos.margin
                self.closed_trades.append(pos)
            self.positions[symbol] = []

        return self._get_results()

    def _close_position(self, pos, price, time, reason):
        """Close a position and update accounting."""
        pos.force_close(price, time, reason)
        self.balance += pos.pnl
        self.margin_locked = max(0, self.margin_locked - pos.margin)
        self.closed_trades.append(pos)

    def _process_bar(self, symbol, row):
        center = row["ema"]
        spacing = row["atr"] * self.spacing_mult
        high = row["high"]
        low = row["low"]
        close = row["close"]

        if spacing <= 0 or center <= 0:
            return

        is_uptrend = bool(row.get("uptrend", True))

        # 0. Trend-flip: close counter-trend positions when trend changes
        if self.trend_flip and symbol in self.last_trends:
            was_up = self.last_trends[symbol]
            if was_up != is_uptrend:
                # Trend changed! Close all positions on the wrong side
                to_close = []
                for pos in self.positions[symbol]:
                    if is_uptrend and pos.side == "SHORT":
                        to_close.append(pos)
                    elif not is_uptrend and pos.side == "LONG":
                        to_close.append(pos)
                for pos in to_close:
                    self._close_position(pos, close, row.name, "TREND_FLIP")
                    self.positions[symbol].remove(pos)
        self.last_trends[symbol] = is_uptrend

        # 1. Check liquidation first (worst case)
        liquidated = []
        for pos in self.positions[symbol]:
            if pos.check_liquidation(row):
                self.balance += pos.pnl
                self.margin_locked = max(0, self.margin_locked - pos.margin)
                self.closed_trades.append(pos)
                liquidated.append(pos)
        for pos in liquidated:
            self.positions[symbol].remove(pos)

        # 2. Check TP
        closed = []
        for pos in self.positions[symbol]:
            if pos.check_tp(row):
                self.balance += pos.pnl
                self.margin_locked = max(0, self.margin_locked - pos.margin)
                self.closed_trades.append(pos)
                closed.append(pos)
        for pos in closed:
            self.positions[symbol].remove(pos)

        if self.balance < self.min_balance:
            return

        # 3. Available margin check
        available_margin = (self.balance * self.max_margin_pct) - self.margin_locked
        if available_margin <= 0:
            return

        # 4. Grid sides
        fill_buys = True
        fill_sells = True
        if self.trend_aligned or self.trend_flip:
            fill_buys = is_uptrend
            fill_sells = not is_uptrend

        # 5. Position size (COMPOUNDING)
        n_total = self.n_buy_grids + self.n_sell_grids
        if n_total <= 0:
            return
        total_margin_budget = self.balance * self.deploy_pct
        margin_per_grid = total_margin_budget / n_total
        if margin_per_grid < 0.3:
            return

        # 6. Grid levels
        buy_levels = [(center - i * spacing, -i) for i in range(1, self.n_buy_grids + 1)]
        sell_levels = [(center + i * spacing, i) for i in range(1, self.n_sell_grids + 1)]

        current_count = len(self.positions[symbol])

        # 7. Fill buy grids
        if fill_buys:
            for level, idx in buy_levels:
                if current_count >= self.max_pos_per_symbol:
                    break
                if level <= 0:
                    continue
                if available_margin < margin_per_grid:
                    break

                too_close = any(
                    pos.side == "LONG" and abs(pos.entry_price - level) < spacing * 0.4
                    for pos in self.positions[symbol]
                )
                if too_close:
                    continue

                if low <= level:
                    notional = margin_per_grid * self.leverage
                    quantity = notional / level
                    pair_info = config.PAIR_INFO.get(symbol, {})
                    min_qty = pair_info.get("min_qty", 0.1)
                    qty_step = pair_info.get("qty_step", 0.1)
                    min_notional = pair_info.get("min_notional", 5.0)

                    quantity = math.floor(quantity / qty_step) * qty_step
                    if quantity < min_qty or quantity * level < min_notional:
                        continue

                    tp = level + spacing * self.tp_mult
                    actual_margin = (quantity * level) / self.leverage

                    pos = GridPosition(
                        symbol=symbol, side="LONG",
                        entry_price=level, quantity=quantity,
                        take_profit=tp, leverage=self.leverage,
                        entry_time=row.name, grid_idx=idx,
                        margin=actual_margin,
                    )
                    self.positions[symbol].append(pos)
                    current_count += 1
                    self.total_fills += 1
                    self.margin_locked += actual_margin
                    available_margin -= actual_margin

        # 8. Fill sell grids
        if fill_sells:
            for level, idx in sell_levels:
                if current_count >= self.max_pos_per_symbol:
                    break
                if level <= 0:
                    continue
                if available_margin < margin_per_grid:
                    break

                too_close = any(
                    pos.side == "SHORT" and abs(pos.entry_price - level) < spacing * 0.4
                    for pos in self.positions[symbol]
                )
                if too_close:
                    continue

                if high >= level:
                    notional = margin_per_grid * self.leverage
                    quantity = notional / level
                    pair_info = config.PAIR_INFO.get(symbol, {})
                    min_qty = pair_info.get("min_qty", 0.1)
                    qty_step = pair_info.get("qty_step", 0.1)
                    min_notional = pair_info.get("min_notional", 5.0)

                    quantity = math.floor(quantity / qty_step) * qty_step
                    if quantity < min_qty or quantity * level < min_notional:
                        continue

                    tp = level - spacing * self.tp_mult
                    if tp <= 0:
                        continue
                    actual_margin = (quantity * level) / self.leverage

                    pos = GridPosition(
                        symbol=symbol, side="SHORT",
                        entry_price=level, quantity=quantity,
                        take_profit=tp, leverage=self.leverage,
                        entry_time=row.name, grid_idx=idx,
                        margin=actual_margin,
                    )
                    self.positions[symbol].append(pos)
                    current_count += 1
                    self.total_fills += 1
                    self.margin_locked += actual_margin
                    available_margin -= actual_margin

    def _get_results(self):
        if not self.closed_trades:
            return {"error": "No trades executed"}

        pnls = [t.pnl for t in self.closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total = len(self.closed_trades)
        win_rate = len(winners) / total if total > 0 else 0

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        eq = pd.DataFrame(self.equity_curve)
        max_dd = 0
        if len(eq) > 0:
            eq["peak"] = eq["equity"].cummax()
            eq["drawdown"] = (eq["peak"] - eq["equity"]) / eq["peak"]
            max_dd = eq["drawdown"].max()

        avg_monthly = 0
        monthly_rets = pd.Series(dtype=float)
        if len(eq) > 0:
            eq_m = eq.set_index("timestamp").resample("ME")["equity"].last().dropna()
            monthly_rets = eq_m.pct_change().dropna()
            avg_monthly = monthly_rets.mean() if len(monthly_rets) > 0 else 0

        years = 0
        ann_return = 0
        if len(eq) > 1:
            first_ts = eq["timestamp"].iloc[0]
            last_ts = eq["timestamp"].iloc[-1]
            years = (last_ts - first_ts).total_seconds() / (365.25 * 86400)
            if years > 0 and self.balance > 0 and self.initial_balance > 0:
                ann_return = (self.balance / self.initial_balance) ** (1 / years) - 1

        sharpe = 0
        if len(eq) > 1:
            eq["returns"] = eq["equity"].pct_change()
            mean_r = eq["returns"].mean()
            std_r = eq["returns"].std()
            sharpe = (mean_r / std_r) * np.sqrt(8760) if std_r > 0 else 0

        reasons = defaultdict(int)
        for t in self.closed_trades:
            reasons[t.exit_reason] += 1

        pos_months = 0
        neg_months = 0
        median_monthly = 0
        if len(monthly_rets) > 0:
            pos_months = int((monthly_rets > 0).sum())
            neg_months = int((monthly_rets <= 0).sum())
            median_monthly = float(monthly_rets.median())

        return {
            "initial_balance": self.initial_balance,
            "final_balance": round(self.balance, 2),
            "net_return": (self.balance - self.initial_balance) / self.initial_balance,
            "test_duration_years": round(years, 1),
            "annualized_return": ann_return,
            "avg_monthly_return": avg_monthly,
            "median_monthly": median_monthly,
            "monthly_summary": f"{pos_months}+/{neg_months}-",
            "total_trades": total,
            "total_fills": self.total_fills,
            "win_rate": win_rate,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "exit_reasons": dict(reasons),
        }


def run_sweep():
    data_dir = config.BACKTEST["data_dir"]
    data_dict = {}
    for pair in config.TRADING_PAIRS:
        fp = os.path.join(data_dir, f"{pair}_2022_2026.csv")
        if os.path.exists(fp):
            data_dict[pair] = load_data(fp)

    if not data_dict:
        print("ERROR: No data!")
        return

    print("SMART GRID V4 - REALISTIC (LIQUIDATION + TREND-FLIP)")
    print("=" * 155)
    print(f"Data: {', '.join(data_dict.keys())}, ~{len(next(iter(data_dict.values())))} bars\n")

    configs = [
        # --- Both-sided, hold until TP (classic grid) ---
        ("Both/Sp0.7/Lv3", {"spacing_mult": 0.7, "leverage": 3, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40}),
        ("Both/Sp1.0/Lv3", {"spacing_mult": 1.0, "leverage": 3, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40}),
        ("Both/Sp1.5/Lv3", {"spacing_mult": 1.5, "leverage": 3, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40}),
        ("Both/Sp1.0/Lv5", {"spacing_mult": 1.0, "leverage": 5, "deploy_pct": 0.12, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.35}),
        ("Both/Sp1.5/Lv5", {"spacing_mult": 1.5, "leverage": 5, "deploy_pct": 0.12, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.35}),
        # --- Trend-aligned (one-sided fills) ---
        ("TA/Sp0.5/Lv3", {"spacing_mult": 0.5, "leverage": 3, "deploy_pct": 0.20, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.45, "trend_aligned": True}),
        ("TA/Sp0.7/Lv3", {"spacing_mult": 0.7, "leverage": 3, "deploy_pct": 0.20, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.45, "trend_aligned": True}),
        ("TA/Sp1.0/Lv3", {"spacing_mult": 1.0, "leverage": 3, "deploy_pct": 0.20, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.45, "trend_aligned": True}),
        ("TA/Sp0.7/Lv5", {"spacing_mult": 0.7, "leverage": 5, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40, "trend_aligned": True}),
        ("TA/Sp1.0/Lv5", {"spacing_mult": 1.0, "leverage": 5, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40, "trend_aligned": True}),
        # --- Trend-flip (close counter-trend positions on trend change) ---
        ("TF/Sp0.5/Lv3", {"spacing_mult": 0.5, "leverage": 3, "deploy_pct": 0.20, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.45, "trend_flip": True}),
        ("TF/Sp0.7/Lv3", {"spacing_mult": 0.7, "leverage": 3, "deploy_pct": 0.20, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.45, "trend_flip": True}),
        ("TF/Sp1.0/Lv3", {"spacing_mult": 1.0, "leverage": 3, "deploy_pct": 0.20, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.45, "trend_flip": True}),
        ("TF/Sp0.7/Lv5", {"spacing_mult": 0.7, "leverage": 5, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40, "trend_flip": True}),
        ("TF/Sp1.0/Lv5", {"spacing_mult": 1.0, "leverage": 5, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40, "trend_flip": True}),
        ("TF/Sp1.5/Lv5", {"spacing_mult": 1.5, "leverage": 5, "deploy_pct": 0.15, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.40, "trend_flip": True}),
        # --- Trend-flip with more deployment ---
        ("TF/Sp0.7/Lv3/D30", {"spacing_mult": 0.7, "leverage": 3, "deploy_pct": 0.30, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 6, "max_margin": 0.60, "trend_flip": True}),
        ("TF/Sp1.0/Lv3/D30", {"spacing_mult": 1.0, "leverage": 3, "deploy_pct": 0.30, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 6, "max_margin": 0.60, "trend_flip": True}),
        ("TF/Sp0.7/Lv5/D25", {"spacing_mult": 0.7, "leverage": 5, "deploy_pct": 0.25, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 6, "max_margin": 0.50, "trend_flip": True}),
        ("TF/Sp1.0/Lv5/D25", {"spacing_mult": 1.0, "leverage": 5, "deploy_pct": 0.25, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 6, "max_margin": 0.50, "trend_flip": True}),
        # --- Trend-flip with more grid levels ---
        ("TF/Sp0.5/Lv3/6g", {"spacing_mult": 0.5, "leverage": 3, "deploy_pct": 0.25, "n_buy_grids": 6, "n_sell_grids": 6, "max_positions": 8, "max_margin": 0.50, "trend_flip": True}),
        ("TF/Sp0.7/Lv3/6g", {"spacing_mult": 0.7, "leverage": 3, "deploy_pct": 0.25, "n_buy_grids": 6, "n_sell_grids": 6, "max_positions": 8, "max_margin": 0.50, "trend_flip": True}),
        ("TF/Sp0.7/Lv5/6g", {"spacing_mult": 0.7, "leverage": 5, "deploy_pct": 0.20, "n_buy_grids": 6, "n_sell_grids": 6, "max_positions": 8, "max_margin": 0.50, "trend_flip": True}),
        # --- Higher leverage with low deploy ---
        ("TF/Sp1.0/Lv10/D8", {"spacing_mult": 1.0, "leverage": 10, "deploy_pct": 0.08, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.25, "trend_flip": True}),
        ("TF/Sp1.0/Lv7/D12", {"spacing_mult": 1.0, "leverage": 7, "deploy_pct": 0.12, "n_buy_grids": 4, "n_sell_grids": 4, "max_positions": 5, "max_margin": 0.35, "trend_flip": True}),
    ]

    header = (f"{'Config':<20} {'$100→':<12} {'Monthly':<10} {'Median':<10} {'Annual':<10} "
              f"{'Trades':<8} {'WR':<8} {'PF':<8} {'MaxDD':<8} {'Sharpe':<8} "
              f"{'Months':<10} {'Exits'}")
    print(header)
    print("-" * 155)

    best = None
    best_score = -999

    for name, params in configs:
        bt = SmartGridV4(initial_balance=100.0, params=params)
        r = bt.run(data_dict)

        if "error" in r:
            print(f"{name:<20} ERROR: {r['error']}")
            continue

        monthly = r["avg_monthly_return"]
        dd = max(r["max_drawdown"], 0.01)
        score = monthly / dd if monthly > 0 else monthly * dd

        flag = ""
        if score > best_score and monthly > 0:
            best_score = score
            best = (name, params, r)
            flag = " <-- BEST"

        exits = ", ".join(f"{k[:3]}:{v}" for k, v in sorted(r["exit_reasons"].items()))
        print(f"{name:<20} ${r['final_balance']:<11.2f} {monthly:<9.2%} "
              f"{r['median_monthly']:<9.2%} "
              f"{r['annualized_return']:<9.1%} {r['total_trades']:<8} "
              f"{r['win_rate']:<7.1%} {r['profit_factor']:<7.2f} "
              f"{r['max_drawdown']:<7.1%} {r['sharpe_ratio']:<7.2f} "
              f"{r['monthly_summary']:<10} {exits}{flag}")

    print("\n" + "=" * 155)
    if best:
        name, params, r = best
        print(f"\nBEST: {name}")
        print(f"  Params: {params}")
        print(f"  ${r['initial_balance']:.0f} → ${r['final_balance']:.2f}")
        print(f"  Monthly: {r['avg_monthly_return']:.2%} (median: {r['median_monthly']:.2%})")
        print(f"  Annual: {r['annualized_return']:.1%}, WR: {r['win_rate']:.1%}, PF: {r['profit_factor']:.2f}")
        print(f"  Max DD: {r['max_drawdown']:.1%}, Sharpe: {r['sharpe_ratio']:.2f}")
        print(f"  Trades: {r['total_trades']}, Exits: {r['exit_reasons']}")

        # Multi-balance
        print(f"\n{'='*90}")
        print("MULTI-BALANCE TEST")
        print(f"{'='*90}")
        for bal in [10, 100, 1000]:
            bt2 = SmartGridV4(initial_balance=bal, params=params)
            r2 = bt2.run(data_dict)
            if "error" not in r2:
                print(f"  ${bal:>5} → ${r2['final_balance']:.2f} "
                      f"(Mo: {r2['avg_monthly_return']:.2%}, "
                      f"DD: {r2['max_drawdown']:.1%}, "
                      f"WR: {r2['win_rate']:.1%}, "
                      f"PF: {r2['profit_factor']:.2f}, "
                      f"Trades: {r2['total_trades']})")

        # Per-symbol
        print(f"\n{'='*90}")
        print("PER-SYMBOL ($100)")
        print(f"{'='*90}")
        for sym in data_dict:
            bt3 = SmartGridV4(initial_balance=100.0, params=params)
            r3 = bt3.run({sym: data_dict[sym]})
            if "error" not in r3:
                print(f"  {sym}: ${r3['final_balance']:.2f} "
                      f"(Mo: {r3['avg_monthly_return']:.2%}, "
                      f"WR: {r3['win_rate']:.1%}, PF: {r3['profit_factor']:.2f}, "
                      f"DD: {r3['max_drawdown']:.1%}, "
                      f"Exits: {r3['exit_reasons']})")

    return best


if __name__ == "__main__":
    run_sweep()
