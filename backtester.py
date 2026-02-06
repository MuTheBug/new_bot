"""
Backtesting Engine.

Simulates strategy execution on historical data with realistic assumptions:
- Transaction fees (maker/taker)
- Slippage
- Minimum order sizes
- Dynamic leverage
- Walk-forward validation
- Monte Carlo simulation
- Multi-balance testing ($10, $100, $1000)
"""
import logging
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from risk_manager import RiskManager
from strategy import add_indicators, generate_signals

logger = logging.getLogger(__name__)


def load_data(filepath):
    """Load OHLCV CSV data into a DataFrame."""
    df = pd.read_csv(filepath)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }, inplace=True)

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df


class BacktestTrade:
    """Represents a single backtest trade."""

    def __init__(self, symbol, side, entry_price, quantity, stop_loss,
                 take_profit, leverage, entry_time, atr):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.entry_time = entry_time
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.exit_reason = ""
        self.atr = atr
        self.trailing_stop = None
        self.max_favorable = 0.0

    def update(self, row):
        """
        Check if trade should be closed on this bar.
        Returns True if trade was closed.
        Uses high/low to check stop/TP hits within the bar.
        """
        high = row["high"]
        low = row["low"]
        close = row["close"]

        if self.side == "LONG":
            # Check stop-loss (hit if low <= stop)
            if low <= self.stop_loss:
                self._close(self.stop_loss, row.name, "STOP_LOSS")
                return True
            # Check take-profit (hit if high >= take_profit)
            if high >= self.take_profit:
                self._close(self.take_profit, row.name, "TAKE_PROFIT")
                return True
            # Update trailing stop
            unrealized = close - self.entry_price
            self.max_favorable = max(self.max_favorable, unrealized)
            trailing = self._calc_trailing(close)
            if trailing and low <= trailing:
                self._close(trailing, row.name, "TRAILING_STOP")
                return True
            if trailing:
                self.trailing_stop = trailing

        elif self.side == "SHORT":
            if high >= self.stop_loss:
                self._close(self.stop_loss, row.name, "STOP_LOSS")
                return True
            if low <= self.take_profit:
                self._close(self.take_profit, row.name, "TAKE_PROFIT")
                return True
            unrealized = self.entry_price - close
            self.max_favorable = max(self.max_favorable, unrealized)
            trailing = self._calc_trailing(close)
            if trailing and high >= trailing:
                self._close(trailing, row.name, "TRAILING_STOP")
                return True
            if trailing:
                self.trailing_stop = trailing

        return False

    def _calc_trailing(self, current_price):
        """Calculate trailing stop if activation threshold met."""
        activation = config.STRATEGY["trailing_activation"] * self.atr
        trail = config.STRATEGY["trailing_step"] * self.atr

        if self.side == "LONG":
            if (current_price - self.entry_price) >= activation:
                return current_price - trail
        else:
            if (self.entry_price - current_price) >= activation:
                return current_price + trail
        return None

    def _close(self, exit_price, exit_time, reason):
        """Close the trade and calculate PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        if self.side == "LONG":
            raw_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            raw_pnl = (self.entry_price - self.exit_price) * self.quantity

        # Subtract fees (entry + exit)
        fee_rate = config.BACKTEST["commission_rate"]
        slippage_rate = config.BACKTEST["slippage_rate"]
        notional = self.quantity * self.entry_price
        exit_notional = self.quantity * self.exit_price
        total_fees = (notional + exit_notional) * (fee_rate + slippage_rate)

        self.pnl = raw_pnl - total_fees
        self.pnl_pct = self.pnl / (notional / self.leverage) if notional > 0 else 0

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
        }


class Backtester:
    """Main backtesting engine."""

    def __init__(self, initial_balance=100.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_manager = RiskManager()
        self.risk_manager.peak_balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}  # symbol -> BacktestTrade

    def run(self, data_dict, symbol_names=None):
        """
        Run backtest on provided data.
        data_dict: {symbol: DataFrame} with OHLCV data
        """
        if symbol_names is None:
            symbol_names = list(data_dict.keys())

        # Generate signals for each symbol
        signal_data = {}
        for symbol, df in data_dict.items():
            signal_data[symbol] = generate_signals(df)

        # Align all DataFrames by timestamp
        all_timestamps = set()
        for df in signal_data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)

        # Walk through each timestamp
        last_day = None
        for ts in all_timestamps:
            # Reset daily circuit breaker
            current_day = ts.date() if hasattr(ts, 'date') else None
            if current_day and current_day != last_day:
                self.risk_manager.reset_daily()
                if self.risk_manager.is_halted and "Daily" in self.risk_manager.halt_reason:
                    self.risk_manager.is_halted = False
                    self.risk_manager.halt_reason = ""
                last_day = current_day

            for symbol in symbol_names:
                df = signal_data[symbol]
                if ts not in df.index:
                    continue

                row = df.loc[ts]

                # Update existing position
                if symbol in self.open_positions:
                    trade = self.open_positions[symbol]
                    if trade.update(row):
                        self.balance += trade.pnl
                        self.risk_manager.record_trade(trade.pnl, self.balance)
                        self.trades.append(trade)
                        del self.open_positions[symbol]

                # Check for new signals
                if (
                    symbol not in self.open_positions
                    and row.get("signal", 0) != 0
                    and self.risk_manager.can_trade()
                    and len(self.open_positions) < config.RISK["max_open_positions"]
                ):
                    signal = row["signal"]
                    side = "LONG" if signal == 1 else "SHORT"
                    atr = row["atr"]
                    price = row["close"]

                    if pd.isna(atr) or atr <= 0:
                        continue

                    leverage = self.risk_manager.calculate_optimal_leverage(
                        self.balance, atr, price, symbol
                    )
                    quantity = self.risk_manager.calculate_position_size(
                        self.balance, leverage, price, atr, symbol
                    )

                    if quantity <= 0:
                        continue

                    # Calculate stop/TP
                    stop_mult = config.STRATEGY["atr_stop_multiplier"]
                    tp_mult = config.STRATEGY["atr_tp_multiplier"]

                    if side == "LONG":
                        stop_loss = price - (atr * stop_mult)
                        take_profit = price + (atr * tp_mult)
                    else:
                        stop_loss = price + (atr * stop_mult)
                        take_profit = price - (atr * tp_mult)

                    trade = BacktestTrade(
                        symbol=symbol,
                        side=side,
                        entry_price=price,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        leverage=leverage,
                        entry_time=ts,
                        atr=atr,
                    )
                    self.open_positions[symbol] = trade

            # Record equity
            unrealized = sum(
                (
                    (row["close"] - t.entry_price) * t.quantity
                    if t.side == "LONG"
                    else (t.entry_price - row["close"]) * t.quantity
                )
                for sym, t in self.open_positions.items()
                if ts in signal_data.get(sym, pd.DataFrame()).index
                for row in [signal_data[sym].loc[ts]]
            )
            self.equity_curve.append({
                "timestamp": ts,
                "balance": self.balance,
                "equity": self.balance + unrealized,
                "open_positions": len(self.open_positions),
            })

        # Close any remaining open positions at last price
        for symbol, trade in list(self.open_positions.items()):
            last_row = signal_data[symbol].iloc[-1]
            trade._close(last_row["close"], last_row.name, "END_OF_DATA")
            self.balance += trade.pnl
            self.risk_manager.record_trade(trade.pnl, self.balance)
            self.trades.append(trade)
        self.open_positions.clear()

        return self.get_results()

    def get_results(self):
        """Calculate comprehensive backtest statistics."""
        if not self.trades:
            return {"error": "No trades executed"}

        trade_data = [t.to_dict() for t in self.trades]
        df_trades = pd.DataFrame(trade_data)

        # Basic stats
        total_trades = len(df_trades)
        winners = df_trades[df_trades["pnl"] > 0]
        losers = df_trades[df_trades["pnl"] <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers["pnl"].mean()) if len(losers) > 0 else 0

        gross_profit = winners["pnl"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        net_pnl = df_trades["pnl"].sum()
        net_return = net_pnl / self.initial_balance

        # Drawdown analysis
        eq = pd.DataFrame(self.equity_curve)
        if len(eq) > 0:
            eq["peak"] = eq["equity"].cummax()
            eq["drawdown"] = (eq["peak"] - eq["equity"]) / eq["peak"]
            max_drawdown = eq["drawdown"].max()
        else:
            max_drawdown = 0

        # Sharpe ratio (annualized, assuming 1hr bars)
        if len(eq) > 1:
            eq["returns"] = eq["equity"].pct_change()
            avg_return = eq["returns"].mean()
            std_return = eq["returns"].std()
            # Annualize: 8760 hours in a year
            sharpe = (avg_return / std_return) * np.sqrt(8760) if std_return > 0 else 0
        else:
            sharpe = 0

        # Monthly returns
        if len(eq) > 0:
            eq_monthly = eq.set_index("timestamp").resample("ME")["equity"].last()
            monthly_returns = eq_monthly.pct_change().dropna()
            avg_monthly_return = monthly_returns.mean() if len(monthly_returns) > 0 else 0
        else:
            avg_monthly_return = 0

        # Consecutive losses
        max_consec_losses = 0
        current_consec = 0
        for pnl in df_trades["pnl"]:
            if pnl <= 0:
                current_consec += 1
                max_consec_losses = max(max_consec_losses, current_consec)
            else:
                current_consec = 0

        # Risk-reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # Duration stats
        if "entry_time" in df_trades.columns and "exit_time" in df_trades.columns:
            df_trades["duration"] = pd.to_datetime(df_trades["exit_time"]) - pd.to_datetime(df_trades["entry_time"])
            avg_duration = df_trades["duration"].mean()
        else:
            avg_duration = None

        # Test duration and annualized return
        if len(eq) > 1:
            first_ts = eq["timestamp"].iloc[0]
            last_ts = eq["timestamp"].iloc[-1]
            duration = last_ts - first_ts
            years = duration.total_seconds() / (365.25 * 86400)
            if years > 0 and self.balance > 0 and self.initial_balance > 0:
                annualized_return = (self.balance / self.initial_balance) ** (1 / years) - 1
            else:
                annualized_return = 0
        else:
            years = 0
            annualized_return = 0

        # Per-side analysis
        longs = df_trades[df_trades["side"] == "LONG"]
        shorts = df_trades[df_trades["side"] == "SHORT"]

        results = {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "net_pnl": net_pnl,
            "net_return": net_return,
            "test_duration_years": round(years, 1),
            "annualized_return": annualized_return,
            "total_trades": total_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "risk_reward_ratio": risk_reward,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "avg_monthly_return": avg_monthly_return,
            "max_consecutive_losses": max_consec_losses,
            "long_trades": len(longs),
            "short_trades": len(shorts),
            "long_win_rate": (
                len(longs[longs["pnl"] > 0]) / len(longs) if len(longs) > 0 else 0
            ),
            "short_win_rate": (
                len(shorts[shorts["pnl"] > 0]) / len(shorts)
                if len(shorts) > 0
                else 0
            ),
            "avg_trade_duration": str(avg_duration) if avg_duration else "N/A",
            "exit_reasons": df_trades["exit_reason"].value_counts().to_dict(),
        }

        return results


def run_walk_forward(data_dict, n_windows=5, initial_balance=100.0):
    """
    Walk-forward analysis.
    Splits data into n_windows, trains on each window, tests on next.
    Returns results for each out-of-sample window.
    """
    results = []

    # Get total length from first symbol
    first_symbol = list(data_dict.keys())[0]
    total_len = len(data_dict[first_symbol])
    window_size = total_len // (n_windows + 1)

    for i in range(n_windows):
        train_start = 0
        train_end = window_size * (i + 1)
        test_start = train_end
        test_end = min(train_end + window_size, total_len)

        if test_end <= test_start:
            break

        # Use test data for out-of-sample validation
        test_data = {}
        for symbol, df in data_dict.items():
            test_data[symbol] = df.iloc[test_start:test_end].copy()

        bt = Backtester(initial_balance=initial_balance)
        window_results = bt.run(test_data)
        window_results["window"] = i + 1
        window_results["test_start"] = str(test_data[first_symbol].index[0])
        window_results["test_end"] = str(test_data[first_symbol].index[-1])
        results.append(window_results)

    return results


def run_monte_carlo(trade_pnl_pcts, initial_balance=100.0, n_simulations=1000):
    """
    Monte Carlo simulation using bootstrap sampling (with replacement).
    Samples N trades from the pool to create each simulation path,
    producing genuinely different outcomes per run.
    trade_pnl_pcts: list of PnL as fraction of balance at time of trade.
    """
    if len(trade_pnl_pcts) == 0:
        return {}

    n_trades = len(trade_pnl_pcts)
    pnl_arr = np.array(trade_pnl_pcts)
    final_balances = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Bootstrap: sample with replacement
        sampled = pnl_arr[np.random.randint(0, n_trades, size=n_trades)]
        balance = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for pnl_pct in sampled:
            balance *= (1 + pnl_pct)
            if balance <= 0:
                balance = 0
                break
            if balance > peak:
                peak = balance
            if peak > 0:
                dd = (peak - balance) / peak
                max_dd = max(max_dd, dd)

        final_balances.append(balance)
        max_drawdowns.append(max_dd)

    final_balances = np.array(final_balances)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "median_final_balance": float(np.median(final_balances)),
        "mean_final_balance": float(np.mean(final_balances)),
        "p5_final_balance": float(np.percentile(final_balances, 5)),
        "p95_final_balance": float(np.percentile(final_balances, 95)),
        "prob_profit": float(np.mean(final_balances > initial_balance)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "p95_max_drawdown": float(np.percentile(max_drawdowns, 95)),
    }


def run_full_backtest():
    """
    Run comprehensive backtesting suite:
    1. Full dataset backtest at multiple balance levels
    2. In-sample vs out-of-sample comparison
    3. Walk-forward analysis
    4. Monte Carlo simulation
    """
    print("=" * 70)
    print("ADAPTIVE TREND-MOMENTUM (ATM) STRATEGY - BACKTEST SUITE")
    print("=" * 70)

    # Load data
    data_dir = config.BACKTEST["data_dir"]
    data_dict = {}
    for pair in config.TRADING_PAIRS:
        filepath = os.path.join(data_dir, f"{pair}_2022_2026.csv")
        if os.path.exists(filepath):
            data_dict[pair] = load_data(filepath)
            print(f"Loaded {pair}: {len(data_dict[pair])} bars "
                  f"({data_dict[pair].index[0]} to {data_dict[pair].index[-1]})")

    if not data_dict:
        print("ERROR: No data files found!")
        return

    # ── 1. Full Dataset Backtest at Multiple Balance Levels ──────────
    print("\n" + "=" * 70)
    print("1. FULL DATASET BACKTEST")
    print("=" * 70)

    all_results = {}
    for balance in config.BACKTEST["initial_balances"]:
        print(f"\n--- Starting Balance: ${balance} ---")
        bt = Backtester(initial_balance=balance)
        results = bt.run(data_dict)
        all_results[balance] = {"results": results, "trades": bt.trades}
        _print_results(results)

    # ── 2. In-Sample vs Out-of-Sample ───────────────────────────────
    print("\n" + "=" * 70)
    print("2. IN-SAMPLE vs OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    oos_split = config.BACKTEST["oos_split"]
    first_sym = list(data_dict.keys())[0]
    split_idx = int(len(data_dict[first_sym]) * (1 - oos_split))

    is_data = {s: df.iloc[:split_idx].copy() for s, df in data_dict.items()}
    oos_data = {s: df.iloc[split_idx:].copy() for s, df in data_dict.items()}

    print(f"\nIn-Sample: {is_data[first_sym].index[0]} to {is_data[first_sym].index[-1]}")
    print(f"Out-of-Sample: {oos_data[first_sym].index[0]} to {oos_data[first_sym].index[-1]}")

    bt_is = Backtester(initial_balance=100.0)
    is_results = bt_is.run(is_data)
    print("\n  IN-SAMPLE RESULTS:")
    _print_results(is_results, indent=4)

    bt_oos = Backtester(initial_balance=100.0)
    oos_results = bt_oos.run(oos_data)
    print("\n  OUT-OF-SAMPLE RESULTS:")
    _print_results(oos_results, indent=4)

    # Check divergence
    if is_results.get("win_rate", 0) > 0 and oos_results.get("win_rate", 0) > 0:
        wr_divergence = abs(is_results["win_rate"] - oos_results["win_rate"]) / is_results["win_rate"]
        pf_is = is_results.get("profit_factor", 0)
        pf_oos = oos_results.get("profit_factor", 0)
        if pf_is > 0 and pf_is != float("inf"):
            pf_divergence = abs(pf_is - pf_oos) / pf_is
        else:
            pf_divergence = 0

        print(f"\n  Win Rate Divergence: {wr_divergence:.1%} {'PASS' if wr_divergence < 0.25 else 'FAIL'} (<25%)")
        print(f"  Profit Factor Divergence: {pf_divergence:.1%} {'PASS' if pf_divergence < 0.25 else 'FAIL'} (<25%)")

    # ── 3. Walk-Forward Analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. WALK-FORWARD ANALYSIS")
    print("=" * 70)

    wf_results = run_walk_forward(
        data_dict,
        n_windows=config.BACKTEST["walk_forward_windows"],
        initial_balance=100.0,
    )
    for wf in wf_results:
        print(f"\n  Window {wf['window']}: {wf['test_start']} to {wf['test_end']}")
        print(f"    Trades: {wf['total_trades']}, Win Rate: {wf['win_rate']:.1%}, "
              f"PF: {wf.get('profit_factor', 0):.2f}, "
              f"Return: {wf.get('net_return', 0):.1%}")

    # ── 4. Monte Carlo Simulation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("4. MONTE CARLO SIMULATION")
    print("=" * 70)

    # Use trades from $100 backtest - convert to PnL percentages
    if 100 in all_results and all_results[100]["trades"]:
        trades = all_results[100]["trades"]
        trades_pnl_pcts = []
        for t in trades:
            # PnL as fraction of margin used
            notional = t.quantity * t.entry_price
            margin = notional / t.leverage if t.leverage > 0 else notional
            if margin > 0:
                trades_pnl_pcts.append(t.pnl / margin)
            else:
                trades_pnl_pcts.append(0)

        for balance in [10, 100, 1000]:
            mc = run_monte_carlo(
                trades_pnl_pcts, balance,
                config.BACKTEST["monte_carlo_iterations"],
            )
            print(f"\n  Monte Carlo (${balance} start, {config.BACKTEST['monte_carlo_iterations']} sims):")
            print(f"    Median Final: ${mc['median_final_balance']:.2f}")
            print(f"    5th-95th Pctile: ${mc['p5_final_balance']:.2f} - ${mc['p95_final_balance']:.2f}")
            print(f"    Probability of Profit: {mc['prob_profit']:.1%}")
            print(f"    Median Max DD: {mc['median_max_drawdown']:.1%}")
            print(f"    95th Pctile Max DD: {mc['p95_max_drawdown']:.1%}")

    # ── 5. Per-Symbol Analysis ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. PER-SYMBOL ANALYSIS ($100 balance)")
    print("=" * 70)

    for symbol in data_dict:
        single = {symbol: data_dict[symbol]}
        bt = Backtester(initial_balance=100.0)
        res = bt.run(single)
        print(f"\n  {symbol}:")
        _print_results(res, indent=4)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return all_results


def _print_results(results, indent=2):
    """Pretty-print backtest results."""
    pad = " " * indent
    if "error" in results:
        print(f"{pad}ERROR: {results['error']}")
        return

    print(f"{pad}Final Balance: ${results['final_balance']:.2f} "
          f"(from ${results['initial_balance']:.2f})")
    years = results.get('test_duration_years', 0)
    ann = results.get('annualized_return', 0)
    print(f"{pad}Net Return: {results['net_return']:.1%} over {years} years "
          f"({ann:.1%} annualized)")
    print(f"{pad}Total Trades: {results['total_trades']} "
          f"(L:{results['long_trades']}, S:{results['short_trades']})")
    print(f"{pad}Win Rate: {results['win_rate']:.1%} "
          f"(L:{results['long_win_rate']:.1%}, S:{results['short_win_rate']:.1%})")
    print(f"{pad}Avg Win: ${results['avg_win']:.4f}, "
          f"Avg Loss: ${results['avg_loss']:.4f}")
    print(f"{pad}Risk/Reward: {results['risk_reward_ratio']:.2f}")
    print(f"{pad}Profit Factor: {results['profit_factor']:.2f}")
    print(f"{pad}Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"{pad}Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"{pad}Avg Monthly Return: {results['avg_monthly_return']:.1%}")
    print(f"{pad}Max Consecutive Losses: {results['max_consecutive_losses']}")
    print(f"{pad}Avg Trade Duration: {results['avg_trade_duration']}")
    print(f"{pad}Exit Reasons: {results['exit_reasons']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_full_backtest()
