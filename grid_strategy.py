"""
Smart Grid Trading Strategy for Live Bot.

Manages grid levels, position tracking, and signal generation for
the grid trading mode. Used by bot.py when STRATEGY_MODE="grid".

The grid places buy orders below EMA and sell orders above EMA,
capturing profit from natural price oscillation. Trend-aligned mode
only fills grids in the trend direction.
"""
import logging
import math

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def add_grid_indicators(df):
    """Add indicators needed for grid strategy."""
    g = config.GRID
    df = df.copy()
    df["ema"] = compute_ema(df["close"], g["ema_period"])
    df["fast_ema"] = compute_ema(df["close"], 21)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], g["atr_period"])
    df["uptrend"] = (df["fast_ema"] > df["ema"]).fillna(False).astype(bool)
    df["downtrend"] = (df["fast_ema"] < df["ema"]).fillna(False).astype(bool)
    return df


def get_grid_levels(center, atr, spacing_mult, n_buy, n_sell):
    """Calculate grid buy/sell levels around center."""
    spacing = atr * spacing_mult
    if spacing <= 0 or center <= 0:
        return [], []

    buy_levels = [center - i * spacing for i in range(1, n_buy + 1)]
    sell_levels = [center + i * spacing for i in range(1, n_sell + 1)]
    return buy_levels, sell_levels


def get_grid_signals(df, current_positions=None):
    """
    Generate grid trading signals from latest data.

    Returns a list of dicts with grid orders to place:
    [{"side": "LONG"/"SHORT", "price": float, "tp": float, "grid_idx": int}, ...]

    current_positions: list of dicts with {"side", "entry_price"} of open positions
    """
    g = config.GRID
    df = add_grid_indicators(df)

    if len(df) < g["ema_period"] + 5:
        return []

    latest = df.iloc[-1]
    if pd.isna(latest["ema"]) or pd.isna(latest["atr"]) or latest["atr"] <= 0:
        return []

    center = latest["ema"]
    atr = latest["atr"]
    spacing = atr * g["spacing_mult"]
    close = latest["close"]

    if spacing <= 0:
        return []

    # Determine which sides to fill
    fill_buys = True
    fill_sells = True
    if g["trend_aligned"]:
        fill_buys = bool(latest["uptrend"])
        fill_sells = bool(latest["downtrend"])
    if g.get("trend_flip", False):
        fill_buys = bool(latest["uptrend"])
        fill_sells = bool(latest["downtrend"])

    buy_levels, sell_levels = get_grid_levels(
        center, atr, g["spacing_mult"], g["n_buy_grids"], g["n_sell_grids"]
    )

    current_positions = current_positions or []
    signals = []

    # Check buy levels
    if fill_buys:
        for i, level in enumerate(buy_levels):
            if level <= 0:
                continue
            # Skip if we already have a position near this level
            too_close = any(
                p["side"] == "LONG" and abs(p["entry_price"] - level) < spacing * 0.4
                for p in current_positions
            )
            if too_close:
                continue

            # Only signal if current price is near or below the grid level
            if close <= level * 1.005:  # Within 0.5% of the level
                tp = level + spacing * g["tp_mult"]
                signals.append({
                    "side": "LONG",
                    "price": level,
                    "tp": tp,
                    "grid_idx": -(i + 1),
                    "spacing": spacing,
                })

    # Check sell levels
    if fill_sells:
        for i, level in enumerate(sell_levels):
            if level <= 0:
                continue
            too_close = any(
                p["side"] == "SHORT" and abs(p["entry_price"] - level) < spacing * 0.4
                for p in current_positions
            )
            if too_close:
                continue

            if close >= level * 0.995:
                tp = level - spacing * g["tp_mult"]
                if tp <= 0:
                    continue
                signals.append({
                    "side": "SHORT",
                    "price": level,
                    "tp": tp,
                    "grid_idx": i + 1,
                    "spacing": spacing,
                })

    return signals


def calculate_grid_position_size(balance, price, symbol):
    """Calculate position size for a single grid order."""
    g = config.GRID
    n_total = g["n_buy_grids"] + g["n_sell_grids"]
    if n_total <= 0:
        return 0

    total_margin = balance * g["deploy_pct"]
    margin_per_grid = total_margin / n_total
    if margin_per_grid < 0.5:
        return 0

    notional = margin_per_grid * g["leverage"]
    quantity = notional / price

    # Apply Binance minimums
    pair_info = config.PAIR_INFO.get(symbol, {})
    min_qty = pair_info.get("min_qty", 0.1)
    qty_step = pair_info.get("qty_step", 0.1)
    min_notional = pair_info.get("min_notional", 5.0)

    quantity = math.floor(quantity / qty_step) * qty_step
    if quantity < min_qty or quantity * price < min_notional:
        return 0

    return quantity
