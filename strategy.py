"""
Adaptive Trend-Momentum (ATM) Trading Strategy.

Core Edge: Mean-reversion entries within a trend.
- In uptrends, buy oversold dips (RSI drops then recovers)
- In downtrends, sell overbought rallies (RSI rises then fades)
- Trend defined by dual EMA alignment

This exploits the empirical tendency of crypto to:
1. Trend strongly (EMA alignment captures this)
2. Mean-revert within trends (oversold in uptrend = buying opportunity)
"""
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_ema(series, period):
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series, period):
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_adx(high, low, close, period=14):
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = compute_atr(high, low, close, period)
    plus_di = 100 * compute_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * compute_ema(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = compute_ema(dx, period)
    return adx, plus_di, minus_di


def add_indicators(df):
    """Compute all technical indicators."""
    params = config.STRATEGY
    df = df.copy()

    # Trend EMAs
    df["fast_ema"] = compute_ema(df["close"], params["fast_ema_period"])
    df["trend_ema"] = compute_ema(df["close"], params["trend_ema_period"])

    # RSI
    df["rsi"] = compute_rsi(df["close"], params["rsi_period"])
    df["prev_rsi"] = df["rsi"].shift(1)
    df["prev2_rsi"] = df["rsi"].shift(2)

    # ATR
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], params["atr_period"])

    # Volume
    df["volume_ma"] = compute_sma(df["volume"], params["volume_ma_period"])
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

    # ADX
    adx, plus_di, minus_di = compute_adx(
        df["high"], df["low"], df["close"], params["adx_period"]
    )
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Trend direction
    df["uptrend"] = (df["fast_ema"] > df["trend_ema"]).fillna(False).astype(bool)
    df["downtrend"] = (df["fast_ema"] < df["trend_ema"]).fillna(False).astype(bool)

    # Candle
    df["bullish"] = df["close"] > df["open"]
    df["bearish"] = df["close"] < df["open"]

    return df


def detect_regime(df):
    """Detect market regime."""
    if len(df) < 2:
        return "RANGING"
    latest = df.iloc[-1]
    if latest["adx"] > config.STRATEGY["adx_trend_threshold"]:
        if latest["uptrend"]:
            return "TRENDING_UP"
        elif latest["downtrend"]:
            return "TRENDING_DOWN"
    return "RANGING"


def generate_signals(df):
    """
    Generate trading signals using mean-reversion within trend.

    LONG entry conditions (all must be true):
    1. Uptrend: fast EMA > trend EMA
    2. RSI was recently oversold (dropped below threshold in last 3 bars)
    3. RSI is now recovering (current > previous)
    4. Current candle is bullish (confirming the bounce)
    5. Volume at least 0.7x average

    SHORT entry conditions (mirror):
    1. Downtrend: fast EMA < trend EMA
    2. RSI was recently overbought
    3. RSI is now declining
    4. Current candle is bearish
    5. Volume at least 0.7x average
    """
    params = config.STRATEGY
    df = add_indicators(df)
    df["signal"] = 0

    rsi_os = params["rsi_oversold"]
    rsi_ob = params["rsi_overbought"]
    vol_thresh = params["volume_threshold"]
    min_bars = params.get("min_bars_between_trades", 2)
    last_signal_bar = -min_bars - 1

    for i in range(3, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        if pd.isna(row["trend_ema"]) or pd.isna(row["atr"]) or pd.isna(row["rsi"]):
            continue

        if (i - last_signal_bar) < min_bars:
            continue

        vol_ok = row["volume_ratio"] > vol_thresh

        # ── LONG: Strict oversold bounce in uptrend ──────────────────
        if row["uptrend"]:
            # RSI actually crossed below the oversold level recently
            was_oversold = prev["rsi"] <= rsi_os or prev2["rsi"] <= rsi_os
            # RSI is now recovering above the level
            rsi_recovering = row["rsi"] > rsi_os and row["rsi"] > prev["rsi"]
            # Bullish candle confirms bounce
            bullish = row["bullish"]

            if was_oversold and rsi_recovering and bullish and vol_ok:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                last_signal_bar = i
                continue

        # ── SHORT: Strict overbought rejection in downtrend ──────────
        if row["downtrend"]:
            was_overbought = prev["rsi"] >= rsi_ob or prev2["rsi"] >= rsi_ob
            rsi_declining = row["rsi"] < rsi_ob and row["rsi"] < prev["rsi"]
            bearish = row["bearish"]

            if was_overbought and rsi_declining and bearish and vol_ok:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                last_signal_bar = i
                continue

    return df


def get_stop_loss(row, side):
    """Calculate stop-loss price based on ATR."""
    atr = row["atr"]
    mult = config.STRATEGY["atr_stop_multiplier"]
    if side == "LONG":
        return row["close"] - (atr * mult)
    return row["close"] + (atr * mult)


def get_take_profit(row, side):
    """Calculate take-profit price based on ATR."""
    atr = row["atr"]
    mult = config.STRATEGY["atr_tp_multiplier"]
    if side == "LONG":
        return row["close"] + (atr * mult)
    return row["close"] - (atr * mult)


def get_trailing_stop(entry_price, current_price, atr, side):
    """Calculate trailing stop level."""
    activation = config.STRATEGY["trailing_activation"] * atr
    trail = config.STRATEGY["trailing_step"] * atr
    if side == "LONG":
        if (current_price - entry_price) >= activation:
            return current_price - trail
    else:
        if (entry_price - current_price) >= activation:
            return current_price + trail
    return None
