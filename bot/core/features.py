"""Feature engineering pipeline for alpha extraction.

Extracts signals from OHLCV data, order-flow proxies (taker buy ratio),
volume profile, and volatility clustering indicators.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load Binance-format OHLCV CSV into a clean DataFrame."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    atr = _atr(high, low, close, period)
    plus_di = 100 * _ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _ema(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _ema(dx, period)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3
                ) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = _sma(k, d_period)
    return k, d


# ---------------------------------------------------------------------------
# Order-flow proxies
# ---------------------------------------------------------------------------

def _taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
    """Fraction of volume initiated by buyers (aggressor side)."""
    return df["taker_buy_base"] / df["volume"].replace(0, np.nan)


def _volume_delta(df: pd.DataFrame) -> pd.Series:
    """Net taker buy - taker sell volume (base asset)."""
    return 2 * df["taker_buy_base"] - df["volume"]


def _trade_intensity(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Z-scored trade count relative to rolling mean."""
    mean = df["trades"].rolling(period).mean()
    std = df["trades"].rolling(period).std().replace(0, np.nan)
    return (df["trades"] - mean) / std


def _vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP deviation from close price (normalised)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).rolling(period).sum()
    cum_vol = df["volume"].rolling(period).sum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol
    return (df["close"] - vwap) / vwap


# ---------------------------------------------------------------------------
# Volatility clustering features
# ---------------------------------------------------------------------------

def _realised_vol(close: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(24 * 365)  # annualised


def _vol_ratio(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    short_vol = _realised_vol(close, short)
    long_vol = _realised_vol(close, long)
    return short_vol / long_vol.replace(0, np.nan)


def _garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass volatility estimator (more efficient than close-to-close)."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window).mean().apply(np.sqrt)


# ---------------------------------------------------------------------------
# Momentum / mean-reversion features
# ---------------------------------------------------------------------------

def _returns(close: pd.Series, periods: List[int]) -> pd.DataFrame:
    """Log returns over multiple horizons."""
    out = {}
    for p in periods:
        out[f"ret_{p}"] = np.log(close / close.shift(p))
    return pd.DataFrame(out, index=close.index)


def _z_score(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std().replace(0, np.nan)
    return (s - mean) / std


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def make_labels(close: pd.Series, horizon: int = 6,
                threshold: float = 0.0) -> pd.Series:
    """Binary label: 1 if future return > threshold, else 0."""
    future_ret = close.shift(-horizon) / close - 1
    return (future_ret > threshold).astype(int)


def make_threshold_labels(close: pd.Series, horizon: int = 6,
                          threshold: float = 0.01) -> pd.Series:
    """Ternary-filtered binary label.

    Returns 1 if future return > +threshold, 0 if < -threshold, NaN otherwise.
    NaN rows are excluded from training, forcing the model to learn only
    from significant moves (not noise around zero).
    """
    future_ret = close.shift(-horizon) / close - 1
    label = pd.Series(np.nan, index=close.index)
    label[future_ret > threshold] = 1.0
    label[future_ret < -threshold] = 0.0
    return label


def make_regression_labels(close: pd.Series, horizon: int = 6) -> pd.Series:
    """Continuous label: future log return."""
    return np.log(close.shift(-horizon) / close)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(features: pd.DataFrame, labels: np.ndarray,
                    top_n: int = 30) -> List[str]:
    """Select top_n features by LightGBM importance on a quick fit.

    Runs a fast LightGBM to rank features, returns the column names to keep.
    This prevents overfitting from noisy low-importance features.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return list(features.columns[:top_n])

    X = np.nan_to_num(features.values.astype(np.float32))
    split = int(len(X) * 0.8)
    dtrain = lgb.Dataset(X[:split], label=labels[:split])
    dval = lgb.Dataset(X[split:], label=labels[split:], reference=dtrain)

    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5, "verbose": -1,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=100, valid_sets=[dval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    imp = model.feature_importance(importance_type="gain")
    ranked = pd.Series(imp, index=features.columns).sort_values(ascending=False)
    selected = list(ranked.head(top_n).index)
    return selected


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   lookbacks: Optional[List[int]] = None) -> pd.DataFrame:
    """Build the full feature matrix from raw OHLCV data.

    Returns a DataFrame with the same index as *df*, containing ~80+ features.
    Caller is responsible for dropping NaN rows.
    """
    if lookbacks is None:
        lookbacks = [5, 10, 20, 50, 100]

    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- Price-action ---
    for lb in lookbacks:
        feat[f"ema_{lb}"] = _ema(close, lb) / close - 1  # normalised distance
        feat[f"sma_{lb}"] = _sma(close, lb) / close - 1

    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_7"] = _rsi(close, 7)

    macd_l, macd_s, macd_h = _macd(close)
    feat["macd_line"] = macd_l / close
    feat["macd_signal"] = macd_s / close
    feat["macd_hist"] = macd_h / close

    feat["atr_14"] = _atr(high, low, close, 14) / close
    feat["atr_7"] = _atr(high, low, close, 7) / close

    bb_upper, bb_mid, bb_lower = _bollinger(close)
    feat["bb_width"] = (bb_upper - bb_lower) / bb_mid
    feat["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    feat["adx_14"] = _adx(high, low, close, 14)

    stoch_k, stoch_d = _stochastic(high, low, close)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d

    # --- Returns ---
    ret_df = _returns(close, lookbacks)
    feat = pd.concat([feat, ret_df], axis=1)

    # --- Volume / Order-flow ---
    feat["taker_buy_ratio"] = _taker_buy_ratio(df)
    feat["volume_delta"] = _volume_delta(df) / df["volume"].replace(0, np.nan)
    feat["trade_intensity"] = _trade_intensity(df)
    feat["vwap_dev"] = _vwap(df)

    for lb in [5, 10, 20]:
        feat[f"vol_sma_ratio_{lb}"] = (
            df["volume"] / df["volume"].rolling(lb).mean().replace(0, np.nan)
        )
        feat[f"tbr_sma_{lb}"] = _sma(_taker_buy_ratio(df), lb)

    feat["quote_vol_ratio"] = (
        df["quote_volume"] / df["quote_volume"].rolling(20).mean().replace(0, np.nan)
    )

    # --- Volatility clustering ---
    feat["realised_vol_20"] = _realised_vol(close, 20)
    feat["realised_vol_5"] = _realised_vol(close, 5)
    feat["vol_ratio"] = _vol_ratio(close)
    feat["gk_vol"] = _garman_klass_vol(df)

    # --- Z-scores for mean-reversion ---
    feat["close_z_20"] = _z_score(close, 20)
    feat["close_z_50"] = _z_score(close, 50)
    feat["volume_z_20"] = _z_score(df["volume"], 20)

    # --- Candle-body features ---
    body = (close - df["open"]).abs()
    total_range = (high - low).replace(0, np.nan)
    feat["body_ratio"] = body / total_range
    feat["upper_shadow"] = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / total_range
    feat["lower_shadow"] = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / total_range

    # --- Time features (cyclical encoding) ---
    hour = df.index.hour
    dow = df.index.dayofweek
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- Interaction features ---
    feat["rsi_vol_interaction"] = feat["rsi_14"] * feat["realised_vol_20"]
    feat["adx_macd_interaction"] = feat["adx_14"] * feat["macd_hist"]

    return feat
