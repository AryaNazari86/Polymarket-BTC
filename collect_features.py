#!/usr/bin/env python3
"""
Phase 1 — Feature pipeline for BTC 5-minute direction prediction.
Pulls Binance candle + trade data, engineers features, saves features.parquet.
Python 3.9 compatible.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
PARQUET_PATH = "features.parquet"

# ── helpers ──────────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 5) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    return {}


def fetch_klines(interval: str, limit: int = 1000, end_ms: Optional[int] = None) -> pd.DataFrame:
    params: Dict = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    if end_ms:
        params["endTime"] = end_ms
    data = _get(f"{BINANCE_BASE}/api/v3/klines", params)
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume", "taker_buy_vol", "quote_vol"]:
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    df["close_time"] = df["close_time"].astype(int)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def fetch_klines_range(interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all candles between start_ms and end_ms, paginating as needed."""
    frames = []
    cur_start = start_ms
    while cur_start < end_ms:
        params = {
            "symbol": SYMBOL, "interval": interval,
            "startTime": cur_start, "endTime": end_ms, "limit": 1000
        }
        data = _get(f"{BINANCE_BASE}/api/v3/klines", params)
        if not data:
            break
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol", "ignore"
        ])
        for c in ["open", "high", "low", "close", "volume", "taker_buy_vol", "quote_vol"]:
            df[c] = df[c].astype(float)
        df["open_time"] = df["open_time"].astype(int)
        df["close_time"] = df["close_time"].astype(int)
        frames.append(df)
        last_ts = int(df["open_time"].iloc[-1])
        if last_ts <= cur_start:
            break
        cur_start = last_ts + 1
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return result

# ── technical indicators ──────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    prev_cl = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    s = series.rolling(period).std()
    return mid - std * s, mid, mid + std * s


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction) where direction=1 means bullish."""
    hl2 = (df["high"] + df["low"]) / 2
    at = atr(df, period)
    upper_basic = hl2 + multiplier * at
    lower_basic = hl2 - multiplier * at

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = pd.Series(1, index=df.index)
    st = pd.Series(np.nan, index=df.index)

    close = df["close"]
    for i in range(1, len(df)):
        upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1]) if close.iloc[i-1] <= upper.iloc[i-1] else upper_basic.iloc[i]
        lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1]) if close.iloc[i-1] >= lower.iloc[i-1] else lower_basic.iloc[i]

        if st.iloc[i-1] == upper.iloc[i-1]:
            direction.iloc[i] = -1 if close.iloc[i] > upper.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < lower.iloc[i] else -1

        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

    return st, direction


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(window)


# ── feature builder ───────────────────────────────────────────────────────────

def build_features(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> pd.DataFrame:
    print("  Engineering features on 1m data...")

    # ── 1m indicators ──
    df_1m = df_1m.copy()
    df_1m["ema8"] = ema(df_1m["close"], 8)
    df_1m["ema55"] = ema(df_1m["close"], 55)
    df_1m["ema_spread"] = df_1m["ema8"] - df_1m["ema55"]
    df_1m["ema_slope"] = df_1m["ema8"].diff(3)
    df_1m["atr_1m"] = atr(df_1m, 14)
    df_1m["vwap_1m"] = vwap(df_1m)
    df_1m["vwap_dev"] = (df_1m["close"] - df_1m["vwap_1m"]) / df_1m["atr_1m"].replace(0, np.nan)
    df_1m["taker_buy_ratio_30s"] = df_1m["taker_buy_vol"].rolling(1).sum() / df_1m["volume"].rolling(1).sum().replace(0, np.nan)
    df_1m["taker_buy_ratio_2m"] = df_1m["taker_buy_vol"].rolling(2).sum() / df_1m["volume"].rolling(2).sum().replace(0, np.nan)
    df_1m["taker_buy_ratio_5m"] = df_1m["taker_buy_vol"].rolling(5).sum() / df_1m["volume"].rolling(5).sum().replace(0, np.nan)
    df_1m["rvol_5m"] = realized_vol(df_1m["close"], 5)
    df_1m["rvol_15m"] = realized_vol(df_1m["close"], 15)
    df_1m["rvol_1h"] = realized_vol(df_1m["close"], 60)

    _, st_dir = supertrend(df_1m, 10, 3.0)
    df_1m["st_dir"] = st_dir
    # bars since last flip
    flip = (df_1m["st_dir"] != df_1m["st_dir"].shift(1)).astype(int)
    df_1m["st_bars_since_flip"] = flip[::-1].cumsum()[::-1]  # recalc forward
    bars_since = []
    count = 0
    for v in flip:
        if v:
            count = 0
        else:
            count += 1
        bars_since.append(count)
    df_1m["st_bars_since_flip"] = bars_since

    print("  Engineering features on 5m data...")
    df_5m = df_5m.copy()
    df_5m["rsi_5m"] = rsi(df_5m["close"], 14)
    df_5m["atr_5m"] = atr(df_5m, 14)
    bb_lo, bb_mid, bb_hi = bollinger(df_5m["close"], 20, 2.0)
    df_5m["bb_pos_5m"] = (df_5m["close"] - bb_lo) / (bb_hi - bb_lo).replace(0, np.nan)
    df_5m["ema8_5m"] = ema(df_5m["close"], 8)

    print("  Engineering features on 1h data...")
    df_1h = df_1h.copy()
    df_1h["rsi_1h"] = rsi(df_1h["close"], 14)
    bb_lo_1h, _, bb_hi_1h = bollinger(df_1h["close"], 20, 2.0)
    df_1h["bb_pos_1h"] = (df_1h["close"] - bb_lo_1h) / (bb_hi_1h - bb_lo_1h).replace(0, np.nan)
    df_1h["ema8_1h"] = ema(df_1h["close"], 8)

    print("  Merging timeframes onto 5m candles...")
    # Align 1m features to 5m candles (use last 1m bar before each 5m close)
    df_1m_indexed = df_1m.set_index("open_time")
    df_5m_indexed = df_5m.set_index("open_time")
    df_1h_indexed = df_1h.set_index("open_time")

    rows: List[dict] = []

    for i, row5 in df_5m.iterrows():
        ot5 = int(row5["open_time"])
        ct5 = int(row5["close_time"])

        # 1m bars within this 5m window
        mask_1m = (df_1m["open_time"] >= ot5) & (df_1m["open_time"] <= ct5)
        sub_1m = df_1m[mask_1m]

        if sub_1m.empty:
            continue

        last_1m = sub_1m.iloc[-1]

        # 1h bar covering this 5m open
        mask_1h = df_1h["open_time"] <= ot5
        sub_1h = df_1h[mask_1h]
        last_1h = sub_1h.iloc[-1] if not sub_1h.empty else None

        # time-of-day encoding
        dt_utc = datetime.fromtimestamp(ot5 / 1000, tz=timezone.utc)
        minute_of_day = dt_utc.hour * 60 + dt_utc.minute
        tod_sin = math.sin(2 * math.pi * minute_of_day / 1440)
        tod_cos = math.cos(2 * math.pi * minute_of_day / 1440)
        hour_sin = math.sin(2 * math.pi * dt_utc.hour / 24)
        hour_cos = math.cos(2 * math.pi * dt_utc.hour / 24)

        # distance from round $500
        price = float(row5["close"])
        dist_round = abs(price % 500 - 250) / 250  # 0=nearest, 1=furthest

        feat: dict = {
            "open_time": ot5,
            "datetime_utc": dt_utc.strftime("%Y-%m-%d %H:%M"),
            # 5m candle basics
            "open_5m": float(row5["open"]),
            "close_5m": float(row5["close"]),
            "high_5m": float(row5["high"]),
            "low_5m": float(row5["low"]),
            "volume_5m": float(row5["volume"]),
            # 5m indicators
            "rsi_5m": float(row5["rsi_5m"]) if not pd.isna(row5["rsi_5m"]) else np.nan,
            "atr_5m": float(row5["atr_5m"]) if not pd.isna(row5["atr_5m"]) else np.nan,
            "bb_pos_5m": float(row5["bb_pos_5m"]) if not pd.isna(row5["bb_pos_5m"]) else np.nan,
            "ema8_5m": float(row5["ema8_5m"]) if not pd.isna(row5["ema8_5m"]) else np.nan,
            # 1m indicators (last bar in this 5m window)
            "ema8_1m": float(last_1m["ema8"]),
            "ema55_1m": float(last_1m["ema55"]),
            "ema_spread_1m": float(last_1m["ema_spread"]),
            "ema_slope_1m": float(last_1m["ema_slope"]),
            "atr_1m": float(last_1m["atr_1m"]),
            "vwap_dev_1m": float(last_1m["vwap_dev"]) if not pd.isna(last_1m["vwap_dev"]) else np.nan,
            "taker_buy_ratio_30s": float(last_1m["taker_buy_ratio_30s"]) if not pd.isna(last_1m["taker_buy_ratio_30s"]) else np.nan,
            "taker_buy_ratio_2m": float(last_1m["taker_buy_ratio_2m"]) if not pd.isna(last_1m["taker_buy_ratio_2m"]) else np.nan,
            "taker_buy_ratio_5m": float(last_1m["taker_buy_ratio_5m"]) if not pd.isna(last_1m["taker_buy_ratio_5m"]) else np.nan,
            "rvol_5m": float(last_1m["rvol_5m"]) if not pd.isna(last_1m["rvol_5m"]) else np.nan,
            "rvol_15m": float(last_1m["rvol_15m"]) if not pd.isna(last_1m["rvol_15m"]) else np.nan,
            "rvol_1h": float(last_1m["rvol_1h"]) if not pd.isna(last_1m["rvol_1h"]) else np.nan,
            "st_dir_1m": float(last_1m["st_dir"]),
            "st_bars_since_flip_1m": float(last_1m["st_bars_since_flip"]),
            # 1h indicators
            "rsi_1h": float(last_1h["rsi_1h"]) if last_1h is not None and not pd.isna(last_1h["rsi_1h"]) else np.nan,
            "bb_pos_1h": float(last_1h["bb_pos_1h"]) if last_1h is not None and not pd.isna(last_1h["bb_pos_1h"]) else np.nan,
            "ema8_1h": float(last_1h["ema8_1h"]) if last_1h is not None and not pd.isna(last_1h["ema8_1h"]) else np.nan,
            # cross-timeframe divergence: 1m EMA dir vs 1h EMA dir
            "ema_divergence": float(last_1m["ema_slope"]) * (float(last_1h["ema8_1h"]) - float(last_1h["close"])) if last_1h is not None else np.nan,
            # time-of-day
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            # round number distance
            "dist_round_500": dist_round,
        }
        rows.append(feat)

    df_feat = pd.DataFrame(rows)

    print("  Computing target variable...")
    df_feat["candle_return"] = (df_feat["close_5m"] - df_feat["open_5m"]) / df_feat["open_5m"]
    # Exclude doji candles (< 0.02% move)
    df_feat["is_doji"] = df_feat["candle_return"].abs() < 0.0002
    df_feat["target"] = (df_feat["candle_return"] > 0).astype(int)
    df_feat = df_feat[~df_feat["is_doji"]].copy()

    return df_feat


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now_ms = int(time.time() * 1000)
    six_months_ms = 180 * 24 * 60 * 60 * 1000
    start_ms = now_ms - six_months_ms

    print("Fetching 6 months of Binance candle data...")
    print("  1m candles (this may take a few minutes)...")
    df_1m = fetch_klines_range("1m", start_ms, now_ms)
    print(f"  Got {len(df_1m):,} 1m candles")

    print("  5m candles...")
    df_5m = fetch_klines_range("5m", start_ms, now_ms)
    print(f"  Got {len(df_5m):,} 5m candles")

    print("  1h candles...")
    df_1h = fetch_klines_range("1h", start_ms, now_ms)
    print(f"  Got {len(df_1h):,} 1h candles")

    print("  4h candles...")
    df_4h = fetch_klines_range("4h", start_ms, now_ms)
    print(f"  Got {len(df_4h):,} 4h candles")

    print("Building feature matrix...")
    df_feat = build_features(df_1m, df_5m, df_1h, df_4h)

    print(f"  Total rows (non-doji 5m candles): {len(df_feat):,}")
    print(f"  Target distribution: {df_feat['target'].mean():.1%} UP, {1-df_feat['target'].mean():.1%} DOWN")

    print(f"Saving to {PARQUET_PATH}...")
    df_feat.to_parquet(PARQUET_PATH, index=False)
    print("Done. features.parquet saved.")


if __name__ == "__main__":
    main()
