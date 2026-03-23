#!/usr/bin/env python3
"""
Phase 3 — Live signal bot for BTC 5-minute direction prediction.
At the start of each 5-minute window, loads models, builds feature vector,
prints BUY UP / BUY DOWN / STAY OUT signal. Python 3.9 compatible.
"""

from __future__ import annotations

import math
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
MODELS_DIR = "models"
WINDOW_SECONDS = 300
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[38;5;82m"
RED = "\033[38;5;203m"
YELLOW = "\033[38;5;220m"
CYAN = "\033[38;5;87m"
DIM = "\033[2m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# ── Binance helpers ───────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 4) -> object:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(1)
    return None


def fetch_klines(interval: str, limit: int) -> Optional[pd.DataFrame]:
    data = _get(f"{BINANCE_BASE}/api/v3/klines",
                {"symbol": SYMBOL, "interval": interval, "limit": limit})
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume", "taker_buy_vol", "quote_vol"]:
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    return df.sort_values("open_time").reset_index(drop=True)


def get_btc_price() -> Optional[float]:
    data = _get(f"{BINANCE_BASE}/api/v3/ticker/price", {"symbol": SYMBOL})
    if data and "price" in data:
        return float(data["price"])
    return None


# ── indicator functions (same as collect_features.py) ─────────────────────────

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


def bollinger_pos(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    mid = series.rolling(period).mean()
    s = series.rolling(period).std()
    lo = mid - std * s
    hi = mid + std * s
    return (series - lo) / (hi - lo).replace(0, np.nan)


def supertrend_dir(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2
    at = atr(df, period)
    upper_basic = hl2 + multiplier * at
    lower_basic = hl2 - multiplier * at
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = pd.Series(1, index=df.index, dtype=float)
    close = df["close"]
    for i in range(1, len(df)):
        upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1]) if close.iloc[i-1] <= upper.iloc[i-1] else upper_basic.iloc[i]
        lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1]) if close.iloc[i-1] >= lower.iloc[i-1] else lower_basic.iloc[i]
        prev_st = lower.iloc[i-1] if direction.iloc[i-1] == 1 else upper.iloc[i-1]
        if prev_st == upper.iloc[i-1]:
            direction.iloc[i] = -1.0 if close.iloc[i] > upper.iloc[i] else 1.0
        else:
            direction.iloc[i] = 1.0 if close.iloc[i] < lower.iloc[i] else -1.0
    return direction


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(window)


def vwap_series(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)


# ── feature vector builder ────────────────────────────────────────────────────

def build_live_feature(feature_cols: List[str]) -> Optional[pd.DataFrame]:
    """Pull live Binance data and build a single-row feature DataFrame."""
    df_1m = fetch_klines("1m", 120)
    df_5m = fetch_klines("5m", 60)
    df_1h = fetch_klines("1h", 40)

    if df_1m is None or df_5m is None or df_1h is None:
        return None
    if len(df_1m) < 60 or len(df_5m) < 25 or len(df_1h) < 22:
        return None

    # 1m indicators
    df_1m["ema8"] = ema(df_1m["close"], 8)
    df_1m["ema55"] = ema(df_1m["close"], 55)
    df_1m["ema_spread"] = df_1m["ema8"] - df_1m["ema55"]
    df_1m["ema_slope"] = df_1m["ema8"].diff(3)
    df_1m["atr_1m"] = atr(df_1m, 14)
    df_1m["vwap_1m"] = vwap_series(df_1m)
    df_1m["vwap_dev"] = (df_1m["close"] - df_1m["vwap_1m"]) / df_1m["atr_1m"].replace(0, np.nan)
    df_1m["tbr_30s"] = df_1m["taker_buy_vol"].rolling(1).sum() / df_1m["volume"].rolling(1).sum().replace(0, np.nan)
    df_1m["tbr_2m"] = df_1m["taker_buy_vol"].rolling(2).sum() / df_1m["volume"].rolling(2).sum().replace(0, np.nan)
    df_1m["tbr_5m"] = df_1m["taker_buy_vol"].rolling(5).sum() / df_1m["volume"].rolling(5).sum().replace(0, np.nan)
    df_1m["rvol_5m_val"] = realized_vol(df_1m["close"], 5)
    df_1m["rvol_15m_val"] = realized_vol(df_1m["close"], 15)
    df_1m["rvol_1h_val"] = realized_vol(df_1m["close"], 60)
    st_dir = supertrend_dir(df_1m)
    df_1m["st_dir"] = st_dir
    bars_since = []
    count = 0
    flip = (df_1m["st_dir"] != df_1m["st_dir"].shift(1)).astype(int).tolist()
    for v in flip:
        count = 0 if v else count + 1
        bars_since.append(count)
    df_1m["st_bars_since_flip"] = bars_since

    # 5m indicators
    df_5m["rsi_5m_val"] = rsi(df_5m["close"], 14)
    df_5m["atr_5m_val"] = atr(df_5m, 14)
    df_5m["bb_pos_5m_val"] = bollinger_pos(df_5m["close"], 20, 2.0)
    df_5m["ema8_5m_val"] = ema(df_5m["close"], 8)

    # 1h indicators
    df_1h["rsi_1h_val"] = rsi(df_1h["close"], 14)
    df_1h["bb_pos_1h_val"] = bollinger_pos(df_1h["close"], 20, 2.0)
    df_1h["ema8_1h_val"] = ema(df_1h["close"], 8)

    last_1m = df_1m.iloc[-1]
    last_5m = df_5m.iloc[-1]
    last_1h = df_1h.iloc[-1]

    now_ms = int(time.time() * 1000)
    dt_utc = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    minute_of_day = dt_utc.hour * 60 + dt_utc.minute
    price = float(last_1m["close"])

    feat = {
        "rsi_5m": float(last_5m["rsi_5m_val"]),
        "atr_5m": float(last_5m["atr_5m_val"]),
        "bb_pos_5m": float(last_5m["bb_pos_5m_val"]),
        "ema8_5m": float(last_5m["ema8_5m_val"]),
        "ema8_1m": float(last_1m["ema8"]),
        "ema55_1m": float(last_1m["ema55"]),
        "ema_spread_1m": float(last_1m["ema_spread"]),
        "ema_slope_1m": float(last_1m["ema_slope"]),
        "atr_1m": float(last_1m["atr_1m"]),
        "vwap_dev_1m": float(last_1m["vwap_dev"]) if not pd.isna(last_1m["vwap_dev"]) else 0.0,
        "taker_buy_ratio_30s": float(last_1m["tbr_30s"]) if not pd.isna(last_1m["tbr_30s"]) else 0.5,
        "taker_buy_ratio_2m": float(last_1m["tbr_2m"]) if not pd.isna(last_1m["tbr_2m"]) else 0.5,
        "taker_buy_ratio_5m": float(last_1m["tbr_5m"]) if not pd.isna(last_1m["tbr_5m"]) else 0.5,
        "rvol_5m": float(last_1m["rvol_5m_val"]) if not pd.isna(last_1m["rvol_5m_val"]) else 0.0,
        "rvol_15m": float(last_1m["rvol_15m_val"]) if not pd.isna(last_1m["rvol_15m_val"]) else 0.0,
        "rvol_1h": float(last_1m["rvol_1h_val"]) if not pd.isna(last_1m["rvol_1h_val"]) else 0.0,
        "st_dir_1m": float(last_1m["st_dir"]),
        "st_bars_since_flip_1m": float(last_1m["st_bars_since_flip"]),
        "rsi_1h": float(last_1h["rsi_1h_val"]) if not pd.isna(last_1h["rsi_1h_val"]) else 50.0,
        "bb_pos_1h": float(last_1h["bb_pos_1h_val"]) if not pd.isna(last_1h["bb_pos_1h_val"]) else 0.5,
        "ema8_1h": float(last_1h["ema8_1h_val"]),
        "ema_divergence": float(last_1m["ema_slope"]) * (float(last_1h["ema8_1h_val"]) - float(last_1h["close"])),
        "tod_sin": math.sin(2 * math.pi * minute_of_day / 1440),
        "tod_cos": math.cos(2 * math.pi * minute_of_day / 1440),
        "hour_sin": math.sin(2 * math.pi * dt_utc.hour / 24),
        "hour_cos": math.cos(2 * math.pi * dt_utc.hour / 24),
        "dist_round_500": abs(price % 500 - 250) / 250,
    }

    row = pd.DataFrame([feat])[feature_cols]
    row = row.fillna(0)
    return row


# ── model loading ─────────────────────────────────────────────────────────────

def load_models():
    with open(f"{MODELS_DIR}/lgbm.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(f"{MODELS_DIR}/xgb.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(f"{MODELS_DIR}/cat.pkl", "rb") as f:
        cat = pickle.load(f)
    with open(f"{MODELS_DIR}/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return lgbm, xgb, cat, feature_cols


def predict(lgbm, xgb_m, cat, X: pd.DataFrame) -> float:
    p_lgbm = lgbm.predict_proba(X)[0][1]
    p_xgb = xgb_m.predict_proba(X)[0][1]
    p_cat = cat.predict_proba(X)[0][1]
    return float((p_lgbm + p_xgb + p_cat) / 3)


# ── display ───────────────────────────────────────────────────────────────────

def print_signal(window_start: datetime, window_end: datetime, btc_price: Optional[float], prob: float) -> None:
    ws = window_start.strftime("%b %d  %H:%M")
    we = window_end.strftime("%H:%M UTC")
    btc_str = f"${btc_price:,.2f}" if btc_price else "N/A"
    pct = prob * 100
    direction = "UP" if prob > 0.5 else "DOWN"
    conf_pct = pct if prob > 0.5 else 100 - pct

    print("\n" + "═" * 50)
    print(f"  WINDOW: {ws} – {we}")
    print(f"  BTC now: {btc_str}")

    if prob > BUY_THRESHOLD:
        print(_c(GREEN + BOLD, f"  Model confidence: {conf_pct:.0f}% UP"))
        print(_c(GREEN + BOLD, "  ★ SIGNAL: BUY UP"))
        fire_pct = 100 * (1 - BUY_THRESHOLD) * 2
        print(_c(DIM, f"  Expected edge: model fires ~{fire_pct:.0f}% of windows at this threshold"))
    elif prob < SELL_THRESHOLD:
        print(_c(RED + BOLD, f"  Model confidence: {conf_pct:.0f}% DOWN"))
        print(_c(RED + BOLD, "  ★ SIGNAL: BUY DOWN"))
        fire_pct = 100 * (1 - BUY_THRESHOLD) * 2
        print(_c(DIM, f"  Expected edge: model fires ~{fire_pct:.0f}% of windows at this threshold"))
    else:
        print(_c(YELLOW, f"  Model confidence: {conf_pct:.0f}% {direction}  (uncertain)"))
        print(_c(DIM, "  ✗ STAY OUT this window"))

    print("═" * 50 + "\n")


def print_result(signal_prob: float, actual_up: bool, win_count: int, total_count: int) -> None:
    direction = "UP" if signal_prob > BUY_THRESHOLD else "DOWN" if signal_prob < SELL_THRESHOLD else None
    if direction is None:
        return

    predicted_up = direction == "UP"
    correct = predicted_up == actual_up
    candle_str = "closed UP ▲" if actual_up else "closed DOWN ▼"
    result_str = _c(GREEN + BOLD, "✓ CORRECT") if correct else _c(RED + BOLD, "✗ WRONG")
    win_rate = win_count / total_count * 100 if total_count > 0 else 0.0

    print(f"  Result: Candle {candle_str}  {result_str}")
    print(f"  Running win rate: {win_count}/{total_count} ({win_rate:.1f}%)\n")


# ── main loop ─────────────────────────────────────────────────────────────────

def get_window_boundaries() -> tuple:
    """Return (window_start_dt, window_end_dt, window_end_unix)."""
    now = int(time.time())
    window_end_unix = ((now // WINDOW_SECONDS) + 1) * WINDOW_SECONDS
    window_start_unix = window_end_unix - WINDOW_SECONDS
    ws = datetime.fromtimestamp(window_start_unix, tz=timezone.utc)
    we = datetime.fromtimestamp(window_end_unix, tz=timezone.utc)
    return ws, we, window_end_unix


def wait_for_next_window() -> int:
    """Block until the next 5-minute window starts, return its end unix ts."""
    now = int(time.time())
    current_window_end = ((now // WINDOW_SECONDS) + 1) * WINDOW_SECONDS
    sleep_for = current_window_end - now
    if sleep_for > 2:
        print(_c(DIM, f"  Waiting {sleep_for}s for next window..."))
        time.sleep(sleep_for)
    return current_window_end


def main() -> int:
    print(_c(BOLD, "\n  ██████╗ ████████╗ ██████╗"))
    print(_c(BOLD,   "  ██╔══██╗╚══██╔══╝██╔════╝"))
    print(_c(BOLD,   "  ██████╔╝   ██║   ██║"))
    print(_c(BOLD,   "  ██╔══██╗   ██║   ██║"))
    print(_c(BOLD,   "  ██████╔╝   ██║   ╚██████╗"))
    print(_c(BOLD,   "  ╚═════╝    ╚═╝    ╚═════╝"))
    print()
    print(_c(CYAN + BOLD, "  ▄▄▄  ▄▄▄  ▄  ▄  ▄▄▄"))
    print(_c(CYAN + BOLD, "  █▄█  █▄▀  ▀▄▀  █▄█"))
    print(_c(CYAN + BOLD, "  █  █ █ ▀   █   █  █"))
    print()
    print("  Polymarket BTC Signal Bot")
    print("  Powered by LightGBM + XGBoost + CatBoost ensemble\n")

    print("Loading models...")
    try:
        lgbm, xgb_m, cat, feature_cols = load_models()
        print("  Models loaded.\n")
    except FileNotFoundError:
        print(_c(RED + BOLD, "\n  ERROR: No trained models found."))
        print("  Run: python3 collect_features.py && python3 train.py first.\n")
        return 1

    win_count = 0
    total_count = 0
    last_signal_prob: Optional[float] = None
    last_window_end: Optional[int] = None

    while True:
        now = int(time.time())
        window_end_unix = ((now // WINDOW_SECONDS) + 1) * WINDOW_SECONDS
        window_start_unix = window_end_unix - WINDOW_SECONDS
        seconds_in = now - window_start_unix

        # ── check result of previous window ──
        if last_window_end is not None and last_window_end != window_end_unix and last_signal_prob is not None:
            df_prev = fetch_klines("5m", 3)
            if df_prev is not None and len(df_prev) >= 2:
                prev_candle = df_prev.iloc[-2]
                actual_up = float(prev_candle["close"]) > float(prev_candle["open"])
                direction = None
                if last_signal_prob > BUY_THRESHOLD:
                    direction = "UP"
                elif last_signal_prob < SELL_THRESHOLD:
                    direction = "DOWN"
                if direction is not None:
                    predicted_up = direction == "UP"
                    total_count += 1
                    if predicted_up == actual_up:
                        win_count += 1
                print_result(last_signal_prob, actual_up, win_count, total_count)

        # ── new window: wait if we're too early ──
        if seconds_in > 10:
            # already inside the window, wait for next one
            sleep_for = window_end_unix - now + 1
            if sleep_for > 0:
                print(_c(DIM, f"  {seconds_in}s into current window — waiting {sleep_for}s for next window start..."))
                time.sleep(sleep_for)
            continue

        # ── build features and predict ──
        ws_dt = datetime.fromtimestamp(window_start_unix, tz=timezone.utc)
        we_dt = datetime.fromtimestamp(window_end_unix, tz=timezone.utc)

        print(_c(DIM, f"  Fetching live data for window {ws_dt.strftime('%H:%M')}–{we_dt.strftime('%H:%M')} UTC..."))
        X = build_live_feature(feature_cols)
        btc_price = get_btc_price()

        if X is None:
            print(_c(YELLOW, "  Could not build feature vector (API error). Skipping window."))
            time.sleep(WINDOW_SECONDS)
            continue

        prob = predict(lgbm, xgb_m, cat, X)
        last_signal_prob = prob
        last_window_end = window_end_unix

        print_signal(ws_dt, we_dt, btc_price, prob)

        # sleep until ~5s before this window ends to check result
        sleep_to = window_end_unix - now - 5
        if sleep_to > 0:
            time.sleep(sleep_to)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  Signal bot stopped.\n")
        sys.exit(0)
