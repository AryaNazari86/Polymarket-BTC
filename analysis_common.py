#!/usr/bin/env python3
"""Shared data loading + feature engineering for Polymarket 5m odds analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = {
    "window_id",
    "second_in_window",
    "up_cents",
    "down_cents",
    "btc_price_usd",
    "market_slug",
}


@dataclass
class WindowSnapshot:
    key: str
    window_id: str
    market_slug: str
    seconds: pd.Index
    up: pd.Series
    down: pd.Series
    btc: pd.Series


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_all_data(odds_dir: Path) -> pd.DataFrame:
    csv_files = sorted([p for p in odds_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {odds_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing required columns: {sorted(missing)}")
        df["source_file"] = csv_path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = _coerce_numeric(data, ["second_in_window", "up_cents", "down_cents", "btc_price_usd"])
    data = data.dropna(subset=["second_in_window", "up_cents", "down_cents", "btc_price_usd"])
    data["second_in_window"] = data["second_in_window"].astype(int)
    data = data[(data["second_in_window"] >= 0) & (data["second_in_window"] <= 299)].copy()

    data["window_id"] = data["window_id"].astype(str)
    data["market_slug"] = data["market_slug"].astype(str)
    data["window_key"] = data["window_id"] + "__" + data["market_slug"]

    sort_cols = ["window_key", "second_in_window"]
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
        sort_cols.append("timestamp")
    data = data.sort_values(sort_cols)
    data = data.drop_duplicates(subset=["window_key", "second_in_window"], keep="last")

    return data.reset_index(drop=True)


def build_windows(data: pd.DataFrame) -> Dict[str, WindowSnapshot]:
    windows: Dict[str, WindowSnapshot] = {}

    for key, group in data.groupby("window_key", sort=False):
        group = group.sort_values("second_in_window")
        sec_df = group.set_index("second_in_window")
        window_id = str(group["window_id"].iloc[0])
        slug = str(group["market_slug"].iloc[0])
        windows[key] = WindowSnapshot(
            key=key,
            window_id=window_id,
            market_slug=slug,
            seconds=sec_df.index,
            up=sec_df["up_cents"],
            down=sec_df["down_cents"],
            btc=sec_df["btc_price_usd"],
        )

    if not windows:
        raise ValueError("No usable window data after cleaning.")

    return windows


def _window_profit_at_times(window: WindowSnapshot, buy_second: int, sell_second: int) -> Optional[Tuple[float, float]]:
    needed = {buy_second, sell_second}
    if not needed.issubset(set(window.seconds)):
        return None
    up_profit = float(window.up.loc[sell_second] - window.up.loc[buy_second])
    down_profit = float(window.down.loc[sell_second] - window.down.loc[buy_second])
    return up_profit, down_profit


def evaluate_sell_seconds(windows: Dict[str, WindowSnapshot], buy_second: int = 60) -> pd.DataFrame:
    rows = []
    for sell_second in range(240, 286):
        up_profits: List[float] = []
        down_profits: List[float] = []
        for window in windows.values():
            result = _window_profit_at_times(window, buy_second=buy_second, sell_second=sell_second)
            if result is None:
                continue
            up_p, down_p = result
            up_profits.append(up_p)
            down_profits.append(down_p)

        if not up_profits:
            continue

        avg_up = sum(up_profits) / len(up_profits)
        avg_down = sum(down_profits) / len(down_profits)
        combined = (avg_up + avg_down) / 2.0
        rows.append(
            {
                "sell_second": sell_second,
                "n_windows": len(up_profits),
                "avg_up_profit_c": avg_up,
                "avg_down_profit_c": avg_down,
                "combined_avg_profit_c": combined,
            }
        )

    if not rows:
        raise ValueError("No windows contain required seconds for Step 1.")

    return pd.DataFrame(rows).sort_values("sell_second").reset_index(drop=True)


def evaluate_buy_seconds(windows: Dict[str, WindowSnapshot], sell_second: int) -> pd.DataFrame:
    rows = []
    for buy_second in range(30, 151):
        up_profits: List[float] = []
        down_profits: List[float] = []
        for window in windows.values():
            result = _window_profit_at_times(window, buy_second=buy_second, sell_second=sell_second)
            if result is None:
                continue
            up_p, down_p = result
            up_profits.append(up_p)
            down_profits.append(down_p)

        if not up_profits:
            continue

        avg_up = sum(up_profits) / len(up_profits)
        avg_down = sum(down_profits) / len(down_profits)
        combined = (avg_up + avg_down) / 2.0
        rows.append(
            {
                "buy_second": buy_second,
                "n_windows": len(up_profits),
                "avg_up_profit_c": avg_up,
                "avg_down_profit_c": avg_down,
                "combined_avg_profit_c": combined,
            }
        )

    if not rows:
        raise ValueError("No windows contain required seconds for Step 2.")

    return pd.DataFrame(rows).sort_values("buy_second").reset_index(drop=True)


def _bucket_up_price(value: float) -> str:
    if value < 45.0:
        return "lt45"
    if value > 55.0:
        return "gt55"
    return "45to55"


def _bucket_slope(value: float, flat_threshold: float = 0.01) -> str:
    if value > flat_threshold:
        return "positive"
    if value < -flat_threshold:
        return "negative"
    return "flat"


def _bucket_spread(value: float, neutral_threshold: float = 5.0) -> str:
    if value > neutral_threshold:
        return "up_dominant"
    if value < -neutral_threshold:
        return "down_dominant"
    return "neutral"


def _parse_window_ts(window: WindowSnapshot) -> int:
    # window_id is usually unix end timestamp; fallback to slug suffix (start ts)
    try:
        return int(float(window.window_id))
    except ValueError:
        pass

    parts = window.market_slug.split("-")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def build_feature_dataset(
    windows: Dict[str, WindowSnapshot],
    target_buy_second: int,
    target_sell_second: int,
) -> pd.DataFrame:
    feature_rows = []

    for window in windows.values():
        required_points = {0, 10, 20, 30, target_buy_second, target_sell_second}
        if not required_points.issubset(set(window.seconds)):
            continue

        early = window.up[window.up.index <= target_buy_second]
        if early.empty:
            continue

        up_0 = float(window.up.loc[0])
        up_10 = float(window.up.loc[10])
        up_20 = float(window.up.loc[20])
        up_30 = float(window.up.loc[30])

        btc_0 = float(window.btc.loc[0])
        btc_30 = float(window.btc.loc[30])

        up_slope = (up_30 - up_0) / 30.0
        btc_slope = (btc_30 - btc_0) / 30.0

        crossed_below_45 = bool((early < 45.0).any())
        crossed_above_55 = bool((early > 55.0).any())

        spread_30 = float(window.up.loc[30] - window.down.loc[30])

        up_profit = float(window.up.loc[target_sell_second] - window.up.loc[target_buy_second])
        down_profit = float(window.down.loc[target_sell_second] - window.down.loc[target_buy_second])

        feature_rows.append(
            {
                "window_key": window.key,
                "window_ts": _parse_window_ts(window),
                "window_id": window.window_id,
                "market_slug": window.market_slug,
                "target_buy_second": target_buy_second,
                "target_sell_second": target_sell_second,
                "up_0": up_0,
                "up_10": up_10,
                "up_20": up_20,
                "up_30": up_30,
                "up_0_bucket": _bucket_up_price(up_0),
                "up_10_bucket": _bucket_up_price(up_10),
                "up_20_bucket": _bucket_up_price(up_20),
                "up_30_bucket": _bucket_up_price(up_30),
                "up_slope_0_30": up_slope,
                "up_slope_dir": _bucket_slope(up_slope),
                "btc_slope_0_30": btc_slope,
                "btc_slope_dir": _bucket_slope(btc_slope),
                "crossed_below_45_before_buy": crossed_below_45,
                "crossed_above_55_before_buy": crossed_above_55,
                "spread_30": spread_30,
                "spread_30_bucket": _bucket_spread(spread_30),
                "up_profit_c": up_profit,
                "down_profit_c": down_profit,
                "up_win": up_profit > 0.0,
                "down_win": down_profit > 0.0,
            }
        )

    if not feature_rows:
        raise ValueError("No windows contain required points for Step 3 feature extraction.")

    return pd.DataFrame(feature_rows).sort_values("window_ts").reset_index(drop=True)


def build_pattern_breakdown(features_df: pd.DataFrame) -> pd.DataFrame:
    pattern_cols = [
        "up_0_bucket",
        "up_10_bucket",
        "up_20_bucket",
        "up_30_bucket",
        "up_slope_dir",
        "btc_slope_dir",
        "crossed_below_45_before_buy",
        "crossed_above_55_before_buy",
        "spread_30_bucket",
    ]

    grouped = (
        features_df.groupby(pattern_cols, dropna=False)
        .agg(
            count_windows=("window_key", "count"),
            up_win_rate=("up_win", "mean"),
            down_win_rate=("down_win", "mean"),
            avg_up_profit_c=("up_profit_c", "mean"),
            avg_down_profit_c=("down_profit_c", "mean"),
        )
        .reset_index()
    )

    grouped["up_win_rate_pct"] = grouped["up_win_rate"] * 100.0
    grouped["down_win_rate_pct"] = grouped["down_win_rate"] * 100.0
    grouped["best_win_rate_pct"] = grouped[["up_win_rate_pct", "down_win_rate_pct"]].max(axis=1)
    grouped["best_side"] = grouped.apply(
        lambda row: "Up" if row["up_win_rate_pct"] >= row["down_win_rate_pct"] else "Down", axis=1
    )
    grouped = grouped.sort_values(
        ["best_win_rate_pct", "count_windows", "avg_up_profit_c", "avg_down_profit_c"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    grouped["feature_pattern"] = grouped.apply(
        lambda row: (
            f"up0={row['up_0_bucket']}, up10={row['up_10_bucket']}, up20={row['up_20_bucket']}, "
            f"up30={row['up_30_bucket']}, up_slope={row['up_slope_dir']}, btc_slope={row['btc_slope_dir']}, "
            f"below45={bool(row['crossed_below_45_before_buy'])}, above55={bool(row['crossed_above_55_before_buy'])}, "
            f"spread30={row['spread_30_bucket']}"
        ),
        axis=1,
    )

    cols = [
        "feature_pattern",
        "count_windows",
        "up_win_rate_pct",
        "down_win_rate_pct",
        "avg_up_profit_c",
        "avg_down_profit_c",
        "best_side",
        "best_win_rate_pct",
    ]
    return grouped[cols]


def pick_best_rule(pattern_df: pd.DataFrame, min_windows: int = 5) -> Optional[pd.Series]:
    eligible = pattern_df[pattern_df["count_windows"] >= min_windows].copy()
    if eligible.empty:
        return None

    eligible["best_avg_profit_c"] = eligible.apply(
        lambda row: row["avg_up_profit_c"] if row["best_side"] == "Up" else row["avg_down_profit_c"], axis=1
    )

    eligible = eligible.sort_values(
        ["best_win_rate_pct", "best_avg_profit_c", "count_windows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return eligible.iloc[0]
