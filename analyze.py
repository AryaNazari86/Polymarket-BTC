#!/usr/bin/env python3
"""Rule-based Polymarket BTC 5-minute odds analyzer."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from analysis_common import (
    build_feature_dataset,
    build_pattern_breakdown,
    build_windows,
    evaluate_buy_seconds,
    evaluate_sell_seconds,
    load_all_data,
    pick_best_rule,
)


def _fmt_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df.empty:
        return "(no rows)"
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if shown[col].dtype.kind in {"f"}:
            shown[col] = shown[col].map(lambda x: f"{x:,.2f}")
    return shown.to_string(index=False)


def _rule_to_english(rule: pd.Series, buy_second: int, sell_second: int) -> str:
    best_side = rule["best_side"]
    win_rate = rule["best_win_rate_pct"]
    avg_profit = rule["avg_up_profit_c"] if best_side == "Up" else rule["avg_down_profit_c"]
    count_windows = int(rule["count_windows"])

    return (
        f"When {rule['feature_pattern']} -> buy {best_side} at second {buy_second}, "
        f"sell at second {sell_second} -> {win_rate:.1f}% win rate, "
        f"avg profit {avg_profit:+.2f}c across {count_windows} windows."
    )


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    odds_dir = project_dir / "odds_data"

    try:
        data = load_all_data(odds_dir)
        windows = build_windows(data)

        step1 = evaluate_sell_seconds(windows, buy_second=60)
        best_sell_row = step1.loc[step1["combined_avg_profit_c"].idxmax()]
        target_sell_second = int(best_sell_row["sell_second"])

        step2 = evaluate_buy_seconds(windows, sell_second=target_sell_second)
        best_buy_row = step2.loc[step2["combined_avg_profit_c"].idxmax()]
        target_buy_second = int(best_buy_row["buy_second"])

        features = build_feature_dataset(
            windows,
            target_buy_second=target_buy_second,
            target_sell_second=target_sell_second,
        )
        pattern_df = build_pattern_breakdown(features)
        best_rule = pick_best_rule(pattern_df, min_windows=5)

    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    n_files = len(list(odds_dir.glob("*.csv")))
    print("\n=== POLYMARKET BTC 5M TOKEN-TRADING ANALYSIS (RULES) ===")
    print(f"CSV files loaded: {n_files}")
    print(f"Rows after cleaning: {len(data):,}")
    print(f"Usable windows: {len(windows):,}")

    print("\nStep 1 - Optimal Sell Time")
    print(f"Target sell second: {target_sell_second}")
    print(_fmt_table(step1.sort_values("combined_avg_profit_c", ascending=False).head(10), max_rows=10))

    print("\nStep 2 - Optimal Buy Time")
    print(f"Target buy second: {target_buy_second}")
    print(_fmt_table(step2.sort_values("combined_avg_profit_c", ascending=False).head(10), max_rows=10))

    print("\nStep 3 - Feature Pattern Breakdown")
    print(_fmt_table(pattern_df, max_rows=len(pattern_df)))

    print("\nStep 4 - Actionable Rule")
    if best_rule is None:
        print("No pattern had at least 5 windows. Collect more data to produce a stable rule.")
    else:
        print(_rule_to_english(best_rule, target_buy_second, target_sell_second))

    return 0


if __name__ == "__main__":
    sys.exit(main())
