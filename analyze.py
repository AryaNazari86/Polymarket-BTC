#!/usr/bin/env python3
"""Rule-based Polymarket BTC 5-minute odds analyzer."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from analysis_common import (
    build_coarse_pattern_breakdown,
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
    return (
        f"When {rule['feature_pattern']} -> buy {rule['best_side']} at second {buy_second}, "
        f"sell at second {sell_second} -> win rate {rule['best_win_rate_pct']:.1f}% "
        f"(95% floor {rule['best_win_lb95_pct']:.1f}%), avg profit {rule['best_avg_profit_c']:+.2f}c "
        f"(95% floor {rule['best_profit_lb95_c']:+.2f}c), n={int(rule['count_windows'])}."
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
        coarse_pattern_df = build_coarse_pattern_breakdown(features)

        min_windows = 5 if len(features) < 80 else 8
        best_rule_fine = pick_best_rule(pattern_df, min_windows=min_windows)
        best_rule_coarse = pick_best_rule(coarse_pattern_df, min_windows=min_windows)

    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    n_files = int(data["source_file"].nunique()) if "source_file" in data.columns else 0
    print("\n=== POLYMARKET BTC 5M TOKEN-TRADING ANALYSIS (RULES) ===")
    print(f"CSV files loaded: {n_files}")
    print(f"Rows after cleaning: {len(data):,}")
    print(f"Usable windows: {len(windows):,}")
    print(f"Feature windows: {len(features):,}")

    print("\nStep 1 - Optimal Sell Time")
    print(f"Target sell second: {target_sell_second}")
    print(_fmt_table(step1.sort_values("combined_avg_profit_c", ascending=False).head(10), max_rows=10))

    print("\nStep 2 - Optimal Buy Time")
    print(f"Target buy second: {target_buy_second}")
    print(_fmt_table(step2.sort_values("combined_avg_profit_c", ascending=False).head(10), max_rows=10))

    print("\nStep 3A - Coarse Feature Pattern Breakdown (More Tradable)")
    print(_fmt_table(coarse_pattern_df, max_rows=20))

    print("\nStep 3B - Fine Feature Pattern Breakdown (Detailed)")
    print(_fmt_table(pattern_df, max_rows=20))

    print("\nStep 4 - Actionable Trading Rule")
    if best_rule_coarse is None and best_rule_fine is None:
        print(
            f"No pattern met min sample size n>={min_windows}. "
            "Collect more data before trusting feature-conditioned entries."
        )
    else:
        selected = best_rule_coarse if best_rule_coarse is not None else best_rule_fine
        print(_rule_to_english(selected, target_buy_second, target_sell_second))

        if float(selected["best_profit_lb95_c"]) <= 0 or float(selected["best_win_lb95_pct"]) < 50.0:
            print(
                "Risk flag: confidence floor is weak (profit floor <= 0c or win-rate floor < 50%). "
                "Treat this as experimental, not production size."
            )

    print("\nTop 5 Candidate Rules (coarse, robust ranking)")
    top_cols = [
        "feature_pattern",
        "count_windows",
        "best_side",
        "best_win_rate_pct",
        "best_win_lb95_pct",
        "best_avg_profit_c",
        "best_profit_lb95_c",
        "robust_score",
    ]
    coarse_eligible = coarse_pattern_df[coarse_pattern_df["count_windows"] >= min_windows]
    if coarse_eligible.empty:
        print(f"(no coarse rules with n>={min_windows})")
    else:
        print(_fmt_table(coarse_eligible[top_cols], max_rows=5))

    return 0


if __name__ == "__main__":
    sys.exit(main())
