#!/usr/bin/env python3
"""ML layer for Polymarket BTC 5-minute token-trading signals (pandas + stdlib only)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

from analysis_common import (
    build_feature_dataset,
    build_windows,
    evaluate_buy_seconds,
    evaluate_sell_seconds,
    load_all_data,
)

FEATURE_COLUMNS = [
    "up_0",
    "up_10",
    "up_20",
    "up_30",
    "up_slope_0_30",
    "btc_slope_0_30",
    "crossed_below_45_before_buy",
    "crossed_above_55_before_buy",
    "spread_30",
    "up_volatility_0_buy",
    "btc_volatility_0_buy",
    "up_range_0_buy",
    "btc_change_0_buy",
    "up_change_10_30",
]


def _sigmoid(value: float) -> float:
    clamped = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-clamped))


def _prepare_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    matrix = df[feature_cols].copy()
    bool_cols = ["crossed_below_45_before_buy", "crossed_above_55_before_buy"]
    for col in bool_cols:
        matrix[col] = matrix[col].astype(int)
    return matrix.astype(float)


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    learning_rate: float = 0.05,
    epochs: int = 2200,
    l2_penalty: float = 0.01,
) -> Tuple[pd.Series, float]:
    weights = pd.Series(0.0, index=x_train.columns)
    bias = 0.0

    pos_frac = float(y_train.mean())
    neg_frac = 1.0 - pos_frac
    pos_w = 0.5 / max(pos_frac, 1e-6)
    neg_w = 0.5 / max(neg_frac, 1e-6)

    sample_w = y_train.map(lambda y: pos_w if y == 1 else neg_w)

    for _ in range(epochs):
        z = x_train.dot(weights) + bias
        probs = z.map(_sigmoid)
        errors = (probs - y_train) * sample_w

        grad_w = x_train.mul(errors, axis=0).mean(axis=0) + l2_penalty * weights
        grad_b = float(errors.mean())

        weights = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b

    return weights, bias


def predict_probabilities(x: pd.DataFrame, weights: pd.Series, bias: float) -> pd.Series:
    z = x.dot(weights) + bias
    return z.map(_sigmoid)


def _metrics_for_threshold(
    probs: pd.Series,
    y_true: pd.Series,
    profits: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    pred = probs >= threshold
    selected_count = int(pred.sum())

    if selected_count > 0:
        selected_win_rate = float(y_true[pred].mean()) * 100.0
        selected_avg_profit = float(profits[pred].mean())
        total_profit = float(profits[pred].sum())
        avg_pred = float(probs[pred].mean())
    else:
        selected_win_rate = 0.0
        selected_avg_profit = 0.0
        total_profit = 0.0
        avg_pred = 0.0

    return {
        "threshold": threshold,
        "selected_count": float(selected_count),
        "selected_win_rate_pct": selected_win_rate,
        "selected_avg_profit_c": selected_avg_profit,
        "selected_total_profit_c": total_profit,
        "avg_predicted_prob_selected": avg_pred,
        "baseline_win_rate_pct": float(y_true.mean() * 100.0),
        "baseline_avg_profit_c": float(profits.mean()),
    }


def tune_threshold(
    probs: pd.Series,
    y_true: pd.Series,
    profits: pd.Series,
    min_selected: int,
) -> Dict[str, float]:
    best = None
    for step in range(50, 81):
        thr = step / 100.0
        m = _metrics_for_threshold(probs, y_true, profits, threshold=thr)
        if int(m["selected_count"]) < min_selected:
            continue
        if best is None:
            best = m
            continue

        # Prefer higher total expected cents; tie-break with avg profit then win rate.
        if (
            m["selected_total_profit_c"] > best["selected_total_profit_c"]
            or (
                m["selected_total_profit_c"] == best["selected_total_profit_c"]
                and m["selected_avg_profit_c"] > best["selected_avg_profit_c"]
            )
            or (
                m["selected_total_profit_c"] == best["selected_total_profit_c"]
                and m["selected_avg_profit_c"] == best["selected_avg_profit_c"]
                and m["selected_win_rate_pct"] > best["selected_win_rate_pct"]
            )
        ):
            best = m

    if best is None:
        # Fallback to 0.60 if no threshold passes min_selected.
        best = _metrics_for_threshold(probs, y_true, profits, threshold=0.60)
    return best


def _fmt_metrics(label: str, metrics: Dict[str, float]) -> str:
    return (
        f"{label}: thr={metrics['threshold']:.2f}, selected={int(metrics['selected_count'])}, "
        f"win={metrics['selected_win_rate_pct']:.1f}%, avg={metrics['selected_avg_profit_c']:+.2f}c, "
        f"total={metrics['selected_total_profit_c']:+.2f}c, baseline_avg={metrics['baseline_avg_profit_c']:+.2f}c"
    )


def _top_coefficients(weights: pd.Series, top_n: int = 6) -> pd.DataFrame:
    out = pd.DataFrame({"feature": weights.index, "coefficient": weights.values})
    out["abs_coeff"] = out["coefficient"].abs()
    out = out.sort_values("abs_coeff", ascending=False).head(top_n)
    return out[["feature", "coefficient"]]


def _standardize(
    x_train_raw: pd.DataFrame, x_val_raw: pd.DataFrame, x_test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = x_train_raw.mean(axis=0)
    stds = x_train_raw.std(axis=0, ddof=0).replace(0.0, 1.0)
    x_train = (x_train_raw - means) / stds
    x_val = (x_val_raw - means) / stds
    x_test = (x_test_raw - means) / stds
    return x_train, x_val, x_test


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    odds_dir = project_dir / "odds_data"

    try:
        data = load_all_data(odds_dir)
        windows = build_windows(data)

        step1 = evaluate_sell_seconds(windows, buy_second=60)
        target_sell_second = int(step1.loc[step1["combined_avg_profit_c"].idxmax(), "sell_second"])

        step2 = evaluate_buy_seconds(windows, sell_second=target_sell_second)
        target_buy_second = int(step2.loc[step2["combined_avg_profit_c"].idxmax(), "buy_second"])

        features = build_feature_dataset(
            windows,
            target_buy_second=target_buy_second,
            target_sell_second=target_sell_second,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if len(features) < 40:
        print(
            "Error: Need at least 40 windows with complete features for robust ML analysis. "
            f"Found {len(features)}."
        )
        return 1

    features = features.sort_values("window_ts").reset_index(drop=True)
    n = len(features)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train_end = max(20, min(train_end, n - 15))
    val_end = max(train_end + 10, min(val_end, n - 5))

    train_df = features.iloc[:train_end].copy()
    val_df = features.iloc[train_end:val_end].copy()
    test_df = features.iloc[val_end:].copy()

    x_train_raw = _prepare_matrix(train_df, FEATURE_COLUMNS)
    x_val_raw = _prepare_matrix(val_df, FEATURE_COLUMNS)
    x_test_raw = _prepare_matrix(test_df, FEATURE_COLUMNS)
    x_train, x_val, x_test = _standardize(x_train_raw, x_val_raw, x_test_raw)

    y_train_up = train_df["up_win"].astype(int)
    y_val_up = val_df["up_win"].astype(int)
    y_test_up = test_df["up_win"].astype(int)

    y_train_down = train_df["down_win"].astype(int)
    y_val_down = val_df["down_win"].astype(int)
    y_test_down = test_df["down_win"].astype(int)

    up_weights, up_bias = train_logistic_regression(x_train, y_train_up)
    down_weights, down_bias = train_logistic_regression(x_train, y_train_down)

    up_probs_val = predict_probabilities(x_val, up_weights, up_bias)
    down_probs_val = predict_probabilities(x_val, down_weights, down_bias)

    min_selected_val = max(5, int(len(val_df) * 0.20))
    up_tuned = tune_threshold(up_probs_val, y_val_up, val_df["up_profit_c"], min_selected=min_selected_val)
    down_tuned = tune_threshold(
        down_probs_val,
        y_val_down,
        val_df["down_profit_c"],
        min_selected=min_selected_val,
    )

    up_probs_test = predict_probabilities(x_test, up_weights, up_bias)
    down_probs_test = predict_probabilities(x_test, down_weights, down_bias)

    up_test = _metrics_for_threshold(
        up_probs_test,
        y_test_up,
        test_df["up_profit_c"],
        threshold=float(up_tuned["threshold"]),
    )
    down_test = _metrics_for_threshold(
        down_probs_test,
        y_test_down,
        test_df["down_profit_c"],
        threshold=float(down_tuned["threshold"]),
    )

    print("\n=== POLYMARKET BTC 5M TOKEN-TRADING ANALYSIS (ML) ===")
    print(f"Rows after cleaning: {len(data):,}")
    print(f"Usable windows: {len(windows):,}")
    print(f"Feature windows: {len(features):,}")
    print(f"Train/Val/Test windows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"Buy/Sell timing used: buy @{target_buy_second}s, sell @{target_sell_second}s")

    print("\nValidation-Tuned Thresholds")
    print(_fmt_metrics("Up (validation)", up_tuned))
    print(_fmt_metrics("Down (validation)", down_tuned))

    print("\nOut-of-Sample Test Performance (using tuned thresholds)")
    print(_fmt_metrics("Up (test)", up_test))
    print(_fmt_metrics("Down (test)", down_test))

    print("\nTop Coefficients (signal strength)")
    print("Up model:")
    print(_top_coefficients(up_weights).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nDown model:")
    print(_top_coefficients(down_weights).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\nActionable ML Decision")
    candidates = [("Up", up_test), ("Down", down_test)]
    candidates.sort(key=lambda x: (x[1]["selected_avg_profit_c"], x[1]["selected_total_profit_c"]), reverse=True)
    best_side, best = candidates[0]

    if best["selected_count"] < 3:
        print("No-trade: model selected too few opportunities on test set.")
    elif best["selected_avg_profit_c"] <= 0:
        print(
            "No-trade: best model filter is still negative expectancy out-of-sample. "
            f"Best was {best_side} with avg {best['selected_avg_profit_c']:+.2f}c."
        )
    elif best["selected_avg_profit_c"] <= best["baseline_avg_profit_c"]:
        print(
            "No-trade: filtered trades do not beat baseline average expectancy on test set. "
            f"Best {best_side} filter avg={best['selected_avg_profit_c']:+.2f}c vs baseline={best['baseline_avg_profit_c']:+.2f}c."
        )
    else:
        print(
            f"Trade filter is usable: prefer {best_side} when model probability >= {best['threshold']:.2f}. "
            f"Test selected={int(best['selected_count'])}, win={best['selected_win_rate_pct']:.1f}%, "
            f"avg={best['selected_avg_profit_c']:+.2f}c, total={best['selected_total_profit_c']:+.2f}c."
        )

    features_out = project_dir / "odds_data" / "analysis_features_latest.csv"
    features.to_csv(features_out, index=False)
    print(f"\nSaved feature dataset for reuse: {features_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
