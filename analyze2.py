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
]


def _sigmoid(value: float) -> float:
    clamped = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-clamped))


def _prepare_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    matrix = df[feature_cols].copy()
    matrix["crossed_below_45_before_buy"] = matrix["crossed_below_45_before_buy"].astype(int)
    matrix["crossed_above_55_before_buy"] = matrix["crossed_above_55_before_buy"].astype(int)
    return matrix.astype(float)


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    learning_rate: float = 0.08,
    epochs: int = 1500,
    l2_penalty: float = 0.001,
) -> Tuple[pd.Series, float]:
    weights = pd.Series(0.0, index=x_train.columns)
    bias = 0.0

    for _ in range(epochs):
        z = x_train.dot(weights) + bias
        probs = z.map(_sigmoid)
        errors = probs - y_train

        grad_w = x_train.mul(errors, axis=0).mean(axis=0) + l2_penalty * weights
        grad_b = float(errors.mean())

        weights = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b

    return weights, bias


def predict_probabilities(x: pd.DataFrame, weights: pd.Series, bias: float) -> pd.Series:
    z = x.dot(weights) + bias
    return z.map(_sigmoid)


def evaluate_classifier(
    probs: pd.Series,
    y_true: pd.Series,
    profits: pd.Series,
    threshold: float = 0.60,
) -> Dict[str, float]:
    pred = probs >= threshold

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    selected_count = int(pred.sum())
    if selected_count > 0:
        selected_win_rate = float(y_true[pred].mean()) * 100.0
        selected_avg_profit = float(profits[pred].mean())
        avg_pred_proba = float(probs[pred].mean())
    else:
        selected_win_rate = 0.0
        selected_avg_profit = 0.0
        avg_pred_proba = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "selected_count": float(selected_count),
        "selected_win_rate_pct": selected_win_rate,
        "selected_avg_profit_c": selected_avg_profit,
        "avg_predicted_prob_selected": avg_pred_proba,
        "baseline_win_rate_pct": float(y_true.mean() * 100.0),
        "baseline_avg_profit_c": float(profits.mean()),
    }


def _fmt_metrics(label: str, metrics: Dict[str, float]) -> str:
    return (
        f"{label}: acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, selected={int(metrics['selected_count'])}, "
        f"selected_win={metrics['selected_win_rate_pct']:.1f}%, "
        f"selected_avg_profit={metrics['selected_avg_profit_c']:+.2f}c, "
        f"baseline_win={metrics['baseline_win_rate_pct']:.1f}%, "
        f"baseline_avg_profit={metrics['baseline_avg_profit_c']:+.2f}c"
    )


def _top_coefficients(weights: pd.Series, top_n: int = 5) -> pd.DataFrame:
    out = pd.DataFrame({"feature": weights.index, "coefficient": weights.values})
    out["abs_coeff"] = out["coefficient"].abs()
    out = out.sort_values("abs_coeff", ascending=False).head(top_n)
    return out[["feature", "coefficient"]]


def _standardize(
    x_train_raw: pd.DataFrame, x_test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    means = x_train_raw.mean(axis=0)
    stds = x_train_raw.std(axis=0, ddof=0).replace(0.0, 1.0)
    x_train = (x_train_raw - means) / stds
    x_test = (x_test_raw - means) / stds
    return x_train, x_test, means, stds


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    odds_dir = project_dir / "odds_data"

    try:
        data = load_all_data(odds_dir)
        windows = build_windows(data)

        # Keep analyze2 aligned with analyze.py timings.
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

    if len(features) < 20:
        print(
            "Error: Need at least 20 windows with complete features for ML analysis. "
            f"Found {len(features)}."
        )
        return 1

    features = features.sort_values("window_ts").reset_index(drop=True)
    split_idx = int(len(features) * 0.7)
    split_idx = max(10, min(split_idx, len(features) - 5))

    train_df = features.iloc[:split_idx].copy()
    test_df = features.iloc[split_idx:].copy()

    x_train_raw = _prepare_matrix(train_df, FEATURE_COLUMNS)
    x_test_raw = _prepare_matrix(test_df, FEATURE_COLUMNS)
    x_train, x_test, _, _ = _standardize(x_train_raw, x_test_raw)

    y_train_up = train_df["up_win"].astype(int)
    y_test_up = test_df["up_win"].astype(int)
    y_train_down = train_df["down_win"].astype(int)
    y_test_down = test_df["down_win"].astype(int)

    up_weights, up_bias = train_logistic_regression(x_train, y_train_up)
    down_weights, down_bias = train_logistic_regression(x_train, y_train_down)

    up_probs = predict_probabilities(x_test, up_weights, up_bias)
    down_probs = predict_probabilities(x_test, down_weights, down_bias)

    up_metrics = evaluate_classifier(up_probs, y_test_up, test_df["up_profit_c"], threshold=0.60)
    down_metrics = evaluate_classifier(down_probs, y_test_down, test_df["down_profit_c"], threshold=0.60)

    decision_side = "Up"
    decision_metrics = up_metrics
    if down_metrics["selected_avg_profit_c"] > up_metrics["selected_avg_profit_c"]:
        decision_side = "Down"
        decision_metrics = down_metrics

    print("\n=== POLYMARKET BTC 5M TOKEN-TRADING ANALYSIS (ML) ===")
    print(f"Rows after cleaning: {len(data):,}")
    print(f"Usable windows: {len(windows):,}")
    print(f"Feature windows: {len(features):,}")
    print(f"Train windows: {len(train_df):,}")
    print(f"Test windows: {len(test_df):,}")
    print(f"Buy/Sell timing used: buy @{target_buy_second}s, sell @{target_sell_second}s")

    print("\nModel Performance (test split, threshold=0.60)")
    print(_fmt_metrics("Up model", up_metrics))
    print(_fmt_metrics("Down model", down_metrics))

    print("\nTop Coefficients (signal strength)")
    print("Up model:")
    print(_top_coefficients(up_weights).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nDown model:")
    print(_top_coefficients(down_weights).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\nActionable ML Filter")
    print(
        f"Based on out-of-sample selected-trade avg profit, prefer {decision_side} opportunities "
        f"when model probability >= 0.60. On test data: selected={int(decision_metrics['selected_count'])}, "
        f"win_rate={decision_metrics['selected_win_rate_pct']:.1f}%, "
        f"avg_profit={decision_metrics['selected_avg_profit_c']:+.2f}c."
    )

    features_out = project_dir / "odds_data" / "analysis_features_latest.csv"
    features.to_csv(features_out, index=False)
    print(f"\nSaved feature dataset for reuse: {features_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
