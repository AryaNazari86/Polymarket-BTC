#!/usr/bin/env python3
"""
Phase 2 — Model training for BTC 5-minute direction prediction.
Expanding-window CV, LightGBM + XGBoost + CatBoost ensemble, Optuna tuning.
Python 3.9 compatible.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PARQUET_PATH = "features.parquet"
MODELS_DIR = "models"
FEATURE_COLS: List[str] = [
    "rsi_5m", "atr_5m", "bb_pos_5m", "ema8_5m",
    "ema8_1m", "ema55_1m", "ema_spread_1m", "ema_slope_1m",
    "atr_1m", "vwap_dev_1m",
    "taker_buy_ratio_30s", "taker_buy_ratio_2m", "taker_buy_ratio_5m",
    "rvol_5m", "rvol_15m", "rvol_1h",
    "st_dir_1m", "st_bars_since_flip_1m",
    "rsi_1h", "bb_pos_1h", "ema8_1h", "ema_divergence",
    "tod_sin", "tod_cos", "hour_sin", "hour_cos",
    "dist_round_500",
]

INITIAL_TRAIN_DAYS = 60
EXPAND_DAYS = 15
VAL_DAYS = 10
PURGE_HOURS = 3
CONFIDENCE_THRESHOLD = 0.65
N_OPTUNA_TRIALS = 30


def expanding_folds(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, val_idx) tuples using expanding window."""
    df = df.sort_values("open_time").reset_index(drop=True)
    ms_per_day = 86_400_000
    purge_ms = PURGE_HOURS * 3_600_000

    t_min = int(df["open_time"].iloc[0])
    t_max = int(df["open_time"].iloc[-1])

    folds = []
    train_end = t_min + INITIAL_TRAIN_DAYS * ms_per_day

    while train_end + VAL_DAYS * ms_per_day <= t_max:
        val_start = train_end + purge_ms
        val_end = val_start + VAL_DAYS * ms_per_day

        train_idx = df[df["open_time"] < train_end].index.to_numpy()
        val_idx = df[(df["open_time"] >= val_start) & (df["open_time"] < val_end)].index.to_numpy()

        if len(train_idx) >= 500 and len(val_idx) >= 50:
            folds.append((train_idx, val_idx))

        train_end += EXPAND_DAYS * ms_per_day

    return folds


def train_lgbm(X_tr: pd.DataFrame, y_tr: pd.Series,
               X_val: pd.DataFrame, y_val: pd.Series,
               params: dict) -> object:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params, n_estimators=500, verbose=-1, n_jobs=-1)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return model


def train_xgb(X_tr: pd.DataFrame, y_tr: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              params: dict) -> object:
    import xgboost as xgb
    model = xgb.XGBClassifier(**params, n_estimators=500, eval_metric="logloss",
                               early_stopping_rounds=50, verbosity=0, use_label_encoder=False)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_cat(X_tr: pd.DataFrame, y_tr: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              params: dict) -> object:
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(**params, iterations=500, eval_metric="Logloss",
                                early_stopping_rounds=50, verbose=0)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return model


def tune_lgbm(X: pd.DataFrame, y: pd.Series, folds: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
    import optuna
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        scores = []
        for tr_idx, val_idx in folds[-10:]:  # use last 10 folds for speed
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params, n_estimators=200, verbose=-1, n_jobs=-1)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, prob))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    return study.best_params


def tune_xgb(X: pd.DataFrame, y: pd.Series, folds: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        scores = []
        for tr_idx, val_idx in folds[-10:]:
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = xgb.XGBClassifier(**params, n_estimators=200, eval_metric="logloss",
                                       verbosity=0, use_label_encoder=False)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, prob))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    return study.best_params


def tune_cat(X: pd.DataFrame, y: pd.Series, folds: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        }
        scores = []
        for tr_idx, val_idx in folds[-10:]:
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = CatBoostClassifier(**params, iterations=200, verbose=0)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, prob))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    return study.best_params


def evaluate_oof(df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]],
                 lgbm_params: dict, xgb_params: dict, cat_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Run OOF predictions across all folds, return (oof_probs, oof_labels)."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    X = df[FEATURE_COLS].copy()
    y = df["target"].copy()

    oof_prob = np.full(len(df), np.nan)
    oof_label = y.to_numpy()

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_i+1}/{len(folds)} — train={len(tr_idx):,} val={len(val_idx):,}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        m_lgbm = lgb.LGBMClassifier(**lgbm_params, n_estimators=500, verbose=-1, n_jobs=-1)
        m_lgbm.fit(X_tr, y_tr)

        m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=500, eval_metric="logloss",
                                    verbosity=0, use_label_encoder=False)
        m_xgb.fit(X_tr, y_tr)

        m_cat = CatBoostClassifier(**cat_params, iterations=500, verbose=0)
        m_cat.fit(X_tr, y_tr)

        p = (m_lgbm.predict_proba(X_val)[:, 1] +
             m_xgb.predict_proba(X_val)[:, 1] +
             m_cat.predict_proba(X_val)[:, 1]) / 3

        oof_prob[val_idx] = p

    return oof_prob, oof_label


def main() -> None:
    from sklearn.metrics import roc_auc_score

    print("Loading features.parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values("open_time").reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(method="ffill").fillna(0)
    print(f"  {len(df):,} rows, {df['target'].mean():.1%} UP")

    print("Building expanding folds...")
    folds = expanding_folds(df)
    print(f"  {len(folds)} folds")

    X = df[FEATURE_COLS]
    y = df["target"]

    print("\nTuning LightGBM...")
    lgbm_params = tune_lgbm(X, y, folds)
    print(f"  Best: {lgbm_params}")

    print("\nTuning XGBoost...")
    xgb_params = tune_xgb(X, y, folds)
    print(f"  Best: {xgb_params}")

    print("\nTuning CatBoost...")
    cat_params = tune_cat(X, y, folds)
    print(f"  Best: {cat_params}")

    print("\nRunning OOF evaluation across all folds...")
    oof_prob, oof_label = evaluate_oof(df, folds, lgbm_params, xgb_params, cat_params)

    valid_mask = ~np.isnan(oof_prob)
    prob_v = oof_prob[valid_mask]
    label_v = oof_label[valid_mask]

    auc = roc_auc_score(label_v, prob_v)
    acc = ((prob_v > 0.5) == label_v).mean()

    confident_mask = (prob_v < (1 - CONFIDENCE_THRESHOLD)) | (prob_v > CONFIDENCE_THRESHOLD)
    n_confident = confident_mask.sum()
    acc_confident = ((prob_v[confident_mask] > 0.5) == label_v[confident_mask]).mean() if n_confident > 0 else 0.0
    fire_rate = n_confident / len(prob_v)

    print("\n" + "═" * 50)
    print("  TRAINING REPORT")
    print("═" * 50)
    print(f"  OOF AUC             : {auc:.4f}")
    print(f"  OOF Accuracy (all)  : {acc:.2%}")
    print(f"  Confident threshold : >{CONFIDENCE_THRESHOLD:.0%} or <{1-CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Confident trades    : {n_confident:,} / {len(prob_v):,} ({fire_rate:.1%} of windows)")
    print(f"  Confident accuracy  : {acc_confident:.2%}")
    print("═" * 50)

    print("\nTraining final models on ALL data...")
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    X_all = df[FEATURE_COLS].copy()
    y_all = df["target"].copy()

    final_lgbm = lgb.LGBMClassifier(**lgbm_params, n_estimators=500, verbose=-1, n_jobs=-1)
    final_lgbm.fit(X_all, y_all)

    final_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=500, eval_metric="logloss",
                                    verbosity=0, use_label_encoder=False)
    final_xgb.fit(X_all, y_all)

    final_cat = CatBoostClassifier(**cat_params, iterations=500, verbose=0)
    final_cat.fit(X_all, y_all)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/lgbm.pkl", "wb") as f:
        pickle.dump(final_lgbm, f)
    with open(f"{MODELS_DIR}/xgb.pkl", "wb") as f:
        pickle.dump(final_xgb, f)
    with open(f"{MODELS_DIR}/cat.pkl", "wb") as f:
        pickle.dump(final_cat, f)
    with open(f"{MODELS_DIR}/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    print(f"Models saved to {MODELS_DIR}/")
    print("Training complete.")


if __name__ == "__main__":
    main()
