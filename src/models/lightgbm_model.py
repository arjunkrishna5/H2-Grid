# H2-Grid India: Tier 3 Tabular Gradient Boosting — LightGBM Forecasting Engine
# Fits an ensemble of gradient-boosted decision trees on 27 unscaled domain features.
# Evaluates holdout test performance (R2, RMSE, MAE) and extracts feature importance rankings.
# Enforces nocturnal zero-clipping and saves trained tree models and comparative metrics.

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.cross_val import HUBS, get_tabular_splits

# Suppress lightgbm scikit-learn API deprecation notices
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass
class LightGBMMetrics:
    """Evaluation metrics and feature attribution for the LightGBM model."""
    hub_id: str
    model_type: str
    r2_overall: float
    rmse_overall: float
    mae_overall: float
    r2_daytime: float
    rmse_daytime: float
    mae_daytime: float
    best_iteration: int
    train_time_sec: float
    eval_time_sec: float
    test_hours: int
    daytime_hours: int
    top_5_features: List[Tuple[str, float]]


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 1000,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    stopping_rounds: int = 40,
    random_state: int = 42,
) -> Tuple[lgb.LGBMRegressor, float]:
    """
    Trains a LightGBM regressor with early stopping on validation split.
    
    Returns:
        (fitted_model, train_time_seconds)
    """
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_time = time.time() - t0
    return model, train_time


def predict_lightgbm(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    solar_elevation: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, float]:
    """
    Generates out-of-sample test predictions with nocturnal physical zero-clipping.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)

    if solar_elevation is not None:
        night_mask = solar_elevation <= 0.0
        y_pred[night_mask] = 0.0

    eval_time = time.time() - t0
    return y_pred, eval_time


def evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
    daytime_threshold: float = 5.0,
) -> Dict[str, float]:
    """Computes overall and daytime-only regression metrics."""
    y_true_arr = y_true.to_numpy()
    rmse_overall = float(np.sqrt(mean_squared_error(y_true_arr, y_pred)))
    mae_overall = float(mean_absolute_error(y_true_arr, y_pred))
    r2_overall = float(r2_score(y_true_arr, y_pred))

    day_mask = y_true_arr > daytime_threshold
    if day_mask.sum() > 0:
        rmse_day = float(np.sqrt(mean_squared_error(y_true_arr[day_mask], y_pred[day_mask])))
        mae_day = float(mean_absolute_error(y_true_arr[day_mask], y_pred[day_mask]))
        r2_day = float(r2_score(y_true_arr[day_mask], y_pred[day_mask]))
    else:
        rmse_day, mae_day, r2_day = rmse_overall, mae_overall, r2_overall

    return {
        "rmse_overall": round(rmse_overall, 2),
        "mae_overall": round(mae_overall, 2),
        "r2_overall": round(r2_overall, 4),
        "rmse_daytime": round(rmse_day, 2),
        "mae_daytime": round(mae_day, 2),
        "r2_daytime": round(r2_day, 4),
        "test_hours": int(len(y_true)),
        "daytime_hours": int(day_mask.sum()),
    }


def get_feature_importances(
    model: lgb.LGBMRegressor,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Extracts normalized gain-based feature importance percentages."""
    gains = model.feature_importances_
    total_gain = float(np.sum(gains))
    if total_gain > 0:
        normalized = [(name, round(float(gain / total_gain * 100), 2)) for name, gain in zip(feature_names, gains)]
    else:
        normalized = [(name, 0.0) for name in feature_names]
    return sorted(normalized, key=lambda x: x[1], reverse=True)


def run_lightgbm_benchmark(
    hub_id: str,
    val_hours: int = 4000,
    save_results: bool = True,
) -> Tuple[LightGBMMetrics, np.ndarray]:
    """
    Runs the complete LightGBM benchmark pipeline for a given renewable hub.
    """
    hub_id = hub_id.lower().strip()
    X_tr, y_tr, X_te, y_te, df_tr, df_te = get_tabular_splits(hub_id, scaled=False)

    # Train / Validation chronological split
    X_train, y_train = X_tr.iloc[:-val_hours], y_tr.iloc[:-val_hours]
    X_val, y_val = X_tr.iloc[-val_hours:], y_tr.iloc[-val_hours:]

    # Fit LightGBM model
    model, train_time = train_lightgbm(X_train, y_train, X_val, y_val)

    # Predict on 2023 holdout test set (8,760 hours)
    elev_test = X_te["solar_elevation"] if "solar_elevation" in X_te.columns else None
    y_pred, eval_time = predict_lightgbm(model, X_te, solar_elevation=elev_test)

    # Evaluate regression performance
    scores = evaluate_regression(y_te, y_pred)

    # Calculate feature importances
    importances = get_feature_importances(model, list(X_tr.columns))
    top_5 = importances[:5]

    metrics = LightGBMMetrics(
        hub_id=hub_id,
        model_type="LightGBM Regressor (GBDT, Gain-based)",
        r2_overall=scores["r2_overall"],
        rmse_overall=scores["rmse_overall"],
        mae_overall=scores["mae_overall"],
        r2_daytime=scores["r2_daytime"],
        rmse_daytime=scores["rmse_daytime"],
        mae_daytime=scores["mae_daytime"],
        best_iteration=int(model.best_iteration_),
        train_time_sec=round(train_time, 2),
        eval_time_sec=round(eval_time, 2),
        test_hours=scores["test_hours"],
        daytime_hours=scores["daytime_hours"],
        top_5_features=top_5,
    )

    if save_results:
        # Save model joblib & native text
        model_joblib_path = PROCESSED_DIR / f"{hub_id}_lightgbm.joblib"
        joblib.dump(model, model_joblib_path)

        model_txt_path = PROCESSED_DIR / f"{hub_id}_lightgbm.txt"
        model.booster_.save_model(str(model_txt_path))

        # Save metrics
        metrics_file = PROCESSED_DIR / f"{hub_id}_lightgbm_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        # Save test predictions
        preds_df = pd.DataFrame(
            {"y_true": y_te, "y_pred_lightgbm": y_pred},
            index=df_te.index,
        )
        preds_file = PROCESSED_DIR / f"{hub_id}_lightgbm_preds.parquet"
        preds_df.to_parquet(preds_file)

    return metrics, y_pred


def benchmark_all_hubs() -> Dict[str, Dict]:
    """Runs the Tier 3 LightGBM benchmark across all 4 national renewable hubs."""
    print("\n" + "=" * 80)
    print("H2-Grid India: Tier 3 Tabular Gradient Boosting (LightGBM) Benchmark")
    print("=" * 80)

    all_metrics = {}
    for hub in HUBS:
        print(f"\n>>> Training & Benchmarking LightGBM on Hub: {hub.upper()}...")
        metrics, _ = run_lightgbm_benchmark(hub, save_results=True)
        all_metrics[hub] = asdict(metrics)
        print(f"    Trained {metrics.best_iteration} trees in {metrics.train_time_sec:.2f}s")
        print(f"    Overall:  R2 = {metrics.r2_overall:.4f} | RMSE = {metrics.rmse_overall:.2f} W/m2 | MAE = {metrics.mae_overall:.2f} W/m2")
        print(f"    Daytime:  R2 = {metrics.r2_daytime:.4f} | RMSE = {metrics.rmse_daytime:.2f} W/m2 | MAE = {metrics.mae_daytime:.2f} W/m2")
        top_features_str = ", ".join([f"{name} ({pct}%)" for name, pct in metrics.top_5_features[:3]])
        print(f"    Key Drivers: {top_features_str}")

    summary_path = PROCESSED_DIR / "lightgbm_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'Hub':<16} | {'Overall R2':<10} | {'RMSE (W/m2)':<11} | {'Daytime R2':<10} | {'Day RMSE':<10} | {'Train Time':<10}")
    print("-" * 80)
    for hub, m in all_metrics.items():
        print(
            f"{hub.capitalize():<16} | {m['r2_overall']:<10.4f} | {m['rmse_overall']:<11.2f} | "
            f"{m['r2_daytime']:<10.4f} | {m['rmse_daytime']:<10.2f} | {m['train_time_sec']:<8.2f}s"
        )
    print("=" * 80)
    print(f"[SUCCESS] Saved LightGBM benchmark summary -> {summary_path}")
    return all_metrics


if __name__ == "__main__":
    benchmark_all_hubs()

