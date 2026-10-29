# H2-Grid India: Tier 1 Statistical Baseline — SARIMA Forecasting Engine
# Fits Seasonal ARIMA (p, d, q) x (P, D, Q)_24 to capture diurnal solar cycles.
# Evaluates baseline performance (R2, RMSE, MAE) across 2023 holdout test sets.
# Enforces nocturnal zero-clipping and saves baseline metrics for multi-tier benchmarking.

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from src.models.cross_val import HUBS, get_tabular_splits

# Suppress statsmodels convergence / frequency warnings during batch evaluation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass
class SarimaMetrics:
    """Evaluation metrics for the SARIMA baseline model."""
    hub_id: str
    order: str
    seasonal_order: str
    r2_overall: float
    rmse_overall: float
    mae_overall: float
    r2_daytime: float
    rmse_daytime: float
    mae_daytime: float
    aic: float
    fit_time_sec: float
    eval_time_sec: float
    test_hours: int
    daytime_hours: int


def fit_sarima(
    y_train: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 24),
    maxiter: int = 25,
) -> Tuple[SARIMAXResultsWrapper, float]:
    """
    Fits a Seasonal ARIMA model on training time-series series.
    
    Args:
        y_train: Training target series (unscaled GHI in W/m^2).
        order: Non-seasonal (p, d, q) order (default: (1, 0, 0)).
        seasonal_order: Seasonal (P, D, Q, s) order with s=24 daily cycle (default: (1, 1, 0, 24)).
        maxiter: Maximum iterations for Kalman maximum likelihood convergence.
    Returns:
        (fitted_model_result, fit_time_seconds)
    """
    t0 = time.time()
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=maxiter)
    fit_time = time.time() - t0
    return res, fit_time


def predict_sarima(
    model_res: SARIMAXResultsWrapper,
    y_test: pd.Series,
    solar_elevation: Optional[pd.Series] = None,
) -> Tuple[pd.Series, float]:
    """
    Applies the fitted SARIMA filter to out-of-sample test series (1-step rolling forward).
    Enforces nocturnal zero-clipping when the sun is below the horizon.
    
    Args:
        model_res: Fitted SARIMAX results object.
        y_test: Unseen test series (2023 holdout, 8,760 hours).
        solar_elevation: Optional solar elevation series to zero out night drift.
    Returns:
        (y_pred_series, eval_time_seconds)
    """
    t0 = time.time()
    res_test = model_res.apply(y_test)
    y_pred = res_test.fittedvalues.copy()
    
    # 1. Non-negative physical clipping
    y_pred = np.clip(y_pred, 0.0, None)
    
    # 2. Strict nocturnal zero-clipping if solar elevation is provided
    if solar_elevation is not None:
        night_mask = solar_elevation <= 0.0
        y_pred[night_mask] = 0.0
        
    eval_time = time.time() - t0
    y_pred_series = pd.Series(y_pred, index=y_test.index, name="ghi_pred_sarima")
    return y_pred_series, eval_time


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    daytime_threshold: float = 5.0,
) -> Dict[str, float]:
    """
    Computes standard regression metrics across overall time and daytime-only hours.
    
    Args:
        y_true: Actual GHI values (W/m^2).
        y_pred: Predicted GHI values (W/m^2).
        daytime_threshold: Threshold above which hours are treated as active daytime.
    Returns:
        Dictionary of computed metrics.
    """
    rmse_overall = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_overall = float(mean_absolute_error(y_true, y_pred))
    r2_overall = float(r2_score(y_true, y_pred))
    
    # Daytime-only subset (removes nighttime 0s to measure true active generation accuracy)
    day_mask = y_true > daytime_threshold
    if day_mask.sum() > 0:
        rmse_day = float(np.sqrt(mean_squared_error(y_true[day_mask], y_pred[day_mask])))
        mae_day = float(mean_absolute_error(y_true[day_mask], y_pred[day_mask]))
        r2_day = float(r2_score(y_true[day_mask], y_pred[day_mask]))
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


def run_sarima_benchmark(
    hub_id: str,
    training_lookback_hours: int = 8760,  # 1 full year baseline (2022) for fast, robust convergence
    save_results: bool = True,
) -> Tuple[SarimaMetrics, pd.Series]:
    """
    Executes the complete SARIMA baseline benchmark for a given renewable hub.
    
    Args:
        hub_id: Hub identifier (e.g. 'bhadla').
        training_lookback_hours: Number of recent training hours to fit SARIMA parameters.
        save_results: If True, writes predictions and metrics to data/processed/.
    Returns:
        (metrics_dataclass, test_predictions_series)
    """
    hub_id = hub_id.lower().strip()
    X_tr, y_tr, X_te, y_te, df_tr, df_te = get_tabular_splits(hub_id, scaled=False)
    
    # Use recent training window for SARIMA state-space efficiency
    y_train_subset = y_tr.iloc[-training_lookback_hours:] if len(y_tr) > training_lookback_hours else y_tr
    
    order = (1, 0, 0)
    seasonal_order = (1, 1, 0, 24)
    
    # 1. Fit SARIMA
    model_res, fit_time = fit_sarima(y_train_subset, order=order, seasonal_order=seasonal_order)
    
    # 2. Predict 2023 holdout test set with solar elevation clipping
    elev_test = X_te["solar_elevation"] if "solar_elevation" in X_te.columns else None
    y_pred, eval_time = predict_sarima(model_res, y_te, solar_elevation=elev_test)
    
    # 3. Evaluate metrics
    scores = evaluate_predictions(y_te, y_pred)
    
    metrics = SarimaMetrics(
        hub_id=hub_id,
        order=str(order),
        seasonal_order=str(seasonal_order),
        r2_overall=scores["r2_overall"],
        rmse_overall=scores["rmse_overall"],
        mae_overall=scores["mae_overall"],
        r2_daytime=scores["r2_daytime"],
        rmse_daytime=scores["rmse_daytime"],
        mae_daytime=scores["mae_daytime"],
        aic=round(float(model_res.aic), 1),
        fit_time_sec=round(fit_time, 2),
        eval_time_sec=round(eval_time, 2),
        test_hours=scores["test_hours"],
        daytime_hours=scores["daytime_hours"],
    )
    
    # 4. Save results
    if save_results:
        metrics_file = PROCESSED_DIR / f"{hub_id}_sarima_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)
            
        preds_df = pd.DataFrame({"y_true": y_te, "y_pred_sarima": y_pred})
        preds_file = PROCESSED_DIR / f"{hub_id}_sarima_preds.parquet"
        preds_df.to_parquet(preds_file)
        
    return metrics, y_pred


def benchmark_all_hubs() -> Dict[str, Dict]:
    """
    Runs the Tier 1 SARIMA baseline benchmark across all 4 national hubs
    and writes a unified comparative summary JSON.
    """
    print("\n" + "=" * 80)
    print("H2-Grid India: Tier 1 Statistical Baseline (SARIMA) Benchmark")
    print("=" * 80)
    
    all_metrics = {}
    for hub in HUBS:
        print(f"\n>>> Running SARIMA Baseline on Hub: {hub.upper()}...")
        metrics, _ = run_sarima_benchmark(hub, save_results=True)
        all_metrics[hub] = asdict(metrics)
        print(f"    Fit Time: {metrics.fit_time_sec:.2f}s | Eval Time: {metrics.eval_time_sec:.2f}s (8,760 hrs)")
        print(f"    Overall:  R2 = {metrics.r2_overall:.4f} | RMSE = {metrics.rmse_overall:.2f} W/m2 | MAE = {metrics.mae_overall:.2f} W/m2")
        print(f"    Daytime:  R2 = {metrics.r2_daytime:.4f} | RMSE = {metrics.rmse_daytime:.2f} W/m2 | MAE = {metrics.mae_daytime:.2f} W/m2")
        
    summary_path = PROCESSED_DIR / "sarima_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        
    # Print formatted comparative leaderboard
    print("\n" + "=" * 80)
    print(f"{'Hub':<16} | {'Overall R2':<10} | {'RMSE (W/m2)':<11} | {'Daytime R2':<10} | {'Day RMSE':<10} | {'Fit Time':<8}")
    print("-" * 80)
    for hub, m in all_metrics.items():
        print(
            f"{hub.capitalize():<16} | {m['r2_overall']:<10.4f} | {m['rmse_overall']:<11.2f} | "
            f"{m['r2_daytime']:<10.4f} | {m['rmse_daytime']:<10.2f} | {m['fit_time_sec']:<6.2f}s"
        )
    print("=" * 80)
    print(f"[SUCCESS] Saved SARIMA baseline benchmark summary -> {summary_path}")
    return all_metrics


if __name__ == "__main__":
    benchmark_all_hubs()

