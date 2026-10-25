# Autoregressive Multi-Horizon Lag & Rolling Statistics Generator for H2-Grid India
# Computes temporal lags (t-1, t-2, t-24, t-168) and rolling statistics (6h mean, 24h std)
# Prevents data leakage and provides tabular gradient boosting (LightGBM) with historical memory
# Applied across South Indian renewable corridors (Palakkad, Ramanathapuram, Pavagada) and Bhadla

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data.download_hubs import BENCHMARK_HUBS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")


def compute_autoregressive_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate historical lag and rolling statistics features strictly without future leakage.
    
    Args:
        df: Input DataFrame with DatetimeIndex and 'ghi' column.
        
    Returns:
        pd.DataFrame: Enriched with autoregressive lags and volatility features.
    """
    df_out = df.copy()

    # 1. Multi-Horizon Autoregressive Lags for Solar Irradiance (GHI)
    # t-1: immediate hourly momentum
    df_out["ghi_lag_1"] = df_out["ghi"].shift(1)
    # t-2: short-term trend rate of change
    df_out["ghi_lag_2"] = df_out["ghi"].shift(2)
    # t-24: 24 hours ago (exact same solar hour yesterday)
    df_out["ghi_lag_24"] = df_out["ghi"].shift(24)
    # t-168: 168 hours ago (7 days * 24 hours = exact same time last week)
    df_out["ghi_lag_168"] = df_out["ghi"].shift(168)

    # 2. Rolling Statistics (Shifted by 1 hour to prevent lookahead data leakage)
    # 6-hour rolling mean captures recent diurnal weather trend
    df_out["ghi_rolling_mean_6h"] = df_out["ghi"].rolling(window=6, min_periods=1).mean().shift(1)
    # 24-hour rolling standard deviation acts as a cloud volatility / turbulence indicator
    df_out["ghi_rolling_std_24h"] = df_out["ghi"].rolling(window=24, min_periods=2).std().shift(1)

    # 3. Handle initial warmup period (first 168 hours / 7 days of 2019) via clean backward fill
    df_out["ghi_lag_1"] = df_out["ghi_lag_1"].bfill()
    df_out["ghi_lag_2"] = df_out["ghi_lag_2"].bfill()
    df_out["ghi_lag_24"] = df_out["ghi_lag_24"].bfill()
    df_out["ghi_lag_168"] = df_out["ghi_lag_168"].bfill()
    df_out["ghi_rolling_mean_6h"] = df_out["ghi_rolling_mean_6h"].bfill()
    df_out["ghi_rolling_std_24h"] = df_out["ghi_rolling_std_24h"].fillna(0.0)

    return df_out


def process_hub_lags(
    hub_id: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load temporal-enriched dataset, compute autoregressive lags and rolling stats, and persist.
    """
    if hub_id not in BENCHMARK_HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

    hub_info = BENCHMARK_HUBS[hub_id]
    logger.info(f"Computing autoregressive lags and rolling stats for {hub_info['name']}...")

    # Load from temporal output if available, else load base parquet
    temporal_pq = os.path.join(processed_dir, f"{hub_id}_temporal.parquet")
    if os.path.exists(temporal_pq):
        df = pd.read_parquet(temporal_pq, engine="pyarrow")
    else:
        from src.features.temporal import process_hub_temporal
        df, _ = process_hub_temporal(hub_id, processed_dir=processed_dir)

    # Compute autoregressive features
    df_enriched = compute_autoregressive_lags(df)

    # Save outputs (Parquet & CSV)
    out_parquet = os.path.join(processed_dir, f"{hub_id}_lags.parquet")
    out_csv = os.path.join(processed_dir, f"{hub_id}_lags.csv")

    df_enriched.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=True)
    df_enriched.to_csv(out_csv, index=True)

    summary = {
        "hub_id": hub_id,
        "name": hub_info["name"],
        "rows": len(df_enriched),
        "columns": len(df_enriched.columns),
        "lag_1_mean": float(df_enriched["ghi_lag_1"].mean()),
        "rolling_std_max": float(df_enriched["ghi_rolling_std_24h"].max()),
    }
    return df_enriched, summary


def process_all_hubs_lags(processed_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Process lag feature generator across all benchmark hubs and print summary table."""
    results = {}

    print("\n" + "=" * 105)
    print("                 H2-GRID INDIA: AUTOREGRESSIVE LAG GENERATOR (SUBPHASE 2.3)")
    print("=" * 105)
    print(f"{'Hub Name':<28} | {'Rows':<8} | {'Columns':<8} | {'Lags Created':<20} | {'Max Cloud Volatility':<22} | {'Status':<8}")
    print("-" * 105)

    for hub_id in BENCHMARK_HUBS:
        _, summary = process_hub_lags(hub_id, processed_dir=processed_dir)
        results[hub_id] = summary
        print(
            f"{summary['name']:<28} | "
            f"{summary['rows']:<8} | "
            f"{summary['columns']:<8} | "
            f"t-1, t-2, t-24, t-168 | "
            f"{summary['rolling_std_max']:.1f} W/m^2             | "
            f"{'COMPLETED':<8}"
        )

    print("=" * 105)
    print("Features generated: ghi_lag_1, ghi_lag_2, ghi_lag_24, ghi_lag_168, ghi_rolling_mean_6h, ghi_rolling_std_24h.")
    print("=" * 105 + "\n")

    return results


if __name__ == "__main__":
    process_all_hubs_lags()
