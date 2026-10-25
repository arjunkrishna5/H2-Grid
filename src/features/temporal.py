# Continuous Cyclical Temporal Encoding Engine for H2-Grid India
# Projects 24-hour diurnal and 365-day seasonal time onto continuous orthogonal sine/cosine waves
# Eliminates artificial midnight and year-end boundary cliffs for neural network training
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


def compute_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute smooth cyclical orthogonal sine/cosine transformations for diurnal and annual cycles.
    
    Args:
        df: Input DataFrame with DatetimeIndex.
        
    Returns:
        pd.DataFrame: Enriched with hour_sin, hour_cos, day_sin, day_cos columns.
    """
    df_out = df.copy()
    idx = df_out.index

    # 1. 24-Hour Diurnal Cycle (Hour of day: 0 to 23, with fractional minute precision)
    hours = idx.hour + (idx.minute / 60.0)
    df_out["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df_out["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)

    # 2. 365.25-Day Annual Seasonal Cycle (Day of year: 1 to 365/366)
    # Using 365.25 to smoothly account for leap year orbital periodicity
    day_of_year = idx.dayofyear - 1
    df_out["day_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df_out["day_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return df_out


def process_hub_temporal(
    hub_id: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load astronomy-enriched dataset, compute cyclical temporal features, and persist.
    """
    if hub_id not in BENCHMARK_HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

    hub_info = BENCHMARK_HUBS[hub_id]
    logger.info(f"Computing cyclical temporal encodings for {hub_info['name']}...")

    # Load from astronomy output if available, else load base parquet
    astronomy_pq = os.path.join(processed_dir, f"{hub_id}_astronomy.parquet")
    if os.path.exists(astronomy_pq):
        df = pd.read_parquet(astronomy_pq, engine="pyarrow")
    else:
        from src.data.cache_engine import load_hub_dataset
        df = load_hub_dataset(hub_id, file_format="parquet")

    # Compute cyclical sine and cosine features
    df_enriched = compute_cyclical_temporal_features(df)

    # Save outputs (Parquet & CSV)
    out_parquet = os.path.join(processed_dir, f"{hub_id}_temporal.parquet")
    out_csv = os.path.join(processed_dir, f"{hub_id}_temporal.csv")

    df_enriched.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=True)
    df_enriched.to_csv(out_csv, index=True)

    summary = {
        "hub_id": hub_id,
        "name": hub_info["name"],
        "rows": len(df_enriched),
        "columns": len(df_enriched.columns),
        "hour_sin_range": (float(df_enriched["hour_sin"].min()), float(df_enriched["hour_sin"].max())),
        "day_sin_range": (float(df_enriched["day_sin"].min()), float(df_enriched["day_sin"].max())),
    }
    return df_enriched, summary


def process_all_hubs_temporal(processed_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Process temporal feature engine across all benchmark hubs and print summary table."""
    results = {}

    print("\n" + "=" * 105)
    print("                 H2-GRID INDIA: CYCLICAL TEMPORAL ENCODINGS (SUBPHASE 2.2)")
    print("=" * 105)
    print(f"{'Hub Name':<28} | {'Rows':<8} | {'Columns':<8} | {'Hour (Sin/Cos)':<18} | {'Day (Sin/Cos)':<18} | {'Status':<8}")
    print("-" * 105)

    for hub_id in BENCHMARK_HUBS:
        _, summary = process_hub_temporal(hub_id, processed_dir=processed_dir)
        results[hub_id] = summary
        print(
            f"{summary['name']:<28} | "
            f"{summary['rows']:<8} | "
            f"{summary['columns']:<8} | "
            f"[-1.0, 1.0]         | "
            f"[-1.0, 1.0]         | "
            f"{'COMPLETED':<8}"
        )

    print("=" * 105)
    print("Features generated: hour_sin, hour_cos (24h diurnal) & day_sin, day_cos (365.25d annual).")
    print("=" * 105 + "\n")

    return results


if __name__ == "__main__":
    process_all_hubs_temporal()
