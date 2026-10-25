# Indian Southwest Monsoon & Seasonal Indicator Flag Engine for H2-Grid India
# Implements India Meteorological Department (IMD) seasonal classification and monsoon window flags
# Provides regime-switching awareness for extreme cloud attenuation across Indian renewable corridors
# Applied across South Indian benchmark hubs (Palakkad, Ramanathapuram, Pavagada) and Bhadla

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


def compute_monsoon_seasonal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific IMD seasonal regimes and Southwest Monsoon flags.
    
    Args:
        df: Input DataFrame with DatetimeIndex.
        
    Returns:
        pd.DataFrame: Enriched with is_monsoon, season_code, and transition flags.
    """
    df_out = df.copy()
    idx = df_out.index
    month = idx.month
    day = idx.day

    # 1. Official IMD Southwest Monsoon Window (June 1 to September 30)
    # Binary flag: 1 during active monsoon, 0 otherwise
    is_monsoon = (month >= 6) & (month <= 9)
    df_out["is_monsoon"] = is_monsoon.astype(int)

    # 2. Four IMD Climatological Seasons (Categorical Code 0 to 3)
    # 0 = Winter (Jan-Feb)
    # 1 = Pre-Monsoon / Summer (Mar-May)
    # 2 = Southwest Monsoon (Jun-Sep)
    # 3 = Post-Monsoon / Northeast Monsoon (Oct-Dec)
    conditions = [
        (month.isin([1, 2])),
        (month.isin([3, 4, 5])),
        (month.isin([6, 7, 8, 9])),
        (month.isin([10, 11, 12])),
    ]
    choices = [0, 1, 2, 3]
    df_out["season_code"] = np.select(conditions, choices, default=0)

    # 3. One-Hot Encoded Seasonal Binary Columns (For Linear Models & Trees)
    df_out["season_winter"] = (df_out["season_code"] == 0).astype(int)
    df_out["season_summer"] = (df_out["season_code"] == 1).astype(int)
    df_out["season_monsoon"] = (df_out["season_code"] == 2).astype(int)
    df_out["season_post_monsoon"] = (df_out["season_code"] == 3).astype(int)

    # 4. Monsoon Transition Boundary Window (Onset: May 20 - June 10, Retreat: Sep 20 - Oct 10)
    # Captures maximum cloud turbulence and seasonal regime shifts
    onset_transition = ((month == 5) & (day >= 20)) | ((month == 6) & (day <= 10))
    retreat_transition = ((month == 9) & (day >= 20)) | ((month == 10) & (day <= 10))
    df_out["is_monsoon_transition"] = (onset_transition | retreat_transition).astype(int)

    return df_out


def process_hub_monsoon(
    hub_id: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load lag-enriched dataset, compute monsoon & seasonal indicator flags, and persist.
    """
    if hub_id not in BENCHMARK_HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

    hub_info = BENCHMARK_HUBS[hub_id]
    logger.info(f"Computing monsoon and seasonal flags for {hub_info['name']}...")

    # Load from lag output if available, else load base parquet
    lags_pq = os.path.join(processed_dir, f"{hub_id}_lags.parquet")
    if os.path.exists(lags_pq):
        df = pd.read_parquet(lags_pq, engine="pyarrow")
    else:
        from src.features.lags import process_hub_lags
        df, _ = process_hub_lags(hub_id, processed_dir=processed_dir)

    # Compute seasonal and monsoon flags
    df_enriched = compute_monsoon_seasonal_flags(df)

    # Save outputs (Parquet & CSV)
    out_parquet = os.path.join(processed_dir, f"{hub_id}_monsoon.parquet")
    out_csv = os.path.join(processed_dir, f"{hub_id}_monsoon.csv")

    df_enriched.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=True)
    df_enriched.to_csv(out_csv, index=True)

    summary = {
        "hub_id": hub_id,
        "name": hub_info["name"],
        "rows": len(df_enriched),
        "columns": len(df_enriched.columns),
        "monsoon_hours": int(df_enriched["is_monsoon"].sum()),
        "transition_hours": int(df_enriched["is_monsoon_transition"].sum()),
    }
    return df_enriched, summary


def process_all_hubs_monsoon(processed_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Process monsoon flag engine across all benchmark hubs and print summary table."""
    results = {}

    print("\n" + "=" * 105)
    print("                 H2-GRID INDIA: INDIAN MONSOON & SEASONAL FLAGS (SUBPHASE 2.4)")
    print("=" * 105)
    print(f"{'Hub Name':<28} | {'Rows':<8} | {'Columns':<8} | {'Monsoon Hours':<16} | {'Transition Hours':<18} | {'Status':<8}")
    print("-" * 105)

    for hub_id in BENCHMARK_HUBS:
        _, summary = process_hub_monsoon(hub_id, processed_dir=processed_dir)
        results[hub_id] = summary
        print(
            f"{summary['name']:<28} | "
            f"{summary['rows']:<8} | "
            f"{summary['columns']:<8} | "
            f"{summary['monsoon_hours']:<16} | "
            f"{summary['transition_hours']:<18} | "
            f"{'COMPLETED':<8}"
        )

    print("=" * 105)
    print("Flags generated: is_monsoon, season_code, season_winter/summer/monsoon/post, is_monsoon_transition.")
    print("=" * 105 + "\n")

    return results


if __name__ == "__main__":
    process_all_hubs_monsoon()
