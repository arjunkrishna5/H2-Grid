# Raw Data Validation, Missing Value Handling & Quality Audit Engine for H2-Grid India
# Validates timestamp continuity, enforces physical atmospheric bounds, and repairs sensor dropouts
# Produces certified hourly datasets and comprehensive data quality audit summaries
# Covers South Indian renewable corridors (Palakkad, Ramanathapuram, Pavagada) and Bhadla

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data.download_hubs import BENCHMARK_HUBS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Atmospheric physical boundaries per solar physics and meteorological standards
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ghi": (0.0, 1367.0),          # Solar constant upper ceiling (W/m^2)
    "dni": (0.0, 1367.0),          # Direct normal irradiance upper ceiling (W/m^2)
    "temperature": (-10.0, 60.0),   # Plausible Indian ground ambient temperatures (deg C)
    "humidity": (0.0, 100.0),       # Relative humidity percentage range (%)
    "wind_speed": (0.0, 50.0),      # Wind speed limits (m/s)
    "pressure": (75.0, 110.0),      # Ground surface barometric pressure range (kPa)
}

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")


class DataValidator:
    """Validates raw meteorological datasets and produces quality audit metrics."""

    def __init__(self, bounds: Dict[str, Tuple[float, float]] = PHYSICAL_BOUNDS):
        self.bounds = bounds

    def validate_hub(self, hub_id: str, raw_dir: str = RAW_DATA_DIR, output_dir: str = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate, clean, and audit a single hub's raw dataset.
        
        Args:
            hub_id: Unique hub key in BENCHMARK_HUBS (e.g., 'palakkad').
            raw_dir: Path to directory containing raw CSV files.
            output_dir: Path to save validated CSV outputs.
            
        Returns:
            Tuple of (cleaned_df, audit_report_dict).
        """
        raw_csv = os.path.join(raw_dir, f"{hub_id}_raw.csv")
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(f"Raw dataset not found at '{raw_csv}'. Run download_hubs first.")

        hub_name = BENCHMARK_HUBS.get(hub_id, {}).get("name", hub_id)
        logger.info(f"--- Starting Validation & Quality Audit: {hub_name} ---")

        # 1. Load raw data with timestamp index
        df = pd.read_csv(raw_csv, parse_dates=["timestamp"], index_col="timestamp")
        initial_rows = len(df)
        audit: Dict[str, Any] = {
            "hub_id": hub_id,
            "hub_name": hub_name,
            "initial_rows": initial_rows,
            "missing_timestamps_filled": 0,
            "duplicate_timestamps_removed": 0,
            "missing_values_repaired": {},
            "bounds_violations_corrected": {},
        }

        # 2. Check and clean duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            df = df[~df.index.duplicated(keep="first")]
            audit["duplicate_timestamps_removed"] = int(duplicates)
            logger.warning(f"Removed {duplicates} duplicate timestamps.")

        # 3. Enforce strictly continuous hourly chronological frequency
        df = df.sort_index()
        expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1h")
        missing_hours = len(expected_range) - len(df)
        if missing_hours > 0:
            df = df.reindex(expected_range)
            audit["missing_timestamps_filled"] = int(missing_hours)
            logger.warning(f"Detected and reindexed {missing_hours} missing hourly timestamps.")

        df.index.name = "timestamp"

        # 4. Audit and correct physical bound violations
        for col, (lower_limit, upper_limit) in self.bounds.items():
            if col in df.columns:
                # Count missing before bounds check
                nulls_before = int(df[col].isna().sum())
                audit["missing_values_repaired"][col] = nulls_before

                # Count bound violations
                violations_lower = (df[col] < lower_limit).sum()
                violations_upper = (df[col] > upper_limit).sum()
                total_violations = int(violations_lower + violations_upper)
                audit["bounds_violations_corrected"][col] = total_violations

                # Clip variables to physical ranges
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

        # 5. Night-time physical solar constraint (nocturnal irradiance must be strictly 0.0)
        # Night hours: 19:00 to 05:00 local solar time
        night_mask = (df.index.hour >= 19) | (df.index.hour <= 5)
        night_drifts_ghi = int((df.loc[night_mask, "ghi"] > 0.0).sum())
        df.loc[night_mask, "ghi"] = 0.0
        df.loc[night_mask, "dni"] = 0.0
        audit["night_drifts_clipped_to_zero"] = night_drifts_ghi

        # 6. Time-based linear interpolation for remaining sensor dropouts
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].interpolate(method="time").bfill().ffill()

        # 7. Final integrity audit
        audit["final_rows"] = len(df)
        audit["final_null_count"] = int(df.isna().sum().sum())
        audit["date_range"] = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"

        # 8. Save validated dataset
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, f"{hub_id}_validated.csv")
        df.to_csv(out_csv, index=True)
        logger.info(f"Saved validated dataset to {out_csv} ({len(df):,} clean rows, 0 nulls).")

        return df, audit


def validate_all_hubs(raw_dir: str = RAW_DATA_DIR, output_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Validate all benchmark hubs and print formatted audit summary table."""
    validator = DataValidator()
    all_audits = {}

    print("\n" + "=" * 95)
    print("                      H2-GRID INDIA: DATA QUALITY AUDIT REPORT")
    print("=" * 95)
    print(f"{'Hub Name':<28} | {'Rows':<8} | {'Missing Timestamps':<18} | {'Repaired NaNs':<14} | {'Status':<10}")
    print("-" * 95)

    for hub_id in BENCHMARK_HUBS:
        _, audit = validator.validate_hub(hub_id, raw_dir=raw_dir, output_dir=output_dir)
        all_audits[hub_id] = audit
        total_nans_repaired = sum(audit["missing_values_repaired"].values())
        print(
            f"{audit['hub_name']:<28} | "
            f"{audit['final_rows']:<8} | "
            f"{audit['missing_timestamps_filled']:<18} | "
            f"{total_nans_repaired:<14} | "
            f"{'PASSED' if audit['final_null_count'] == 0 else 'FAILED':<10}"
        )

    print("=" * 95)
    print("Physical bounds verified: GHI/DNI >= 0 W/m^2, Humidity [0,100]%, Night-time drift clamped strictly to 0.0 W/m^2.")
    print("=" * 95 + "\n")

    return all_audits


if __name__ == "__main__":
    validate_all_hubs()
