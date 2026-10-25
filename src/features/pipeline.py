# Physical Night-Time Zero-Clipping & Feature Scaling Pipeline for H2-Grid India
# Enforces zero solar radiation when Solar Zenith Angle exceeds 90 degrees
# Fits MinMaxScaler and generates model-ready scaled datasets and invertible scaler artifacts
# Applied across South Indian renewable corridors (Palakkad, Ramanathapuram, Pavagada) and Bhadla

import os
import json
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.data.download_hubs import BENCHMARK_HUBS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

# Target prediction variable
TARGET_COLUMN = "ghi"

# Categorical and cyclical columns that are already normalized or bounded
PASSTHROUGH_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_monsoon",
    "season_code",
    "season_winter",
    "season_summer",
    "season_monsoon",
    "season_post_monsoon",
    "is_monsoon_transition",
]


class FeaturePipeline:
    """Manages physical zero-clipping, normalization, and scaler persistence."""

    def __init__(self, processed_dir: str = PROCESSED_DATA_DIR):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def apply_physical_zero_clipping(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Enforce strict zero solar radiation when solar zenith angle >= 90 degrees.
        """
        df_out = df.copy()
        if "solar_zenith" not in df_out.columns:
            raise KeyError("Column 'solar_zenith' required for physical zero-clipping.")

        # Night condition: Sun is on or below the horizon
        night_mask = df_out["solar_zenith"] >= 90.0
        nocturnal_leakage_count = int(((df_out["ghi"] > 0.0) & night_mask).sum())

        # Clamp solar columns to absolute zero during night
        df_out.loc[night_mask, "ghi"] = 0.0
        if "dni" in df_out.columns:
            df_out.loc[night_mask, "dni"] = 0.0
        if "cos_zenith" in df_out.columns:
            df_out.loc[night_mask, "cos_zenith"] = 0.0

        return df_out, nocturnal_leakage_count

    def process_hub(self, hub_id: str) -> Dict[str, Any]:
        """
        Execute full end-to-end feature pipeline for a single hub.
        """
        if hub_id not in BENCHMARK_HUBS:
            raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

        hub_info = BENCHMARK_HUBS[hub_id]
        logger.info(f"--- Running Final Feature Pipeline: {hub_info['name']} ---")

        # 1. Load latest feature-engineered dataset (from Subphase 2.4 monsoon step)
        monsoon_pq = os.path.join(self.processed_dir, f"{hub_id}_monsoon.parquet")
        if os.path.exists(monsoon_pq):
            df = pd.read_parquet(monsoon_pq, engine="pyarrow")
        else:
            from src.features.monsoon import process_hub_monsoon
            df, _ = process_hub_monsoon(hub_id, processed_dir=self.processed_dir)

        # 2. Enforce physical nocturnal zero-clipping
        df_clipped, night_clipped_count = self.apply_physical_zero_clipping(df)

        # 3. Identify columns to scale vs passthrough
        all_cols = list(df_clipped.columns)
        scale_cols = [c for c in all_cols if c not in PASSTHROUGH_COLUMNS]

        # 4. Fit MinMaxScaler on continuous physical features
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled_values = scaler.fit_transform(df_clipped[scale_cols])
        df_scaled = df_clipped.copy()
        df_scaled[scale_cols] = scaled_values

        # 5. Persist artifacts
        # A. Fitted Scaler object for future inference & prediction inversion
        scaler_path = os.path.join(self.processed_dir, f"{hub_id}_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # B. Unscaled complete features (for tree models / inspection)
        unscaled_pq = os.path.join(self.processed_dir, f"{hub_id}_features_unscaled.parquet")
        df_clipped.to_parquet(unscaled_pq, engine="pyarrow", compression="snappy", index=True)

        # C. Scaled model-ready features (for BiLSTM / neural nets)
        scaled_pq = os.path.join(self.processed_dir, f"{hub_id}_features_scaled.parquet")
        scaled_csv = os.path.join(self.processed_dir, f"{hub_id}_features_scaled.csv")
        df_scaled.to_parquet(scaled_pq, engine="pyarrow", compression="snappy", index=True)
        df_scaled.to_csv(scaled_csv, index=True)

        # D. Metadata schema
        metadata_path = os.path.join(self.processed_dir, f"{hub_id}_metadata.json")
        meta = {
            "hub_id": hub_id,
            "hub_name": hub_info["name"],
            "total_rows": len(df_scaled),
            "total_columns": len(df_scaled.columns),
            "target_column": TARGET_COLUMN,
            "scaled_columns": scale_cols,
            "passthrough_columns": PASSTHROUGH_COLUMNS,
            "night_clipped_count": night_clipped_count,
            "date_range": f"{df_scaled.index.min().strftime('%Y-%m-%d')} to {df_scaled.index.max().strftime('%Y-%m-%d')}",
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Successfully processed and scaled {hub_info['name']} ({len(df_scaled):,} rows, {len(all_cols)} cols).")
        return meta


def build_all_pipelines(processed_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Build and scale final feature matrices across all 4 benchmark hubs."""
    pipeline = FeaturePipeline(processed_dir=processed_dir)
    results = {}

    print("\n" + "=" * 105)
    print("               H2-GRID INDIA: FINAL FEATURE SCALING PIPELINE (SUBPHASE 2.5)")
    print("=" * 105)
    print(f"{'Hub Name':<28} | {'Rows':<8} | {'Features':<9} | {'Scaled Cols':<12} | {'Night Clips':<12} | {'Status':<8}")
    print("-" * 105)

    for hub_id in BENCHMARK_HUBS:
        meta = pipeline.process_hub(hub_id)
        results[hub_id] = meta
        print(
            f"{meta['hub_name']:<28} | "
            f"{meta['total_rows']:<8} | "
            f"{meta['total_columns']:<9} | "
            f"{len(meta['scaled_columns']):<12} | "
            f"{meta['night_clipped_count']:<12} | "
            f"{'COMPLETED':<8}"
        )

    print("=" * 105)
    print("All datasets certified and normalized [0.0, 1.0]. Scalers saved for model training.")
    print("=" * 105 + "\n")

    return results


if __name__ == "__main__":
    build_all_pipelines()
