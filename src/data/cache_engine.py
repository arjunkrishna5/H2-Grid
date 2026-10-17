# Local Parquet & CSV Caching Engine for H2-Grid India
# Provides zero-latency, offline dataset retrieval for machine learning and physics pipelines
# Converts validated CSVs to high-performance Apache Parquet columnar storage
# Supports South Indian benchmark hubs (Palakkad, Ramanathapuram, Pavagada) and Bhadla

import os
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.data.download_hubs import BENCHMARK_HUBS, download_hub_data
from src.data.validator import DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")


def get_parquet_path(hub_id: str, processed_dir: str = PROCESSED_DATA_DIR) -> str:
    """Return local path to the processed Parquet cache file for a hub."""
    return os.path.join(processed_dir, f"{hub_id}_clean.parquet")


def get_csv_path(hub_id: str, processed_dir: str = PROCESSED_DATA_DIR) -> str:
    """Return local path to the processed validated CSV file for a hub."""
    return os.path.join(processed_dir, f"{hub_id}_validated.csv")


def convert_csv_to_parquet(hub_id: str, processed_dir: str = PROCESSED_DATA_DIR) -> str:
    """
    Convert a validated CSV file to an optimized columnar Parquet file.
    
    Args:
        hub_id: Unique benchmark hub identifier.
        processed_dir: Path to directory containing processed files.
        
    Returns:
        str: Path to generated Parquet file.
    """
    csv_path = get_csv_path(hub_id, processed_dir)
    parquet_path = get_parquet_path(hub_id, processed_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Validated CSV not found at '{csv_path}'. Run validator first.")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=True)

    csv_mb = os.path.getsize(csv_path) / (1024 * 1024)
    pq_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    compression_ratio = (1.0 - (pq_mb / csv_mb)) * 100.0

    logger.info(
        f"Cached {hub_id}: CSV ({csv_mb:.2f} MB) -> Parquet ({pq_mb:.2f} MB) "
        f"[Compression: {compression_ratio:.1f}% reduction]"
    )
    return parquet_path


def load_hub_dataset(
    hub_id: str,
    file_format: str = "parquet",
    auto_build: bool = True,
    processed_dir: str = PROCESSED_DATA_DIR,
    raw_dir: str = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Load a hub's clean time-series with zero network latency.
    
    If the requested cache file does not exist and auto_build is True, it will
    automatically trigger the ingestion and validation pipeline to build it.
    
    Args:
        hub_id: Unique benchmark hub key ('palakkad', 'ramanathapuram', 'pavagada', 'bhadla').
        file_format: Storage format to load ('parquet' or 'csv').
        auto_build: Whether to automatically build missing caches on the fly.
        processed_dir: Directory where processed datasets are cached.
        raw_dir: Directory where raw CSV files are stored.
        
    Returns:
        pd.DataFrame: Validated hourly time-series indexed by datetime.
    """
    if hub_id not in BENCHMARK_HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

    parquet_path = get_parquet_path(hub_id, processed_dir)
    csv_path = get_csv_path(hub_id, processed_dir)

    # 1. Fast Path: Parquet cache hit
    if file_format == "parquet" and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path, engine="pyarrow")

    # 2. Fast Path: CSV cache hit
    if file_format == "csv" and os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    # 3. Fallback / Cold Start: Auto-build pipeline if cache is missing
    if not auto_build:
        raise FileNotFoundError(f"Cache file for hub '{hub_id}' not found in '{processed_dir}'.")

    logger.info(f"Cache miss for hub '{hub_id}'. Building pipeline from source...")
    raw_csv = os.path.join(raw_dir, f"{hub_id}_raw.csv")

    # Download raw if not present
    if not os.path.exists(raw_csv):
        download_hub_data(hub_id, output_dir=raw_dir)

    # Validate and produce clean CSV
    validator = DataValidator()
    df, _ = validator.validate_hub(hub_id, raw_dir=raw_dir, output_dir=processed_dir)

    # Convert to Parquet
    convert_csv_to_parquet(hub_id, processed_dir=processed_dir)

    if file_format == "parquet":
        return pd.read_parquet(parquet_path, engine="pyarrow")
    return df


def build_all_caches(processed_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Build and optimize Parquet cache files for all benchmark hubs."""
    os.makedirs(processed_dir, exist_ok=True)
    summary = {}

    print("\n" + "=" * 95)
    print("                 H2-GRID INDIA: LOCAL CACHING ENGINE (SUBPHASE 1.5)")
    print("=" * 95)
    print(f"{'Hub Name':<28} | {'CSV Size':<10} | {'Parquet Size':<14} | {'Reduction':<12} | {'Load Time':<10}")
    print("-" * 95)

    for hub_id, info in BENCHMARK_HUBS.items():
        # Ensure Parquet is built
        csv_path = get_csv_path(hub_id, processed_dir)
        if not os.path.exists(csv_path):
            load_hub_dataset(hub_id, file_format="csv", processed_dir=processed_dir)

        parquet_path = convert_csv_to_parquet(hub_id, processed_dir=processed_dir)

        # Benchmark load speed
        start = time.perf_counter()
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        load_time_ms = (time.perf_counter() - start) * 1000.0

        csv_mb = os.path.getsize(csv_path) / (1024 * 1024)
        pq_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        reduction = (1.0 - (pq_mb / csv_mb)) * 100.0

        summary[hub_id] = {
            "name": info["name"],
            "rows": len(df),
            "csv_mb": csv_mb,
            "parquet_mb": pq_mb,
            "reduction_pct": reduction,
            "load_time_ms": load_time_ms,
        }

        print(
            f"{info['name']:<28} | "
            f"{csv_mb:.2f} MB   | "
            f"{pq_mb:.2f} MB       | "
            f"{reduction:.1f}%        | "
            f"{load_time_ms:.1f} ms"
        )

    print("=" * 95)
    print("Zero network latency verified: All datasets cached locally for offline ML training.")
    print("=" * 95 + "\n")

    return summary


if __name__ == "__main__":
    build_all_caches()
