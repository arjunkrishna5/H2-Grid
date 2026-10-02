# Automated multi-year dataset extraction engine for H2-Grid India
# Downloads 5-year hourly solar irradiance and meteorological data from NASA POWER API
# Targets South Indian renewable corridors (Palakkad, Ramanathapuram, Pavagada) and Bhadla
# Saves raw tabular time-series into data/raw/ directory with validation logging

import os
import time
import logging
from typing import Dict, Any
import pandas as pd

from src.data.nasa_client import NASAClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Benchmark Hub Registry per Dossier v2 (Section 5)
BENCHMARK_HUBS: Dict[str, Dict[str, Any]] = {
    "palakkad": {
        "name": "Palakkad, Kerala",
        "latitude": 10.78,
        "longitude": 76.65,
        "climate_zone": "Wet tropical / Western Ghats monsoon",
        "description": "South India monsoon cloud turbulence & high-rainfall stress-test site",
    },
    "ramanathapuram": {
        "name": "Ramanathapuram, Tamil Nadu",
        "latitude": 9.36,
        "longitude": 78.83,
        "climate_zone": "Tropical coastal",
        "description": "South India NGHM port corridor; co-located solar and marine breeze",
    },
    "pavagada": {
        "name": "Pavagada, Karnataka",
        "latitude": 14.10,
        "longitude": 77.28,
        "climate_zone": "Semi-arid Deccan Plateau",
        "description": "South India mega-scale solar park with high direct irradiance",
    },
    "bhadla": {
        "name": "Bhadla, Rajasthan",
        "latitude": 27.53,
        "longitude": 71.91,
        "climate_zone": "Hot arid / Desert",
        "description": "National benchmark site; highest DNI in India, extreme heat derating",
    },
}

DEFAULT_START_DATE = "20190101"
DEFAULT_END_DATE = "20231231"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")


def download_hub_data(
    hub_id: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: str = RAW_DATA_DIR,
    force_redownload: bool = False,
) -> str:
    """Download multi-year hourly data for a single hub and save to data/raw/."""
    if hub_id not in BENCHMARK_HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

    hub_info = BENCHMARK_HUBS[hub_id]
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{hub_id}_raw.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    if os.path.exists(csv_path) and not force_redownload:
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        logger.info(f"Dataset for {hub_info['name']} already exists at {csv_path} ({file_size_mb:.2f} MB). Skipping.")
        return csv_path

    logger.info(f"--- Starting Download: {hub_info['name']} ({hub_info['climate_zone']}) ---")
    logger.info(f"Coordinates: ({hub_info['latitude']}°N, {hub_info['longitude']}°E) | Period: {start_date} to {end_date}")

    client = NASAClient(max_retries=3, backoff_factor=2.0, timeout=60)
    start_time = time.time()

    df = client.fetch_hourly_data(
        latitude=hub_info["latitude"],
        longitude=hub_info["longitude"],
        start_date=start_date,
        end_date=end_date,
    )

    # Save to CSV with indexed timestamp
    df.to_csv(csv_path, index=True)
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    logger.info(
        f"Successfully saved {hub_info['name']} to {csv_filename} "
        f"({len(df):,} rows, {file_size_mb:.2f} MB) in {elapsed:.1f}s"
    )
    return csv_path


def download_all_hubs(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: str = RAW_DATA_DIR,
    force_redownload: bool = False,
) -> Dict[str, str]:
    """Download multi-year hourly data for all benchmark hubs."""
    results = {}
    total_start = time.time()
    logger.info(f"Initializing extraction for {len(BENCHMARK_HUBS)} benchmark hubs ({start_date} to {end_date})...")

    for hub_id in BENCHMARK_HUBS:
        try:
            csv_path = download_hub_data(
                hub_id=hub_id,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                force_redownload=force_redownload,
            )
            results[hub_id] = csv_path
            # Polite delay between NASA API requests
            time.sleep(1.0)
        except Exception as e:
            logger.error(f"Failed downloading data for hub '{hub_id}': {e}")
            raise

    total_elapsed = time.time() - total_start
    logger.info(f"All {len(results)} hubs extracted successfully in {total_elapsed:.1f}s.")
    return results


if __name__ == "__main__":
    download_all_hubs()
