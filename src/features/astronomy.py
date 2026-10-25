# Solar Astronomical Geometry Engine for H2-Grid India
# Computes physical solar position features: Zenith, Declination, Hour Angle, and Azimuth
# Provides deterministic celestial physics to ground machine learning solar forecasts
# Applied across South Indian renewable corridors (Palakkad, Ramanathapuram, Pavagada) and Bhadla

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import pvlib

from src.data.download_hubs import BENCHMARK_HUBS
from src.data.cache_engine import load_hub_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")


def compute_solar_geometry(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate high-precision astronomical solar geometry features using pvlib.
    
    Args:
        df: DataFrame with DatetimeIndex (hourly frequency).
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        altitude: Elevation above sea level in meters.
        
    Returns:
        pd.DataFrame: Original DataFrame enriched with celestial physics columns.
    """
    df_out = df.copy()

    # Ensure index is timezone-aware for exact solar positioning (NASA POWER defaults to UTC)
    if df_out.index.tz is None:
        times = df_out.index.tz_localize("UTC")
    else:
        times = df_out.index.tz_convert("UTC")

    # 1. Solar Position (Zenith, Elevation, Azimuth) via NREL SPA / ephemeris
    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    df_out["solar_zenith"] = solpos["zenith"].values
    df_out["solar_elevation"] = solpos["elevation"].values
    df_out["solar_azimuth"] = solpos["azimuth"].values

    # 2. Solar Declination (Seasonal tilt angle in degrees: -23.45 to +23.45)
    # Using Cooper (1969) / Spencer equation via day of year
    doy = times.dayofyear.values
    declination_rad = np.radians(23.45) * np.sin(np.radians((360.0 / 365.0) * (284 + doy)))
    df_out["solar_declination"] = np.degrees(declination_rad)

    # 3. Solar Hour Angle (Degrees from local solar noon: 15 deg/hr)
    # Equation of Time (EoT) in minutes
    gamma = 2.0 * np.pi * (doy - 1) / 365.0
    eot_minutes = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    # Local Solar Time (LST) in hours
    utc_hours = times.hour + (times.minute / 60.0)
    solar_time_hours = utc_hours + (4.0 * longitude + eot_minutes) / 60.0
    # Hour angle: 0 at solar noon, negative before noon, positive after noon
    hour_angle_deg = (solar_time_hours - 12.0) * 15.0
    # Wrap hour angle strictly to [-180, 180]
    hour_angle_deg = (hour_angle_deg + 180.0) % 360.0 - 180.0
    df_out["hour_angle"] = hour_angle_deg

    # 4. Cosine of Solar Zenith (Inductive physics bias: beam projection on horizontal surface)
    # Clamped to 0.0 when sun is at or below the horizon (zenith >= 90 deg)
    cos_zenith = np.cos(np.radians(df_out["solar_zenith"].values))
    df_out["cos_zenith"] = np.where(df_out["solar_zenith"].values < 90.0, np.maximum(0.0, cos_zenith), 0.0)

    return df_out


def process_hub_astronomy(
    hub_id: str,
    output_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load validated hub data, calculate solar astronomy features, and save enriched dataset.
    """
    if hub_id not in BENCHMARK_HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Choose from: {list(BENCHMARK_HUBS.keys())}")

    hub_info = BENCHMARK_HUBS[hub_id]
    logger.info(f"Computing solar geometry features for {hub_info['name']}...")

    # Load validated dataset from Phase 1
    df = load_hub_dataset(hub_id, file_format="parquet")

    # Compute astronomical geometry
    df_enriched = compute_solar_geometry(
        df=df,
        latitude=hub_info["latitude"],
        longitude=hub_info["longitude"],
    )

    # Save enriched outputs (both CSV and Parquet)
    os.makedirs(output_dir, exist_ok=True)
    out_parquet = os.path.join(output_dir, f"{hub_id}_astronomy.parquet")
    out_csv = os.path.join(output_dir, f"{hub_id}_astronomy.csv")

    df_enriched.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=True)
    df_enriched.to_csv(out_csv, index=True)

    summary = {
        "hub_id": hub_id,
        "name": hub_info["name"],
        "rows": len(df_enriched),
        "columns": len(df_enriched.columns),
        "zenith_min": float(df_enriched["solar_zenith"].min()),
        "zenith_max": float(df_enriched["solar_zenith"].max()),
        "declination_min": float(df_enriched["solar_declination"].min()),
        "declination_max": float(df_enriched["solar_declination"].max()),
    }
    return df_enriched, summary


def process_all_hubs_astronomy(output_dir: str = PROCESSED_DATA_DIR) -> Dict[str, Dict[str, Any]]:
    """Process astronomical feature engine across all benchmark hubs and print summary table."""
    results = {}

    print("\n" + "=" * 105)
    print("                 H2-GRID INDIA: SOLAR ASTRONOMICAL GEOMETRY (SUBPHASE 2.1)")
    print("=" * 105)
    print(f"{'Hub Name':<28} | {'Rows':<8} | {'Columns':<8} | {'Zenith Range':<18} | {'Declination Range':<20} | {'Status':<8}")
    print("-" * 105)

    for hub_id in BENCHMARK_HUBS:
        _, summary = process_hub_astronomy(hub_id, output_dir=output_dir)
        results[hub_id] = summary
        print(
            f"{summary['name']:<28} | "
            f"{summary['rows']:<8} | "
            f"{summary['columns']:<8} | "
            f"{summary['zenith_min']:.1f}° to {summary['zenith_max']:.1f}°   | "
            f"{summary['declination_min']:.1f}° to {summary['declination_max']:.1f}°    | "
            f"{'COMPLETED':<8}"
        )

    print("=" * 105)
    print("Physics features generated: solar_zenith, solar_elevation, solar_azimuth, solar_declination, hour_angle, cos_zenith.")
    print("=" * 105 + "\n")

    return results


if __name__ == "__main__":
    process_all_hubs_astronomy()
