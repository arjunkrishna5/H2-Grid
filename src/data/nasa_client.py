# NASA POWER API Automated Ingestion Client for H2-Grid India
# Queries hourly solar irradiance (GHI, DNI) and meteorology (T2M, RH2M, WS10M, PS)
# Implements retry logic with exponential backoff and returns structured DataFrames
# Primary focus: South Indian Renewable Corridors and Pan-India validation hubs

import time
import logging
from typing import Optional, Dict, List
import requests
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# NASA POWER parameters per Dossier Section 5
DEFAULT_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",  # Global Horizontal Irradiance (GHI, W/m^2)
    "ALLSKY_SFC_SW_DNI",  # Direct Normal Irradiance (DNI, W/m^2)
    "T2M",                # Air Temperature at 2 meters (deg C)
    "RH2M",               # Relative Humidity at 2 meters (%)
    "WS10M",              # Wind Speed at 10 meters (m/s)
    "PS",                 # Surface Pressure (kPa)
]

COLUMN_MAPPING = {
    "ALLSKY_SFC_SW_DWN": "ghi",
    "ALLSKY_SFC_SW_DNI": "dni",
    "T2M": "temperature",
    "RH2M": "humidity",
    "WS10M": "wind_speed",
    "PS": "pressure",
}

class NASAClient:
    """Automated client for NASA POWER Hourly Point API."""
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, timeout: int = 30):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def fetch_hourly_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly weather and irradiance data for a specific coordinate.
        
        Args:
            latitude: Target latitude in decimal degrees.
            longitude: Target longitude in decimal degrees.
            start_date: Start date string (YYYYMMDD or YYYY-MM-DD).
            end_date: End date string (YYYYMMDD or YYYY-MM-DD).
            parameters: Optional list of NASA POWER parameter codes.
            
        Returns:
            pd.DataFrame: Hourly indexed DataFrame with cleaned weather variables.
        """
        params_list = parameters or DEFAULT_PARAMETERS
        start_fmt = start_date.replace("-", "").strip()
        end_fmt = end_date.replace("-", "").strip()
        
        query_params = {
            "parameters": ",".join(params_list),
            "community": "RE",
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": start_fmt,
            "end": end_fmt,
            "format": "JSON",
        }
        
        logger.info(
            f"Querying NASA POWER API for coords ({latitude:.4f}, {longitude:.4f}) "
            f"from {start_fmt} to {end_fmt}..."
        )
        
        response_json = self._request_with_retries(query_params)
        return self._parse_response(response_json)

    def _request_with_retries(self, query_params: Dict[str, str]) -> dict:
        """Execute HTTP GET with exponential backoff on transient errors."""
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=query_params,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: Server returned HTTP {response.status_code}. Retrying..."
                    )
                else:
                    response.raise_for_status()
                    
            except (requests.RequestException, requests.Timeout) as exc:
                last_exception = exc
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {exc}")
                
            if attempt < self.max_retries:
                sleep_time = self.backoff_factor ** attempt
                time.sleep(sleep_time)
                
        raise RuntimeError(
            f"Failed to fetch data from NASA POWER after {self.max_retries} attempts. "
            f"Last error: {last_exception}"
        )

    def _parse_response(self, response_json: dict) -> pd.DataFrame:
        """Parse raw NASA JSON response into a clean, typed pandas DataFrame."""
        try:
            properties = response_json["properties"]
            param_dict = properties["parameter"]
        except KeyError as e:
            raise ValueError(f"Unexpected NASA API response structure: missing key {e}")

        # Build DataFrame from timestamp dictionaries
        df = pd.DataFrame(param_dict)
        
        # Rename columns to standardized names
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Parse timestamp index (format YYYYMMDDHH)
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        df.index.name = "timestamp"
        
        # NASA uses -999.0 for missing/invalid data values
        df = df.replace(-999.0, np.nan)
        
        return df


if __name__ == "__main__":
    # Quick sanity test on South India testbed: Palakkad, Kerala (10.78°N, 76.65°E)
    client = NASAClient()
    logger.info("Executing sanity test on Palakkad (South India Hub)...")
    sample_df = client.fetch_hourly_data(
        latitude=10.78,
        longitude=76.65,
        start_date="20240101",
        end_date="20240102",
    )
    print("\n--- NASA POWER API Test Output (Palakkad Sample) ---")
    print(sample_df.head())
    print("\nShape:", sample_df.shape)
    print("Columns:", list(sample_df.columns))
