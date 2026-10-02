# Data ingestion package for H2-Grid India
# Exports NASA POWER API client and automated multi-year hub extraction engines
# Configured for South Indian renewable corridors and Pan-India validation hubs

from src.data.nasa_client import NASAClient
from src.data.download_hubs import BENCHMARK_HUBS, download_hub_data, download_all_hubs

__all__ = ["NASAClient", "BENCHMARK_HUBS", "download_hub_data", "download_all_hubs"]
