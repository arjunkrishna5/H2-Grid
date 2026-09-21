# Data ingestion package for H2-Grid India
# Exports the automated NASA POWER API client for solar and meteorological data
# Provides unified data acquisition across South Indian and Pan-India hubs

from src.data.nasa_client import NASAClient

__all__ = ["NASAClient"]
