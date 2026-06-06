"""
dashboard/config.py
════════════════════
Centralized configuration for the dashboard.

WHY THIS FILE EXISTS:
Without this file, API_URL is hardcoded as
"http://localhost:8000" in every page file.

Inside Docker, the API is at "http://api:8000"
not localhost. If we hardcode localhost, the
dashboard works locally but fails in Docker.

By reading from an environment variable, the same
code works in both environments:
- Local dev: API_URL=http://localhost:8000
- Docker:    API_URL=http://api:8000

os.getenv("API_URL", "http://localhost:8000") means:
  "Read API_URL from environment.
   If not set, use http://localhost:8000 as default."
"""

import os

# API connection settings
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev_key_change_in_production")

# Headers used in every API request
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Data paths
DATA_PATH = os.getenv(
    "DATA_PATH",
    "data/processed/telecom_features.parquet"
)