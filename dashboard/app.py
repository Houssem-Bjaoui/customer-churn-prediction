"""
dashboard/app.py
════════════════
This is the MAIN entry point of the Streamlit dashboard.

WHAT CHANGED FROM THE ORIGINAL VERSION:
- API_URL now reads from environment variable
- This makes the same code work in TWO environments:

  Local development:
    API_URL = http://localhost:8000  (default)

  Inside Docker:
    API_URL = http://api:8000  (set in docker-compose.yml)

- We imported the config from dashboard/config.py
  so API_URL is defined in ONE place only.
  No more changing it in every single page file.
"""

import streamlit as st
import os

# ── Page configuration ────────────────────────────────────
# This MUST be the first Streamlit command in the file.
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ───────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; }

    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }

    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# ── API Configuration ─────────────────────────────────────
# THIS IS THE KEY CHANGE FROM THE ORIGINAL FILE.
#
# os.getenv("VARIABLE_NAME", "default_value") works like this:
#
#   1. Python looks for an environment variable called API_URL
#   2. If found → use that value
#   3. If NOT found → use the default "http://localhost:8000"
#
# When running locally:
#   No environment variable set → uses http://localhost:8000
#   Everything works as before.
#
# When running inside Docker:
#   docker-compose.yml sets API_URL=http://api:8000
#   Python reads that value → connects to the api container
#   "api" is the service name in docker-compose.yml
#   Docker's internal DNS resolves "api" to the container IP

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev_key_change_in_production")

# Store in session_state so other pages can access them
# st.session_state is a dictionary that persists across
# page navigation within the same browser session
st.session_state["API_URL"] = API_URL
st.session_state["API_KEY"] = API_KEY
st.session_state["HEADERS"] = {
    "Authorization": f"Bearer {API_KEY}"
}


# ── Welcome page content ──────────────────────────────────
st.title("🔴 ChurnGuard AI")
st.markdown(
    "**Customer Churn Prediction & Explainability Dashboard**"
)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "### 📊 Overview\n"
        "See your customer base at a glance. "
        "Track how many customers are at high, medium, "
        "and low risk of churning."
    )

with col2:
    st.warning(
        "### 🔍 Risk Table\n"
        "Browse all customers ranked by churn probability. "
        "Filter by risk tier. Export your at-risk list to CSV."
    )

with col3:
    st.success(
        "### 🧠 Explainer\n"
        "Select any customer and see exactly WHY the model "
        "gave them that score — powered by SHAP values."
    )

st.markdown("---")
st.markdown("### 👈 Use the sidebar to navigate")
st.markdown(
    "Each page connects to the FastAPI backend. "
    "Make sure the API is running before using this dashboard."
)


# ── Show which environment we are in ─────────────────────
# This is a small debug helper that tells you at a glance
# whether you are running locally or inside Docker.
# Very useful when something is not connecting correctly.

environment = os.getenv("ENVIRONMENT", "development")

if environment == "production":
    # Inside Docker
    st.sidebar.success("🐳 Running in Docker")
else:
    # Local development
    st.sidebar.info("💻 Running locally")

# Show the API URL being used — good for debugging
st.sidebar.caption(f"API: `{API_URL}`")


# ── API Connection Status ─────────────────────────────────
st.markdown("### 🔗 API Connection Status")

import requests

try:
    response = requests.get(
        f"{API_URL}/health",
        timeout=5  # Slightly longer timeout for Docker startup
    )

    if response.status_code == 200:
        data = response.json()

        st.success(
            f"✅ API is online — "
            f"Model: {data.get('model_version', 'unknown')} | "
            f"AUC-ROC: {data.get('auc_roc', 'N/A')} | "
            f"Uptime: {data.get('uptime_seconds', 0):.0f}s"
        )

        # Show environment info in a clean table
        col_env1, col_env2, col_env3 = st.columns(3)
        with col_env1:
            st.metric(
                "Environment",
                "Docker" if environment == "production"
                else "Local"
            )
        with col_env2:
            st.metric(
                "API URL",
                API_URL.replace("http://", "")
            )
        with col_env3:
            st.metric(
                "Model Status",
                "Loaded ✓" if data.get("model_loaded")
                else "Not Loaded ✗"
            )

    else:
        st.error(
            f"❌ API responded with status "
            f"{response.status_code}"
        )

except requests.exceptions.ConnectionError:
    st.error(
        f"❌ Cannot connect to API at `{API_URL}`."
    )

    # Show different instructions based on environment
    if environment == "production":
        # Inside Docker — different troubleshooting steps
        st.markdown("""
        **Docker troubleshooting:**
```bash
        # Check if containers are running
        docker compose ps

        # Check API logs
        docker compose logs api

        # Restart everything
        docker compose down && docker compose up -d
```
        """)
    else:
        # Local development
        st.markdown("""
        **Local troubleshooting:**
```bash
        # Start the API
        uvicorn api.main:app --reload --port 8000
```
        """)

except requests.exceptions.Timeout:
    st.warning(
        f"⚠️ API connection timed out at `{API_URL}`. "
        "The API might still be starting up. "
        "Wait 10 seconds and refresh the page."
    )


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ChurnGuard AI · Built with XGBoost + SHAP + FastAPI + "
    "Streamlit · Portfolio Project"
)