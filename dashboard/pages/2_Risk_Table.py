"""
dashboard/pages/2_Risk_Table.py
════════════════════════════════
PURPOSE: Show a sortable, filterable table of all customers
ranked by churn probability. Allow export to CSV.

KEY CONCEPTS FOR JUNIORS:
- st.sidebar = controls on the left side panel
- st.dataframe() = interactive table (sortable, scrollable)
- st.download_button() = lets user download a file
- Session state = remembers values between page interactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from pathlib import Path
import io

st.set_page_config(
    page_title="Risk Table — ChurnGuard AI",
    page_icon="🔍",
    layout="wide"
)

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev_key_change_in_production")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
DATA_PATH = Path("data/processed/telecom_features.parquet")


# ── Load and predict ──────────────────────────────────────
@st.cache_data
def load_and_predict() -> pd.DataFrame:
    """
    Load customers, call API, merge results.
    Returns one DataFrame with features + predictions.
    Cached so it only runs once per session.
    """
    # Load customer data
    df = pd.read_parquet(DATA_PATH)
    df_sample = df.sample(n=min(500, len(df)),
                           random_state=42).reset_index(drop=True)

    # Add a customer ID column for display
    df_sample['customer_id'] = [
        f"CUST-{i:04d}" for i in range(len(df_sample))
    ]

    # Call batch API
    csv_buffer = io.StringIO()
    df_sample.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/batch",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": ("customers.csv",
                            csv_buffer.getvalue(),
                            "text/csv")},
            timeout=60
        )
        response.raise_for_status()
        preds = pd.DataFrame(
            response.json()['predictions']
        )

        # Merge features with predictions
        df_result = pd.concat([
            df_sample,
            preds[['churn_probability',
                   'churn_label',
                   'risk_tier']]
        ], axis=1)

        return df_result

    except Exception as e:
        st.warning(f"⚠️ API unavailable — showing demo data. ({e})")

        # Demo fallback
        np.random.seed(42)
        probs = np.random.beta(2, 5, len(df_sample))
        df_sample['churn_probability'] = probs
        df_sample['churn_label'] = (probs > 0.42).astype(int)
        df_sample['risk_tier'] = pd.cut(
            probs,
            bins=[0, 0.35, 0.65, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        ).astype(str)
        return df_sample


# ── Page layout ───────────────────────────────────────────
st.title("🔍 Customer Risk Table")
st.markdown(
    "Browse all customers ranked by churn probability. "
    "Use the filters on the left to narrow down the list."
)
st.markdown("---")

# Load data with spinner
with st.spinner("Loading customer data..."):
    df = load_and_predict()

# ── SIDEBAR FILTERS ───────────────────────────────────────
# Everything inside "with st.sidebar:" appears on the left panel
with st.sidebar:
    st.header("🎛️ Filters")

    # FILTER 1: Risk tier (multi-select checkbox)
    # st.multiselect lets the user pick multiple options
    selected_tiers = st.multiselect(
        label="Risk Tier",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
        help="Filter customers by their risk classification"
    )

    # FILTER 2: Churn probability range (slider)
    # st.slider with two handles = range slider
    prob_range = st.slider(
        label="Churn Probability Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),    # Default: show all
        step=0.01,
        format="%.2f",       # Show 2 decimal places
        help="Only show customers within this probability range"
    )

    # FILTER 3: Contract type
    # Get unique values from the data for the options list
    contract_options = df['contract'].unique().tolist()
    selected_contracts = st.multiselect(
        label="Contract Type",
        options=contract_options,
        default=contract_options,
        help="Filter by contract type"
    )

    # FILTER 4: Maximum monthly charges
    max_charge = st.number_input(
        label="Max Monthly Charges ($)",
        min_value=0,
        max_value=500,
        value=300,
        step=10,
        help="Only show customers below this monthly charge"
    )

    st.markdown("---")

    # SORT OPTIONS
    sort_by = st.selectbox(
        label="Sort By",
        options=[
            "churn_probability",
            "monthly_charges",
            "tenure_months"
        ],
        index=0,  # Default to churn_probability
        help="Column to sort the table by"
    )
    sort_ascending = st.checkbox(
        "Sort Ascending",
        value=False  # Default: highest risk first
    )


# ── APPLY FILTERS ─────────────────────────────────────────
# We apply each filter one by one using boolean masks.
# A boolean mask is a True/False Series that selects rows.

# Step 1: Risk tier filter
mask_tier = df['risk_tier'].isin(selected_tiers)

# Step 2: Probability range filter
mask_prob = (
    (df['churn_probability'] >= prob_range[0]) &
    (df['churn_probability'] <= prob_range[1])
)

# Step 3: Contract filter
mask_contract = df['contract'].isin(selected_contracts)

# Step 4: Monthly charges filter
mask_charge = df['monthly_charges'] <= max_charge

# Combine all masks with & (AND — all conditions must be true)
df_filtered = df[
    mask_tier & mask_prob & mask_contract & mask_charge
].copy()

# Apply sort
df_filtered = df_filtered.sort_values(
    by=sort_by,
    ascending=sort_ascending
)


# ── FILTER SUMMARY ────────────────────────────────────────
# Show the user how many rows match their filters
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Customers Shown", f"{len(df_filtered):,}")
with col_info2:
    if len(df_filtered) > 0:
        st.metric(
            "High Risk in Selection",
            f"{(df_filtered['risk_tier']=='HIGH').sum():,}"
        )
with col_info3:
    if len(df_filtered) > 0:
        st.metric(
            "Avg Churn Prob",
            f"{df_filtered['churn_probability'].mean():.1%}"
        )

st.markdown("---")


# ── DISPLAY TABLE ─────────────────────────────────────────
# Select which columns to show in the table
display_cols = [
    'customer_id',
    'tenure_months',
    'contract',
    'internet_service',
    'monthly_charges',
    'churn_probability',
    'risk_tier',
    'churn_label'
]

# Keep only columns that exist in the DataFrame
display_cols = [c for c in display_cols if c in df_filtered.columns]
df_display = df_filtered[display_cols].copy()

# Round probability for cleaner display
df_display['churn_probability'] = (
    df_display['churn_probability'].round(3)
)


# Add color to risk_tier column using Pandas Styler
# This makes HIGH = red background, LOW = green background
def color_risk(val):
    """
    Returns CSS color style based on risk tier value.
    This function is applied cell by cell to the risk_tier column.
    """
    if val == 'HIGH':
        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
    elif val == 'MEDIUM':
        return 'background-color: #fef3c7; color: #92400e; font-weight: bold'
    elif val == 'LOW':
        return 'background-color: #d1fae5; color: #065f46; font-weight: bold'
    return ''


def color_probability(val):
    """
    Color the churn probability cell:
    - High probability (>0.65) = red
    - Medium (0.35-0.65) = orange
    - Low (<0.35) = green
    """
    try:
        if float(val) >= 0.65:
            return 'color: #dc2626; font-weight: bold'
        elif float(val) >= 0.35:
            return 'color: #d97706; font-weight: bold'
        else:
            return 'color: #16a34a; font-weight: bold'
    except Exception:
        return ''


# Apply styling using .style.map()
# .map() applies a function to each cell individually
if len(df_display) > 0 and 'risk_tier' in df_display.columns:
    styled_df = df_display.style.map(
        color_risk, subset=['risk_tier']
    ).map(
        color_probability, subset=['churn_probability']
    )

    # st.dataframe() renders the styled table
    # height controls how many rows are visible before scrolling
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )
else:
    st.warning("No customers match the selected filters.")


# ── EXPORT BUTTON ─────────────────────────────────────────
# st.download_button() creates a button that downloads a file
# The user clicks it and gets a CSV saved to their computer

if len(df_filtered) > 0:
    # Convert DataFrame to CSV string
    # index=False removes the row numbers from the CSV
    csv_export = df_filtered.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Filtered List as CSV",
        data=csv_export,          # The file content
        file_name="at_risk_customers.csv",  # Suggested filename
        mime="text/csv",          # File type
        help="Downloads all customers currently shown in the table"
    )

    st.caption(
        f"💡 Showing {len(df_filtered):,} customers | "
        f"Sorted by {sort_by} "
        f"({'ascending' if sort_ascending else 'descending'})"
    )