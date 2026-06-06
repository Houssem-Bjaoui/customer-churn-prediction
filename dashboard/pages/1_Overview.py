"""
dashboard/pages/1_Overview.py
══════════════════════════════
PURPOSE: Show the big picture of customer churn risk.

WHAT THIS PAGE DOES:
- Loads customer data from the processed parquet file
- Calls the API /predict endpoint for each customer
- Shows KPI metrics cards (high risk count, avg probability, etc.)
- Shows distribution charts

WHY WE USE CACHING:
Calling the API for 15,000 customers every time the page
loads would be extremely slow (minutes). Streamlit's
@st.cache_data decorator saves the result in memory.
The next time the function is called with the same arguments,
it returns the cached result instantly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Overview — ChurnGuard AI",
    page_icon="📊",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev_key_change_in_production")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
DATA_PATH = Path("data/processed/telecom_features.parquet")


# ── Cached data loader ────────────────────────────────────
# @st.cache_data tells Streamlit: "run this function once,
# save the result, and reuse it on future page loads."
# The underscore prefix in _api_url means Streamlit won't
# use it as a cache key (URLs don't affect the data).
@st.cache_data
def load_customer_data() -> pd.DataFrame:
    """
    Load the feature-engineered customer data.
    This is the cleaned + engineered dataset from notebook 03.
    We use a SAMPLE of 500 customers for the dashboard
    to keep things fast.
    """
    df = pd.read_parquet(DATA_PATH)

    # Take a reproducible sample of 500 customers
    # random_state=42 means we always get the same sample
    df_sample = df.sample(n=min(500, len(df)),
                           random_state=42)
    return df_sample.reset_index(drop=True)


@st.cache_data
def get_predictions_for_overview(
    _api_url: str,
    n_customers: int
) -> pd.DataFrame:
    """
    Call the API's /batch endpoint with a sample CSV.
    Returns a DataFrame with churn predictions.

    We use the batch endpoint because it's more efficient
    than calling /predict 500 times in a loop.
    """
    df = load_customer_data()

    # Convert DataFrame to CSV in memory
    # StringIO lets us treat a string as a file
    import io
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Go back to start of the "file"

    try:
        response = requests.post(
            f"{_api_url}/batch",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": ("customers.csv",
                            csv_buffer.getvalue(),
                            "text/csv")},
            timeout=60  # Batch can take up to 60 seconds
        )
        response.raise_for_status()
        result = response.json()

        # Extract predictions from the response
        predictions = pd.DataFrame(result['predictions'])
        return predictions

    except requests.exceptions.ConnectionError:
        # Return a demo DataFrame if API is offline
        # This way the dashboard still shows something
        st.warning(
            "⚠️ API offline — showing demo data. "
            "Start the API to see real predictions."
        )
        return _generate_demo_predictions(len(df))

    except Exception as e:
        st.error(f"Error calling API: {e}")
        return _generate_demo_predictions(len(df))


def _generate_demo_predictions(n: int) -> pd.DataFrame:
    """
    Generate fake predictions for demo purposes.
    Used when the API is not running.
    In your interview, explain this as a graceful fallback.
    """
    np.random.seed(42)
    probs = np.random.beta(a=2, b=5, size=n)  # Skewed distribution
    return pd.DataFrame({
        'churn_probability': probs,
        'churn_label'      : (probs > 0.42).astype(int),
        'risk_tier'        : pd.cut(
            probs,
            bins=[0, 0.35, 0.65, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        ),
        'status': 'success'
    })


# ════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ════════════════════════════════════════════════════════

st.title("📊 Customer Churn Overview")
st.markdown(
    "High-level view of churn risk across your customer base."
)
st.markdown("---")

# Load data
df_customers = load_customer_data()

# Show a spinner while waiting for predictions
# st.spinner() shows a loading animation
with st.spinner("Loading predictions from API..."):
    df_preds = get_predictions_for_overview(
        API_URL, len(df_customers)
    )

# Filter only successful predictions
df_success = df_preds[df_preds['status'] == 'success'].copy()

if len(df_success) == 0:
    st.error("No predictions available.")
    st.stop()  # Stop rendering the page here


# ── KPI METRICS ROW ───────────────────────────────────────
# st.columns(4) creates 4 equal-width columns side by side
st.subheader("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

total     = len(df_success)
high_risk = (df_success['risk_tier'] == 'HIGH').sum()
med_risk  = (df_success['risk_tier'] == 'MEDIUM').sum()
low_risk  = (df_success['risk_tier'] == 'LOW').sum()
avg_prob  = df_success['churn_probability'].mean()

with col1:
    # st.metric() shows a large number with a label
    # The delta parameter shows change vs previous period
    st.metric(
        label="Total Customers Analyzed",
        value=f"{total:,}",
        delta=None
    )

with col2:
    # We calculate the percentage for display
    high_pct = high_risk / total * 100
    st.metric(
        label="🔴 High Risk",
        value=f"{high_risk:,}",
        delta=f"{high_pct:.1f}% of total",
        # delta_color="inverse" makes the number red
        # when it goes up (high risk going up is bad)
        delta_color="inverse"
    )

with col3:
    med_pct = med_risk / total * 100
    st.metric(
        label="🟡 Medium Risk",
        value=f"{med_risk:,}",
        delta=f"{med_pct:.1f}% of total",
        delta_color="off"  # "off" = gray, no color signal
    )

with col4:
    st.metric(
        label="Average Churn Probability",
        value=f"{avg_prob:.1%}",
        delta=None
    )

st.markdown("---")


# ── CHARTS ROW ────────────────────────────────────────────
# Two charts side by side
st.subheader("📉 Risk Distribution")

col_left, col_right = st.columns(2)

with col_left:
    # PIE CHART — risk tier breakdown
    # We use Plotly for interactive charts
    # Plotly charts respond to hover, zoom, click

    risk_counts = df_success['risk_tier'].value_counts()

    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Customer Risk Tier Breakdown",
        color=risk_counts.index,
        # Map each tier to a specific color
        color_discrete_map={
            'HIGH'  : '#ef4444',  # Red
            'MEDIUM': '#f59e0b',  # Orange
            'LOW'   : '#22c55e'   # Green
        },
        hole=0.4  # Makes it a donut chart
    )
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=13
    )
    fig_pie.update_layout(
        showlegend=True,
        height=400
    )
    # st.plotly_chart() renders a Plotly figure
    # use_container_width=True makes it fill the column
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    # HISTOGRAM — distribution of churn probabilities
    fig_hist = px.histogram(
        df_success,
        x='churn_probability',
        nbins=40,           # Number of bars
        title="Distribution of Churn Probabilities",
        labels={
            'churn_probability': 'Churn Probability',
            'count'            : 'Number of Customers'
        },
        color_discrete_sequence=['#3b82f6']  # Blue bars
    )
    # Add a vertical line at the decision threshold
    fig_hist.add_vline(
        x=0.42,              # Our optimal threshold
        line_dash="dash",
        line_color="red",
        annotation_text="Decision Threshold (0.42)",
        annotation_position="top right"
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)


# ── CHURN RATE BY CONTRACT ────────────────────────────────
st.markdown("---")
st.subheader("📋 Churn Rate by Customer Segment")

# Merge predictions with customer features
# We need the original features to color by contract type etc.
df_merged = pd.concat([
    df_customers.reset_index(drop=True),
    df_success[['churn_probability',
                'risk_tier']].reset_index(drop=True)
], axis=1)

col_a, col_b = st.columns(2)

with col_a:
    # GROUP BY CONTRACT
    # .agg() applies multiple aggregation functions at once
    contract_stats = (
        df_merged.groupby('contract')['churn_probability']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={
            'mean' : 'avg_churn_prob',
            'count': 'n_customers'
        })
        .sort_values('avg_churn_prob', ascending=False)
    )

    fig_bar1 = px.bar(
        contract_stats,
        x='contract',
        y='avg_churn_prob',
        title="Average Churn Probability by Contract Type",
        text='avg_churn_prob',
        color='avg_churn_prob',
        color_continuous_scale='RdYlGn_r',  # Red=high, Green=low
        labels={
            'contract'       : 'Contract Type',
            'avg_churn_prob' : 'Avg Churn Probability'
        }
    )
    # Format the text labels on the bars as percentages
    fig_bar1.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside'
    )
    fig_bar1.update_layout(
        coloraxis_showscale=False,
        height=380
    )
    st.plotly_chart(fig_bar1, use_container_width=True)

with col_b:
    # GROUP BY INTERNET SERVICE
    internet_stats = (
        df_merged.groupby('internet_service')['churn_probability']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={
            'mean' : 'avg_churn_prob',
            'count': 'n_customers'
        })
        .sort_values('avg_churn_prob', ascending=False)
    )

    fig_bar2 = px.bar(
        internet_stats,
        x='internet_service',
        y='avg_churn_prob',
        title="Average Churn Probability by Internet Service",
        text='avg_churn_prob',
        color='avg_churn_prob',
        color_continuous_scale='RdYlGn_r',
        labels={
            'internet_service': 'Internet Service',
            'avg_churn_prob'  : 'Avg Churn Probability'
        }
    )
    fig_bar2.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside'
    )
    fig_bar2.update_layout(
        coloraxis_showscale=False,
        height=380
    )
    st.plotly_chart(fig_bar2, use_container_width=True)


# ── SCATTER PLOT ──────────────────────────────────────────
st.markdown("---")
st.subheader("🔵 Tenure vs Monthly Charges — Colored by Risk")

# This scatter plot shows WHERE high-risk customers cluster
# in terms of tenure and monthly charges
fig_scatter = px.scatter(
    df_merged,
    x='tenure_months',
    y='monthly_charges',
    color='risk_tier',
    title="Customer Segments: Tenure vs Monthly Charges",
    color_discrete_map={
        'HIGH'  : '#ef4444',
        'MEDIUM': '#f59e0b',
        'LOW'   : '#22c55e'
    },
    labels={
        'tenure_months'   : 'Tenure (months)',
        'monthly_charges' : 'Monthly Charges ($)',
        'risk_tier'       : 'Risk Tier'
    },
    opacity=0.6,  # Semi-transparent dots
    hover_data=['contract', 'internet_service']
)
fig_scatter.update_layout(height=450)
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "💡 Insight: High-risk customers (red) cluster in "
    "the low-tenure, high-charges area — new customers "
    "paying a lot relative to their loyalty."
)