"""
dashboard/pages/3_Customer_Explainer.py
════════════════════════════════════════
PURPOSE: Let the user enter a customer's data manually
and see the SHAP explanation for why they got their score.

This is the most technically impressive page of the dashboard.
It shows the SHAP waterfall chart and explains each feature's
contribution to the prediction.

KEY CONCEPTS FOR JUNIORS:
- st.form() = groups inputs together, submits all at once
- st.selectbox() = dropdown selection
- st.number_input() = numeric input with min/max validation
- Plotly horizontal bar chart = our custom SHAP waterfall
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Customer Explainer — ChurnGuard AI",
    page_icon="🧠",
    layout="wide"
)

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev_key_change_in_production")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ── Helper: call /predict API ─────────────────────────────
def call_predict_api(customer_data: dict) -> dict | None:
    """
    Send one customer's data to the /predict endpoint.
    Returns the full JSON response or None if it fails.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers=HEADERS,
            json=customer_data,   # Automatically serializes dict to JSON
            timeout=15
        )

        # .raise_for_status() raises an error for 4xx/5xx responses
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to API. "
            "Run: `uvicorn api.main:app --reload --port 8000`"
        )
        return None

    except requests.exceptions.HTTPError as e:
        # Extract the error detail from the response JSON
        detail = response.json().get('detail', str(e))
        st.error(f"❌ API Error: {detail}")
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


# ── Helper: draw SHAP waterfall chart ────────────────────
def draw_shap_waterfall(
    explanation: dict,
    churn_probability: float
) -> None:
    """
    Draw a horizontal bar chart that shows each feature's
    contribution to the final churn probability.

    This is our custom waterfall chart built with Plotly.
    Each bar represents one feature:
    - Red bar pointing RIGHT = increases churn probability
    - Blue bar pointing LEFT = decreases churn probability

    Args:
        explanation: dict from API with base_value and top_features
        churn_probability: the final predicted probability
    """
    features = explanation.get('top_features', [])
    base_val = explanation.get('base_value', 0)

    if not features:
        st.warning("No explanation data available.")
        return

    # Build lists for the chart
    feature_names  = [f['feature'] for f in features]
    shap_values    = [f['shap_value'] for f in features]
    directions     = [f['direction'] for f in features]

    # Colors: red = increases risk, blue = decreases risk
    colors = [
        '#ef4444' if v > 0 else '#3b82f6'
        for v in shap_values
    ]

    # Build Plotly horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=shap_values,
        y=feature_names,
        orientation='h',      # 'h' = horizontal bars
        marker_color=colors,
        marker_line_width=0,  # No border on bars
        text=[
            f"+{v:.4f}" if v > 0 else f"{v:.4f}"
            for v in shap_values
        ],
        textposition='outside',
        textfont=dict(size=11)
    ))

    # Add vertical line at x=0
    fig.add_vline(
        x=0,
        line_color='black',
        line_width=1.5
    )

    # Add annotation for base value and final prediction
    fig.add_annotation(
        x=min(shap_values) * 1.1,
        y=len(features) - 0.5,
        text=f"Base value: {base_val:.3f}",
        showarrow=False,
        font=dict(size=11, color='gray')
    )

    fig.update_layout(
        title=dict(
            text=(
                f"SHAP Feature Contributions<br>"
                f"<sup>Final prediction: {churn_probability:.1%} "
                f"churn probability</sup>"
            ),
            font=dict(size=16)
        ),
        xaxis_title="SHAP Value (contribution to prediction)",
        yaxis_title="Feature",
        height=400,
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=20, r=80, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explanation text below the chart
    st.caption(
        "🔴 Red bars push the prediction TOWARD churn | "
        "🔵 Blue bars push the prediction AWAY from churn | "
        f"Starting point (base value): {base_val:.3f}"
    )


# ════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ════════════════════════════════════════════════════════

st.title("🧠 Customer Churn Explainer")
st.markdown(
    "Enter a customer's details below to get their "
    "churn probability and a full explanation of "
    "**why** the model gave them that score."
)
st.markdown("---")

# Two column layout: form on left, results on right
col_form, col_results = st.columns([1, 1.5])

with col_form:
    st.subheader("📝 Customer Details")
    st.markdown(
        "Fill in the customer information below, "
        "then click **Predict**."
    )

    # st.form() groups all inputs together.
    # The API is only called when the user clicks Submit.
    # Without form(), Streamlit re-runs on every input change.
    with st.form(key="customer_form"):

        st.markdown("**Demographics**")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Customer age in years (18–100)"
        )

        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Customer gender"
        )

        senior_citizen = st.selectbox(
            "Senior Citizen (65+)",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="1 if customer is 65 or older"
        )

        has_partner = st.selectbox(
            "Has Partner",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        has_dependents = st.selectbox(
            "Has Dependents",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        st.markdown("**Contract & Billing**")

        tenure_months = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            help="How many months the customer has been with us"
        )

        contract = st.selectbox(
            "Contract Type",
            options=[
                "Month-to-month",
                "One year",
                "Two year"
            ],
            help="Type of service contract"
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=79.50,
            step=0.50,
            format="%.2f"
        )

        # Auto-calculate total_charges as a suggestion
        # The user can adjust it
        suggested_total = round(tenure_months * monthly_charges, 2)
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=50000.0,
            value=float(suggested_total),
            step=1.0,
            format="%.2f",
            help="Total amount billed. Auto-calculated from "
                 "tenure × monthly charges but can be adjusted."
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=1
        )

        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )

        st.markdown("**Services**")

        internet_service = st.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            options=["Yes", "No"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            options=["Yes", "No"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            options=["Yes", "No"]
        )

        # The submit button — triggers form processing
        # st.form_submit_button() only works inside st.form()
        submitted = st.form_submit_button(
            label="🔮 Predict Churn Risk",
            use_container_width=True,
            type="primary"   # "primary" = blue/highlighted button
        )


# ── HANDLE FORM SUBMISSION ────────────────────────────────
with col_results:
    st.subheader("📊 Prediction Results")

    if not submitted:
        # Show placeholder before first submission
        st.info(
            "👈 Fill in the customer details on the left "
            "and click **Predict Churn Risk** to see results."
        )

        # Show a "demo" high-risk profile
        st.markdown("**Example high-risk profile:**")
        st.markdown(
            "- Tenure: 2 months\n"
            "- Contract: Month-to-month\n"
            "- Internet: Fiber optic\n"
            "- Tech Support: No\n"
            "- Monthly Charges: $95\n"
            "→ Typically produces HIGH risk score"
        )

    else:
        # Build the request payload from form values
        customer_payload = {
            "age"              : int(age),
            "gender"           : gender,
            "senior_citizen"   : int(senior_citizen),
            "has_partner"      : int(has_partner),
            "has_dependents"   : int(has_dependents),
            "tenure_months"    : int(tenure_months),
            "contract"         : contract,
            "paperless_billing": int(paperless_billing),
            "payment_method"   : payment_method,
            "internet_service" : internet_service,
            "online_security"  : online_security,
            "tech_support"     : tech_support,
            "streaming_tv"     : streaming_tv,
            "monthly_charges"  : float(monthly_charges),
            "total_charges"    : float(total_charges)
        }

        # Call the API
        with st.spinner("Calling prediction API..."):
            result = call_predict_api(customer_payload)

        if result:
            prob      = result['churn_probability']
            label     = result['churn_label']
            risk_tier = result['risk_tier']
            threshold = result['threshold_used']
            model_ver = result['model_version']

            # ── MAIN RESULT CARD ──────────────────────
            # Color the card based on risk level
            if risk_tier == "HIGH":
                card_color = "#fee2e2"
                text_color = "#991b1b"
                emoji      = "🔴"
            elif risk_tier == "MEDIUM":
                card_color = "#fef3c7"
                text_color = "#92400e"
                emoji      = "🟡"
            else:
                card_color = "#d1fae5"
                text_color = "#065f46"
                emoji      = "🟢"

            # Use HTML to create a colored result card
            st.markdown(
                f"""
                <div style="
                    background-color: {card_color};
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 16px;
                    text-align: center;
                ">
                    <h1 style="color:{text_color}; margin:0">
                        {emoji} {prob:.1%}
                    </h1>
                    <h3 style="color:{text_color}; margin:4px 0">
                        Churn Probability
                    </h3>
                    <p style="color:{text_color}; margin:0">
                        Risk Tier: <strong>{risk_tier}</strong> |
                        Decision: {'Will Churn' if label==1
                                   else 'Will Stay'}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── METRICS ROW ───────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Churn Probability",
                    f"{prob:.1%}"
                )
            with m2:
                st.metric(
                    "Decision Threshold",
                    f"{threshold:.2f}"
                )
            with m3:
                st.metric(
                    "Model Version",
                    model_ver
                )

            st.markdown("---")

            # ── SHAP WATERFALL CHART ──────────────────
            st.markdown("### 🔍 Why did the model predict this?")
            st.markdown(
                "Each bar below shows how much one feature "
                "**pushed** the prediction up or down."
            )

            draw_shap_waterfall(
                result['explanation'],
                prob
            )

            # ── TOP FEATURES TABLE ────────────────────
            st.markdown("### 📋 Feature Impact Details")

            features_data = result['explanation']['top_features']
            df_features = pd.DataFrame(features_data)

            # Add a human-readable impact column
            df_features['impact_pct'] = (
                df_features['shap_value'].abs() /
                df_features['shap_value'].abs().sum() * 100
            ).round(1)

            df_features = df_features.rename(columns={
                'feature'   : 'Feature',
                'shap_value': 'SHAP Value',
                'direction' : 'Effect',
                'impact_pct': 'Relative Impact (%)'
            })

            st.dataframe(
                df_features,
                use_container_width=True,
                hide_index=True
            )

            # ── BUSINESS RECOMMENDATION ──────────────
            st.markdown("### 💼 Recommended Action")

            if risk_tier == "HIGH":
                st.error(
                    "**Immediate intervention recommended.**\n\n"
                    "This customer has a high probability of "
                    "churning. Consider:\n"
                    "- Personal outreach from retention team\n"
                    "- Offer contract upgrade discount\n"
                    "- Add tech support if on Fiber optic\n"
                    "- Loyalty reward for staying"
                )
            elif risk_tier == "MEDIUM":
                st.warning(
                    "**Monitor closely.**\n\n"
                    "This customer shows moderate risk. Consider:\n"
                    "- Include in next retention campaign\n"
                    "- Send satisfaction survey\n"
                    "- Offer service upgrade trial"
                )
            else:
                st.success(
                    "**No immediate action needed.**\n\n"
                    "This customer appears stable. Focus "
                    "retention resources on higher-risk customers."
                )