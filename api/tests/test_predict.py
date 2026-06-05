"""
API tests using FastAPI TestClient.
Run with: pytest api/tests/ -v
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)
API_KEY = "dev_key_change_in_production"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

VALID_CUSTOMER = {
    "age"              : 32,
    "gender"           : "Female",
    "senior_citizen"   : 0,
    "has_partner"      : 1,
    "has_dependents"   : 0,
    "tenure_months"    : 5,
    "contract"         : "Month-to-month",
    "paperless_billing": 1,
    "payment_method"   : "Electronic check",
    "internet_service" : "Fiber optic",
    "online_security"  : "No",
    "tech_support"     : "No",
    "streaming_tv"     : "Yes",
    "monthly_charges"  : 89.50,
    "total_charges"    : 447.50
}


class TestHealth:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_health_no_auth_required(self):
        # Health should work without API key
        response = client.get("/health")
        assert response.status_code != 403


class TestPredict:

    def test_predict_valid_input(self):
        response = client.post(
            "/predict",
            json=VALID_CUSTOMER,
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data['churn_probability'] <= 1.0
        assert data['churn_label'] in [0, 1]
        assert data['risk_tier'] in ['HIGH', 'MEDIUM', 'LOW']
        assert 'explanation' in data

    def test_predict_explanation_structure(self):
        response = client.post(
            "/predict",
            json=VALID_CUSTOMER,
            headers=HEADERS
        )
        data = response.json()
        explanation = data['explanation']
        assert 'base_value' in explanation
        assert 'top_features' in explanation
        assert len(explanation['top_features']) > 0

        feature = explanation['top_features'][0]
        assert 'feature' in feature
        assert 'shap_value' in feature
        assert 'direction' in feature

    def test_predict_no_api_key_returns_403(self):
        response = client.post(
            "/predict",
            json=VALID_CUSTOMER
        )
        assert response.status_code == 403

    def test_predict_invalid_api_key_returns_403(self):
        response = client.post(
            "/predict",
            json=VALID_CUSTOMER,
            headers={"Authorization": "Bearer wrong_key"}
        )
        assert response.status_code == 403

    def test_predict_missing_field_returns_422(self):
        incomplete = VALID_CUSTOMER.copy()
        del incomplete['tenure_months']
        response = client.post(
            "/predict",
            json=incomplete,
            headers=HEADERS
        )
        assert response.status_code == 422

    def test_predict_invalid_age_returns_422(self):
        bad_data = VALID_CUSTOMER.copy()
        bad_data['age'] = 150   # > 100 — invalid
        response = client.post(
            "/predict",
            json=bad_data,
            headers=HEADERS
        )
        assert response.status_code == 422

    def test_predict_invalid_contract_returns_422(self):
        bad_data = VALID_CUSTOMER.copy()
        bad_data['contract'] = "Weekly"   # not a valid enum
        response = client.post(
            "/predict",
            json=bad_data,
            headers=HEADERS
        )
        assert response.status_code == 422

    def test_predict_high_risk_profile(self):
        """New customer + month-to-month + fiber + no support
        should produce HIGH risk tier."""
        high_risk = VALID_CUSTOMER.copy()
        high_risk['tenure_months']  = 2
        high_risk['contract']       = "Month-to-month"
        high_risk['internet_service'] = "Fiber optic"
        high_risk['tech_support']   = "No"
        high_risk['total_charges']  = 179.0

        response = client.post(
            "/predict",
            json=high_risk,
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        # High risk profile should produce elevated probability
        assert data['churn_probability'] > 0.4

    def test_predict_low_risk_profile(self):
        """Long tenure + two year contract should produce
        LOW or MEDIUM risk."""
        low_risk = VALID_CUSTOMER.copy()
        low_risk['tenure_months']   = 60
        low_risk['contract']        = "Two year"
        low_risk['internet_service'] = "DSL"
        low_risk['tech_support']    = "Yes"
        low_risk['monthly_charges'] = 45.0
        low_risk['total_charges']   = 2700.0

        response = client.post(
            "/predict",
            json=low_risk,
            headers=HEADERS
        )
        assert response.status_code == 200
        data = response.json()
        assert data['churn_probability'] < 0.6


class TestMetrics:

    def test_metrics_returns_200(self):
        response = client.get("/metrics", headers=HEADERS)
        assert response.status_code == 200

    def test_metrics_has_performance_fields(self):
        response = client.get("/metrics", headers=HEADERS)
        data = response.json()
        assert 'performance' in data
        assert 'auc_roc' in data['performance']
        assert 'f1_score' in data['performance']

    def test_metrics_requires_auth(self):
        response = client.get("/metrics")
        assert response.status_code == 403