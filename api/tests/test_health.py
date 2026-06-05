"""Health endpoint specific tests."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data


def test_health_uptime_positive():
    response = client.get("/health")
    data = response.json()
    assert data['uptime_seconds'] >= 0


def test_health_status_healthy():
    response = client.get("/health")
    data = response.json()
    assert data['status'] in ['healthy', 'degraded']


def test_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200