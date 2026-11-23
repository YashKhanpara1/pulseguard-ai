import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_ingest_anomaly_detection(client):
    payload = {
        "service_name": "test-service",
        "endpoint": "/api/v1/pay",
        "latency_ms": 1450.0,
        "status_code": 503,
        "payload_kb": 180.0,
        "cpu_usage_pct": 94.0,
    }
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["event"]["is_anomaly"] is True
    assert body["event"]["risk_score"] > 0.4