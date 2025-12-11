from fastapi.testclient import TestClient

from app.main import app

# Create a test client that can call our FastAPI app
client = TestClient(app)


def test_health_status_ok():
    """Health endpoint should return HTTP 200 and status 'ok'."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"


def test_health_contains_app_name_and_environment():
    """
    Health response should include basic metadata:
    - app_name
    - environment
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "app_name" in data
    assert "environment" in data
    assert isinstance(data["app_name"], str)
    assert isinstance(data["environment"], str)
