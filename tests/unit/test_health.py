from app.main import app
from fastapi.testclient import TestClient


def test_live_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "luoyang-ai-service"