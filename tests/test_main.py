from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "CI/CD test success!"}


def test_gpu_status():
    response = client.get("/gpu")
    assert response.status_code == 200
    assert "cuda_available" in response.json()


def test_chromadb_status():
    response = client.get("/chromadb")
    assert response.status_code == 200
