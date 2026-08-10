import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import create_app  # noqa: E402
from app.config import Settings  # noqa: E402


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        secret_key="test-secret",
        database_path=tmp_path / "test.db",
        model_path=os.path.join(BASE_DIR, "artifacts", "models", "final_model.joblib"),
        model_version="test",
    )
    app = create_app(settings)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_chat_route_valid_input(client):
    response = client.post("/chat", json={"message": "I feel happy today"})
    assert response.status_code == 200
    body = response.get_json()
    assert "response" in body
    assert isinstance(body["response"], str)
    assert len(body["response"]) > 0


def test_chat_route_empty_input(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422
    assert "empty" in response.get_json()["error"]["message"].lower()


def test_chat_route_missing_message_key(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_route_malformed_json(client):
    response = client.post(
        "/chat", data="not json", content_type="application/json"
    )
    assert response.status_code == 400


def test_chat_route_overly_long_input(client):
    long_msg = "a" * 1000
    response = client.post("/chat", json={"message": long_msg})
    assert response.status_code == 422
    assert "exceed" in response.get_json()["error"]["message"].lower()
