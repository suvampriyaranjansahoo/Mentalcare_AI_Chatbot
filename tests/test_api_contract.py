from pathlib import Path

import pytest

from app.api import create_app
from app.config import Settings


@pytest.fixture
def client(tmp_path):
    root = Path(__file__).resolve().parents[1]
    settings = Settings("test", tmp_path / "test.db", root / "artifacts/models/final_model.joblib", "test")
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def test_model_info(client):
    assert client.get("/model-info").status_code == 200


def test_chat_validation_contract(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "validation_error"


def test_crisis_message_is_safety_override(client):
    response = client.post("/chat", json={"message": "I want to die"})
    data = response.get_json()
    assert response.status_code == 200
    assert data["risk_level"] == "potential_self_harm"
    assert data["response_type"] == "safety_override"


def test_feedback_and_session_analytics(client):
    message = client.post("/chat", json={"message": "I feel happy today"}).get_json()
    assert client.post("/feedback", json={"message_id": message["message_id"], "helpful": True}).status_code == 201
    detail = client.get(f"/sessions/{message['session_id']}").get_json()
    assert detail["analytics"]["conversation_length"] == 1
