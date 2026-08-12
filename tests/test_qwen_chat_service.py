from pathlib import Path

from app.config import Settings
from app.database import ConversationRepository
from app.safety import SafetyService
from app.services import ChatService


class StubClassifier:
    def predict(self, text): return "sadness", 0.9


class StubLlm:
    def __init__(self, response=None, fails=False): self.response, self.fails, self.calls = response, fails, 0
    def generate(self, prompt):
        self.calls += 1
        if self.fails:
            raise RuntimeError("offline")
        return self.response


def service(tmp_path, llm):
    settings = Settings("test", tmp_path / "chat.db", Path("artifacts/models/final_model.joblib"), "emotion-sklearn-qwen3-4b-v1")
    return ChatService(settings, ConversationRepository(settings.database_path), StubClassifier(), SafetyService(), llm)


def test_successful_qwen_response_and_model_version_are_logged(tmp_path):
    app = service(tmp_path, StubLlm("That sounds difficult. What feels most important right now?"))
    result = app.chat("I feel low", "session-1")
    stored = app.repository.session_messages("session-1")[0]
    assert result["response_type"] == "ollama_generated"
    assert stored["model_version"] == "emotion-sklearn-qwen3-4b-v1"


def test_unavailable_qwen_uses_deterministic_fallback(tmp_path):
    llm = StubLlm(fails=True)
    result = service(tmp_path, llm).chat("I feel low", "session-2")
    assert result["response_type"] == "ollama_unavailable"
    assert result["response"]


def test_safety_override_never_calls_qwen(tmp_path):
    llm = StubLlm("This must not be used.")
    result = service(tmp_path, llm).chat("I want to die", "session-3")
    assert result["response_type"] == "safety_override"
    assert llm.calls == 0
