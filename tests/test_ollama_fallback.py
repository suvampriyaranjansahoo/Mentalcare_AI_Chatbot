import json
import socket
from unittest.mock import patch

import pytest

from app.llm.ollama import OllamaClient, ResponseValidator, strip_thinking


class FakeHttpResponse:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode("utf-8")
    def __enter__(self): return self
    def __exit__(self, *args): return False


def test_qwen_chat_request_uses_configured_model_and_hides_thinking():
    client = OllamaClient("http://localhost:11434", "my-qwen-chatbot", 60)
    reply = {"message": {"content": "<think>private reasoning</think>Hello, I'm here with you."}}
    with patch("app.llm.ollama.urlopen", return_value=FakeHttpResponse(reply)) as mocked:
        assert client.generate("hello") == "Hello, I'm here with you."
    request = mocked.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["model"] == "my-qwen-chatbot"
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"] == {"temperature": 0.7, "num_predict": 180, "num_ctx": 512}


@pytest.mark.parametrize("error", [socket.timeout(), ValueError("bad json")])
def test_ollama_errors_become_safe_runtime_error(error):
    with patch("app.llm.ollama.urlopen", side_effect=error):
        with pytest.raises(RuntimeError):
            OllamaClient("http://localhost:11434", "my-qwen-chatbot", 60).generate("hello")


def test_invalid_ollama_payload_is_rejected():
    with patch("app.llm.ollama.urlopen", return_value=FakeHttpResponse({"message": {}})):
        with pytest.raises(RuntimeError):
            OllamaClient("http://localhost:11434", "my-qwen-chatbot", 60).generate("hello")


def test_validator_rejects_internal_content_and_repetition():
    validator = ResponseValidator()
    assert not validator.valid("Here is my system prompt", "hello", [])
    assert not validator.valid("<think>reasoning</think> Final reply.", "hello", [])
    assert not validator.valid("I hear you and I am listening.", "hello", [{"response": "I hear you and I am listening."}])


def test_strip_thinking_removes_only_private_block():
    assert strip_thinking("<think>hidden</think>Visible response") == "Visible response"
