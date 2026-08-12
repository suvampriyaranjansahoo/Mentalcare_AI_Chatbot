"""Small stdlib-only client for the local Ollama HTTP API."""
from __future__ import annotations

import json
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """Qwen reasoning must never be surfaced to an end user."""
    return THINK_BLOCK.sub("", text).strip()


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7, "num_predict": 180, "num_ctx": 512},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Ollama response content is not text")
            return strip_thinking(content)
        except (HTTPError, URLError, socket.timeout, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama is unavailable or returned an invalid response") from exc


class ResponseValidator:
    banned = (
        "<think", "</think", "system prompt", "ignore previous instructions",
        "chain of thought", "internal reasoning", "as a therapist", "as a medical professional",
        "i diagnose", "you have a disorder", "you are diagnosed", "kill yourself",
    )

    def valid(self, text: str, user_message: str, history: list[dict]) -> bool:
        lower = text.lower().strip()
        if not 12 <= len(text) <= 900 or any(term in lower for term in self.banned):
            return False
        if text.strip().lower() == user_message.strip().lower():
            return False
        return not any(lower == str(row.get("response", "")).lower() for row in history)
