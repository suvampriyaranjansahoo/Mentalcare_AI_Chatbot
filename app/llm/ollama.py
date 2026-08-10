from __future__ import annotations
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float): self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.7, "num_predict": 180}}).encode()
        try:
            with urlopen(Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=self.timeout) as response:
                return str(json.loads(response.read())["response"]).strip()
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama is unavailable or returned an invalid response") from exc

class ResponseValidator:
    banned = ("system prompt", "ignore previous", "as a therapist", "i diagnose", "you have a disorder", "kill yourself")
    def valid(self, text: str, user_message: str, history: list[dict]) -> bool:
        lower = text.lower().strip()
        if not 12 <= len(text) <= 900 or any(term in lower for term in self.banned): return False
        if text.strip().lower() == user_message.strip().lower(): return False
        return not any(lower == str(row.get("response", "")).lower() for row in history)
