from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    secret_key: str
    database_path: Path
    model_path: Path
    model_version: str
    max_input_length: int = 500
    low_confidence_threshold: float = 0.45
    medium_confidence_threshold: float = 0.70
    retention_days: int = 30
    llm_provider: str = "none"
    ollama_model: str = "my-qwen-chatbot"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 60.0
    memory_message_limit: int = 6
    emotion_provider: str = "sklearn"
    hf_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            secret_key=os.environ.get("FLASK_SECRET_KEY", "development-only-change-me"),
            database_path=Path(os.environ.get("DATABASE_PATH", ROOT_DIR / "data" / "mentalcare.db")),
            model_path=Path(os.environ.get("MODEL_PATH", ROOT_DIR / "artifacts" / "models" / "final_model.joblib")),
            model_version=os.environ.get("MODEL_VERSION", "emotion-sklearn-qwen3-4b-v1"),
            max_input_length=int(os.environ.get("MAX_INPUT_LENGTH", "500")),
            low_confidence_threshold=float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.45")),
            medium_confidence_threshold=float(os.environ.get("MEDIUM_CONFIDENCE_THRESHOLD", "0.70")),
            retention_days=int(os.environ.get("RETENTION_DAYS", "30")),
            llm_provider=os.environ.get("LLM_PROVIDER", "none"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "my-qwen-chatbot"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_timeout_seconds=float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60")),
            memory_message_limit=int(os.environ.get("MEMORY_MESSAGE_LIMIT", "6")),
            emotion_provider=os.environ.get("EMOTION_PROVIDER", "sklearn"),
            hf_emotion_model=os.environ.get("HF_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"),
        )
