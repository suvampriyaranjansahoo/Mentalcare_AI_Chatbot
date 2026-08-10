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

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            secret_key=os.environ.get("FLASK_SECRET_KEY", "development-only-change-me"),
            database_path=Path(os.environ.get("DATABASE_PATH", ROOT_DIR / "data" / "mentalcare.db")),
            model_path=Path(os.environ.get("MODEL_PATH", ROOT_DIR / "artifacts" / "models" / "final_model.joblib")),
            model_version=os.environ.get("MODEL_VERSION", "tfidf-linear-svm-v1"),
            max_input_length=int(os.environ.get("MAX_INPUT_LENGTH", "500")),
            low_confidence_threshold=float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.45")),
            medium_confidence_threshold=float(os.environ.get("MEDIUM_CONFIDENCE_THRESHOLD", "0.70")),
            retention_days=int(os.environ.get("RETENTION_DAYS", "30")),
        )
