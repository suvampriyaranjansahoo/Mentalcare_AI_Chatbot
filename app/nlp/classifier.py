from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np


class EmotionClassifier:
    """Inference-only wrapper. Training lives in scripts/, preventing leakage."""
    def __init__(self, artifact_path: Path):
        artifact = joblib.load(artifact_path)
        self.vectorizer = artifact["vectorizer"]
        self.model = artifact["model"]
        self.labels = tuple(artifact["labels"])

    def predict(self, text: str) -> tuple[str, float]:
        matrix = self.vectorizer.transform([text])
        label = str(self.model.predict(matrix)[0])
        if hasattr(self.model, "predict_proba"):
            return label, float(np.max(self.model.predict_proba(matrix)[0]))
        scores = self.model.decision_function(matrix)[0]
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        # Margin softmax is an uncertainty heuristic, not calibrated probability.
        return label, float(np.max(probabilities))
