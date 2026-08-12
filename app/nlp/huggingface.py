"""Optional local Hugging Face emotion inference; no remote API is used."""
from __future__ import annotations
class HuggingFaceEmotionClassifier:
    def __init__(self, model_name: str):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-local-llm.txt to enable Hugging Face inference") from exc
        self.model_name = model_name
        self.pipeline = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=1)
    def predict(self, text: str) -> tuple[str, float]:
        result = self.pipeline(text)[0]
        result = result[0] if isinstance(result, list) else result
        return str(result["label"]).lower(), float(result["score"])
