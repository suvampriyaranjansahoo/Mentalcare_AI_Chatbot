"""
Phase 9 — Hybrid ML Pipeline

USER INPUT
   |
Input validation
   |
FAQ similarity (TF-IDF cosine, threshold chosen empirically in Phase 8)
   |
High-confidence match?
   |--- YES --> FAQ response
   |--- NO  --> Emotion classifier (TF-IDF + Linear SVM, from Phase 5)
                   |
              Response generation (emotion-appropriate template bank)
                   |
              Quality gate (non-empty, not the input echoed back)
                   |
              Safe fallback if gate fails

Every request is tracked with: session_id, timestamp, input length,
FAQ similarity score, predicted emotion, classifier confidence (if available),
response type, latency. This feeds Phase 10's analytics.
"""

import os
import time
import uuid
import json
import random
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "final_model.joblib")
FAQ_DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
FAQ_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "faq_results.json")
LOG_PATH = os.path.join(BASE_DIR, "flask_app", "logs", "conversations.csv")

# Load the empirically chosen FAQ threshold from Phase 8, falling back to a
# documented default if that experiment hasn't been run yet.
DEFAULT_FAQ_THRESHOLD = 0.6
try:
    with open(FAQ_RESULTS_PATH) as f:
        faq_experiment = json.load(f)
    FAQ_THRESHOLD = faq_experiment["chosen_thresholds"]["tfidf_cosine_similarity"]["threshold"]
except (FileNotFoundError, KeyError, TypeError):
    FAQ_THRESHOLD = DEFAULT_FAQ_THRESHOLD

MAX_INPUT_LENGTH = 500

EMOTION_RESPONSE_BANK = {
    "joy": [
        "That's wonderful to hear! What made today feel good?",
        "I'm glad you're feeling this way. Hold onto that.",
    ],
    "sadness": [
        "I hear you, and it's okay to feel this way. I'm here with you.",
        "That sounds heavy. Do you want to tell me more about it?",
    ],
    "anger": [
        "That sounds really frustrating. Your reaction makes sense.",
        "It's okay to feel angry about this. Want to talk through what happened?",
    ],
    "fear": [
        "That sounds unsettling. Let's slow down for a moment together.",
        "It's okay to feel uneasy. What's on your mind right now?",
    ],
    "surprise": [
        "That sounds unexpected! How are you processing it?",
        "Wow, that's a lot to take in suddenly. How do you feel about it?",
    ],
    "disgust": [
        "That sounds like a strong reaction to something unpleasant. I hear you.",
        "Understandable — that kind of thing can be hard to shake off.",
    ],
    "neutral": [
        "Thanks for sharing that. What's on your mind?",
        "I'm listening — feel free to tell me more.",
    ],
}

FALLBACK_RESPONSE = "I'm here and listening. Could you tell me a bit more about how you're feeling?"


class HybridPipeline:
    def __init__(self):
        artifact = joblib.load(MODEL_PATH)
        self.vectorizer = artifact["vectorizer"]
        self.classifier = artifact["model"]
        self.labels = artifact["labels"]

        faq_df = pd.read_csv(FAQ_DATA_PATH)
        self.faq_texts = faq_df["user_input"].astype(str).tolist()
        self.faq_labels = faq_df["emotion_label"].tolist()
        self.faq_responses = faq_df["bot_response"].astype(str).tolist()

        self.faq_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.faq_matrix = self.faq_vectorizer.fit_transform(self.faq_texts)

        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # -----------------------------------------------------------
    def validate_input(self, text):
        if text is None:
            return False, "Message cannot be empty."
        text = text.strip()
        if len(text) == 0:
            return False, "Message cannot be empty."
        if len(text) > MAX_INPUT_LENGTH:
            return False, f"Message too long (max {MAX_INPUT_LENGTH} characters)."
        return True, text

    # -----------------------------------------------------------
    def faq_lookup(self, text):
        query_vec = self.faq_vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.faq_matrix)[0]
        best_idx = sims.argmax()
        best_score = float(sims[best_idx])
        if best_score >= FAQ_THRESHOLD:
            return {
                "matched": True,
                "score": round(best_score, 4),
                "response": self.faq_responses[best_idx],
                "matched_emotion": self.faq_labels[best_idx],
            }
        return {"matched": False, "score": round(best_score, 4)}

    # -----------------------------------------------------------
    def classify_emotion(self, text):
        vec = self.vectorizer.transform([text])
        pred = self.classifier.predict(vec)[0]
        confidence = None
        if hasattr(self.classifier, "decision_function"):
            # LinearSVC has decision_function, not predict_proba by default.
            scores = self.classifier.decision_function(vec)[0]
            # Convert margin to a rough 0-1 confidence via softmax for reporting only.
            import numpy as np
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            confidence = round(float(np.max(probs)), 4)
        return pred, confidence

    # -----------------------------------------------------------
    def generate_response(self, emotion):
        bank = EMOTION_RESPONSE_BANK.get(emotion, [FALLBACK_RESPONSE])
        return random.choice(bank)

    # -----------------------------------------------------------
    def quality_gate(self, response, user_text):
        if not response or not response.strip():
            return False
        if response.strip().lower() == user_text.strip().lower():
            return False
        return True

    # -----------------------------------------------------------
    def handle_message(self, user_text, session_id=None):
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        valid, cleaned_or_error = self.validate_input(user_text)
        if not valid:
            latency = time.perf_counter() - start
            self._log(session_id, user_text or "", None, None, None, "validation_error", latency)
            return {"response": cleaned_or_error, "type": "validation_error"}

        text = cleaned_or_error
        faq_result = self.faq_lookup(text)

        if faq_result["matched"]:
            response = faq_result["response"]
            response_type = "faq_match"
            emotion = faq_result["matched_emotion"]
            confidence = faq_result["score"]
        else:
            emotion, confidence = self.classify_emotion(text)
            response = self.generate_response(emotion)
            response_type = "classifier_generated"

            if not self.quality_gate(response, text):
                response = FALLBACK_RESPONSE
                response_type = "fallback"

        latency = time.perf_counter() - start
        self._log(session_id, text, faq_result.get("score"), emotion, confidence, response_type, latency)

        return {
            "response": response,
            "type": response_type,
            "emotion": emotion,
            "faq_similarity": faq_result.get("score"),
            "confidence": confidence,
            "latency_seconds": round(latency, 4),
        }

    # -----------------------------------------------------------
    def _log(self, session_id, text, faq_score, emotion, confidence, response_type, latency):
        row = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_length": len(text),
            "faq_similarity_score": faq_score,
            "predicted_emotion": emotion,
            "classifier_confidence": confidence,
            "response_type": response_type,
            "latency_seconds": round(latency, 4),
        }
        file_exists = os.path.isfile(LOG_PATH)
        pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)


_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HybridPipeline()
    return _pipeline_instance
