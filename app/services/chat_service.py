from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings
from app.database import ConversationRepository
from app.nlp import EmotionClassifier
from app.safety import RiskLevel, SafetyService


RESPONSES = {
    "joy": "I’m glad you shared that. What feels meaningful about this moment?",
    "sadness": "That sounds heavy. If you’d like, you can tell me a little more about what’s been weighing on you.",
    "anger": "It makes sense that this feels frustrating. What happened?",
    "fear": "That sounds unsettling. Taking one small step at a time can help—what feels most immediate?",
    "surprise": "That sounds unexpected. How are you making sense of it?",
    "disgust": "That sounds like a strong reaction to something unpleasant. I’m listening.",
    "neutral": "Thanks for checking in. What would you like to talk through?",
}
FALLBACK = "I may not be understanding the feeling clearly. Could you share a little more about what is happening?"


def detect_intent(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("help", "advice", "what should i do")):
        return "support_request"
    if "?" in text or lower.startswith(("how", "what", "why", "can you")):
        return "question"
    if any(token in lower for token in ("thank", "thanks")):
        return "gratitude"
    return "check_in"


class ChatService:
    def __init__(self, settings: Settings, repository: ConversationRepository, classifier: EmotionClassifier, safety: SafetyService):
        self.settings, self.repository, self.classifier, self.safety = settings, repository, classifier, safety

    def chat(self, message: str, session_id: str | None = None) -> dict:
        started = time.perf_counter()
        session_id = session_id or str(uuid4())
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        message = message.strip()
        if len(message) > self.settings.max_input_length:
            raise ValueError(f"message must not exceed {self.settings.max_input_length} characters")

        safety = self.safety.assess(message)
        intent = detect_intent(message)
        emotion, confidence = self.classifier.predict(message)
        if safety.level is not RiskLevel.NORMAL:
            response, response_type = self.safety.response(safety.level), "safety_override"
        elif confidence < self.settings.low_confidence_threshold:
            response, response_type = FALLBACK, "low_confidence_fallback"
        elif confidence < self.settings.medium_confidence_threshold:
            response, response_type = RESPONSES.get(emotion, FALLBACK) + " I may be reading this imperfectly.", "cautious_response"
        else:
            response, response_type = RESPONSES.get(emotion, FALLBACK), "emotion_response"
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        message_id = self.repository.add_message({"session_id": session_id, "message": message, "response": response, "timestamp": datetime.now(timezone.utc).isoformat(), "predicted_emotion": emotion, "confidence": confidence, "intent": intent, "risk_level": safety.level.value, "response_type": response_type, "model_version": self.settings.model_version, "latency_ms": latency_ms})
        return {"message_id": message_id, "session_id": session_id, "response": response, "emotion": emotion, "confidence": round(confidence, 4), "intent": intent, "risk_level": safety.level.value, "response_type": response_type, "model_version": self.settings.model_version, "latency_ms": latency_ms}
