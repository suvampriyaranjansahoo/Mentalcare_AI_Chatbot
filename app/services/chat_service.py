from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings
from app.database import ConversationRepository
from app.nlp import EmotionClassifier
from app.safety import RiskLevel, SafetyService

RESPONSES = {
    "joy": ("I'm glad you shared that. What feels meaningful about this moment?", "That is good to hear. What has helped today feel a little lighter?"),
    "sadness": ("That sounds heavy. You do not have to carry it alone right now. What has been weighing on you most?", "I'm sorry today feels like this. If it helps, we can take it one small piece at a time."),
    "anger": ("It makes sense that this feels frustrating. What happened?", "That sounds upsetting. Would you like to tell me what set this off?"),
    "fear": ("That sounds unsettling. You are here in this moment; what feels most immediate right now?", "Feeling on edge can be exhausting. Would a slow breath and one small next step feel possible?"),
    "surprise": ("That sounds unexpected. How are you making sense of it?", "That is a lot to take in at once. What part is staying with you?"),
    "disgust": ("That sounds like a strong reaction to something unpleasant. I'm listening.", "It makes sense to be put off by that. Do you want to say more about what happened?"),
    "neutral": ("Thanks for checking in. What would you like to talk through?", "I'm here with you. What has your day been like so far?"),
}
LOW_CONFIDENCE_RESPONSES = (
    "I'm not fully sure how to read that, but I hear that something may be off. What would feel most helpful right now: being heard, a small distraction, or thinking through a next step?",
    "Thank you for saying that. I may not have the emotion exactly right, so I don't want to assume. Can you tell me whether today feels more tiring, upsetting, or simply difficult?",
)
PRACTICAL_RESPONSES = (
    "I wish I could bring you food. I can't do that directly, but if you can, a glass of water and something small to eat may help you feel a little steadier. Is there anyone nearby you could ask?",
    "I can't bring food, but I can stay with you while you decide on one easy option. Is there something simple nearby, or someone you could message for help?",
)

def choose(options: tuple[str, ...], session_id: str, message: str) -> str:
    index = int(hashlib.sha256(f"{session_id}:{message}".encode()).hexdigest(), 16) % len(options)
    return options[index]

def detect_intent(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("food", "hungry", "eat", "eating", "meal", "drink", "water")):
        return "practical_need"
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
        started = time.perf_counter(); session_id = session_id or str(uuid4())
        if not isinstance(message, str) or not message.strip(): raise ValueError("message must be a non-empty string")
        message = message.strip()
        if len(message) > self.settings.max_input_length: raise ValueError(f"message must not exceed {self.settings.max_input_length} characters")
        safety, intent = self.safety.assess(message), detect_intent(message)
        emotion, confidence = self.classifier.predict(message)
        if safety.level is not RiskLevel.NORMAL: response, response_type = self.safety.response(safety.level), "safety_override"
        elif intent == "practical_need": response, response_type = choose(PRACTICAL_RESPONSES, session_id, message), "practical_support"
        elif confidence < self.settings.low_confidence_threshold: response, response_type = choose(LOW_CONFIDENCE_RESPONSES, session_id, message), "low_confidence_fallback"
        elif confidence < self.settings.medium_confidence_threshold: response, response_type = choose(RESPONSES.get(emotion, LOW_CONFIDENCE_RESPONSES), session_id, message) + " I may be reading this imperfectly.", "cautious_response"
        else: response, response_type = choose(RESPONSES.get(emotion, LOW_CONFIDENCE_RESPONSES), session_id, message), "emotion_response"
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        message_id = self.repository.add_message({"session_id": session_id, "message": message, "response": response, "timestamp": datetime.now(timezone.utc).isoformat(), "predicted_emotion": emotion, "confidence": confidence, "intent": intent, "risk_level": safety.level.value, "response_type": response_type, "model_version": self.settings.model_version, "latency_ms": latency_ms})
        return {"message_id": message_id, "session_id": session_id, "response": response, "emotion": emotion, "confidence": round(confidence, 4), "intent": intent, "risk_level": safety.level.value, "response_type": response_type, "model_version": self.settings.model_version, "latency_ms": latency_ms}
