from __future__ import annotations
from collections import Counter


def session_analytics(messages: list[dict]) -> dict:
    emotions = [m["predicted_emotion"] for m in messages if m.get("predicted_emotion")]
    transitions = Counter(zip(emotions, emotions[1:]))
    confidences = [m["confidence"] for m in messages if m.get("confidence") is not None]
    return {"session_id": messages[0]["session_id"] if messages else None, "conversation_length": len(messages), "dominant_emotion": Counter(emotions).most_common(1)[0][0] if emotions else None, "emotion_distribution": dict(Counter(emotions)), "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None, "uncertainty_rate": round(sum(c < 0.45 for c in confidences) / len(confidences), 4) if confidences else 0.0, "emotion_transitions": [{"from": a, "to": b, "count": count} for (a, b), count in transitions.items()], "mood_timeline": [{"timestamp": m["timestamp"], "emotion": m["predicted_emotion"], "confidence": m["confidence"]} for m in messages]}
