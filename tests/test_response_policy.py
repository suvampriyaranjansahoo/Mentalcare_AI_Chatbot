from app.services.chat_service import detect_intent
from app.safety import RiskLevel, SafetyService

def test_food_request_has_practical_need_intent():
    assert detect_intent("Can you bring some food?") == "practical_need"

def test_bad_mood_is_distress_signal():
    assert SafetyService().assess("I am in a bad mood").level is RiskLevel.DISTRESS
