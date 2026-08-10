import pytest

from app.safety import RiskLevel, SafetyService


@pytest.mark.parametrize(("text", "level"), [
    ("I feel overwhelmed and hopeless", RiskLevel.DISTRESS),
    ("I want to die", RiskLevel.CRISIS),
    ("I am going to kill myself tonight", RiskLevel.EMERGENCY),
    ("I am happy to finish my project", RiskLevel.NORMAL),
])
def test_safety_routing(text, level):
    assert SafetyService().assess(text).level is level


def test_crisis_response_is_supportive_not_diagnostic():
    response = SafetyService.response(RiskLevel.CRISIS).lower()
    assert "emergency" in response and "diagnos" not in response
