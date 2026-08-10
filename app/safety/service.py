from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    NORMAL = "normal"
    DISTRESS = "emotional_distress"
    CRISIS = "potential_self_harm"
    EMERGENCY = "emergency_high_risk"


@dataclass(frozen=True)
class SafetyResult:
    level: RiskLevel
    matched_signals: tuple[str, ...]


class SafetyService:
    """Conservative rule layer. It routes risk; it never diagnoses a user."""

    EMERGENCY = (
        r"\b(i am|i'm|im) (going to|about to) (kill|end) myself\b",
        r"\b(i have|with) (a |the )?(gun|weapon|pills|rope)\b",
        r"\b(suicide plan|planned my suicide|goodbye everyone)\b",
    )
    CRISIS = (
        r"\b(kill myself|end my life|want to die|don't want to live|suicidal)\b",
        r"\b(hurt myself|self[- ]harm|cut myself)\b",
    )
    DISTRESS = (
        r"\b(overwhelmed|hopeless|worthless|can't cope|cannot cope|panic attack|numb)\b",
        r"\b(feel(?:ing)? (?:so )?(?:low|empty|alone|scared))\b",
    )

    @staticmethod
    def _matches(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))

    def assess(self, text: str) -> SafetyResult:
        for level, patterns in ((RiskLevel.EMERGENCY, self.EMERGENCY), (RiskLevel.CRISIS, self.CRISIS), (RiskLevel.DISTRESS, self.DISTRESS)):
            signals = self._matches(text, patterns)
            if signals:
                return SafetyResult(level, signals)
        return SafetyResult(RiskLevel.NORMAL, ())

    @staticmethod
    def response(level: RiskLevel) -> str:
        if level in (RiskLevel.EMERGENCY, RiskLevel.CRISIS):
            return ("I’m really sorry you’re dealing with this. I can’t provide emergency help, but your safety matters right now. "
                    "Please contact local emergency services or a crisis line in your country now, and if you can, reach out to someone you trust to stay with you. "
                    "If you are in the U.S. or Canada, call or text 988; elsewhere, use your local emergency number or find a local crisis service.")
        if level is RiskLevel.DISTRESS:
            return ("That sounds really difficult. I’m here to listen, but I’m not a mental-health professional. "
                    "If these feelings are becoming hard to manage, consider contacting a licensed professional or someone you trust.")
        raise ValueError("Normal messages do not use a safety override response")
