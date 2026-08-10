import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "flask_app"))

from pipeline import get_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    return get_pipeline()


def test_classifier_returns_valid_label(pipeline):
    emotion, confidence = pipeline.classify_emotion("I feel really anxious about this")
    assert emotion in pipeline.labels


def test_classifier_confidence_in_valid_range(pipeline):
    _, confidence = pipeline.classify_emotion("I feel great today")
    assert confidence is None or 0.0 <= confidence <= 1.0


def test_classifier_handles_short_input(pipeline):
    emotion, _ = pipeline.classify_emotion("sad")
    assert emotion in pipeline.labels


def test_classifier_handles_long_input(pipeline):
    long_text = "I feel really anxious and overwhelmed " * 20
    emotion, _ = pipeline.classify_emotion(long_text)
    assert emotion in pipeline.labels
