import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "flask_app"))

from pipeline import get_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    return get_pipeline()


def test_exact_faq_match_returns_high_similarity(pipeline):
    # Use an actual FAQ bank entry verbatim — should match itself with very high similarity.
    known_text = pipeline.faq_texts[0]
    result = pipeline.faq_lookup(known_text)
    assert result["matched"] is True
    assert result["score"] > 0.9


def test_unrelated_text_has_lower_similarity(pipeline):
    gibberish = "zzz qqq flibbertigibbet unrelated nonsense words"
    result = pipeline.faq_lookup(gibberish)
    # Should score meaningfully lower than an exact match, though may still
    # cross threshold on a small-vocabulary dataset — we only assert relative ordering.
    known_text = pipeline.faq_texts[0]
    known_result = pipeline.faq_lookup(known_text)
    assert result["score"] < known_result["score"]


def test_faq_lookup_returns_expected_keys(pipeline):
    result = pipeline.faq_lookup("I feel happy today")
    assert "matched" in result
    assert "score" in result
