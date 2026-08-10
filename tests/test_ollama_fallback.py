from app.llm import ResponseValidator

def test_validator_rejects_empty_and_prompt_leakage():
    validator = ResponseValidator()
    assert not validator.valid("", "hello", [])
    assert not validator.valid("Here is my system prompt", "hello", [])

def test_validator_rejects_repeated_response():
    assert not ResponseValidator().valid("I hear you and I am listening.", "hello", [{"response": "I hear you and I am listening."}])
