"""
JSON Schema Validator for Customer Support Message Parsing
==========================================================


Validates that LLM output is strict, machine-parseable JSON conforming to:
{
    "intent": "refund|replacement|technical_support|shipping|other",
    "product": "string|null",
    "urgency": "low|medium|high",
    "needs_human": true|false,
    "summary": "string (5-25 words)"
}
"""

import json
from typing import Dict, Optional


# Allowed values for enum-like fields
VALID_INTENTS = {"refund", "replacement", "technical_support", "shipping", "other"}
VALID_URGENCIES = {"low", "medium", "high"}
REQUIRED_KEYS = {"intent", "product", "urgency", "needs_human", "summary"}


class ValidationResult:
    """Result of JSON validation with detailed error information."""

    def __init__(self, valid: bool, error: Optional[str] = None, parsed_json: Optional[Dict] = None):
        '''
         when validate_json_output returns, it will always return a ValidationResult object. 
         The .valid attribute will be True if the JSON is valid, and False if not. 
         The .error attribute will contain a string describing the error if valid is False, or None 
         if valid is True. The .parsed_json attribute will contain the parsed JSON dictionary if
         valid is True, or None if valid is False.
        '''
        self.valid = valid
        self.error = error
        self.parsed_json = parsed_json

    def __bool__(self):
        #  Lets you use if result: instead of if result.valid (from the caller side)
        return self.valid


def validate_json_output(output: str) -> ValidationResult:
    """
    Strict 8-step validator for JSON schema compliance.

    The ENTIRE output must be valid JSON — no extra text before or after.

    Checks:
        1. Entire output parseable as JSON (no trailing text allowed)
        2. Is a JSON object (dict)
        3. Has exactly the required keys (no extras, no missing)
        4. intent is from the allowed set
        5. urgency is from the allowed set
        6. needs_human is a boolean
        7. product is a string or null
        8. summary is a string of 5-25 words

    Returns:
        ValidationResult with .valid, .error, and .parsed_json
    """
    output = output.strip()

    # Step 1: Strict parse — the ENTIRE output must be valid JSON, nothing else
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return ValidationResult(False, f"JSON parse error: {e}")

    # Step 2: Must be a dict
    # A dict is typically composed of key-value pairs enclosed in 
    # curly braces {}.
    if not isinstance(data, dict):
        return ValidationResult(False, f"Expected JSON object, got {type(data).__name__}")

    # Step 3: Exact key match
    actual_keys = set(data.keys())
    if actual_keys != REQUIRED_KEYS:
        extra = actual_keys - REQUIRED_KEYS
        missing = REQUIRED_KEYS - actual_keys
        parts = []
        if extra:
            parts.append(f"extra keys: {extra}")
        if missing:
            parts.append(f"missing keys: {missing}")
        return ValidationResult(False, f"Key mismatch - {', '.join(parts)}")

    # Step 4: intent
    if data["intent"] not in VALID_INTENTS:
        return ValidationResult(False, f"Invalid intent '{data['intent']}', must be one of {VALID_INTENTS}")

    # Step 5: urgency
    if data["urgency"] not in VALID_URGENCIES:
        return ValidationResult(False, f"Invalid urgency '{data['urgency']}', must be one of {VALID_URGENCIES}")

    # Step 6: needs_human
    if not isinstance(data["needs_human"], bool):
        return ValidationResult(False, f"needs_human must be boolean, got {type(data['needs_human']).__name__}")

    # Step 7: product
    if data["product"] is not None and not isinstance(data["product"], str):
        return ValidationResult(False, f"product must be string or null, got {type(data['product']).__name__}")

    # Step 8: summary word count
    if not isinstance(data["summary"], str):
        return ValidationResult(False, f"summary must be string, got {type(data['summary']).__name__}")
    word_count = len(data["summary"].split())
    if word_count < 5 or word_count > 25:
        return ValidationResult(False, f"summary must be 5-25 words, got {word_count}")

    return ValidationResult(True, parsed_json=data)
