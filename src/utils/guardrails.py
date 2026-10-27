"""
Task 9: Reliability and Safety Controls.
Validates user inputs, screens for prompt injections and unsafe queries
(including PII, HIPAA/medical records, and direct medical prescriptions),
verifies file format/size constraints, and prevents unsafe outputs.
"""
import os
import re
from typing import Tuple
from src.config import Config

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?system prompts",
    r"you are now in developer mode",
    r"bypass safety",
    r"reveal your system prompt",
    r"repeat your instructions",
    r"output raw secrets",
]

UNSAFE_REQUEST_PATTERNS = [
    (
        r"\b(ssn|social security number|credit card|cvv|user password|master password)\b",
        "Privacy Guardrail: Requests for sensitive credentials or personal financial identifiers (PII) are restricted.",
    ),
    (
        r"\b(medical record|medical history|hipaa|patient record|doctor'?s personal|personal information of doctor)\b",
        "HIPAA & Privacy Guardrail: I am not authorized to share confidential patient medical records, medical history, or personal staff information under enterprise privacy policies.",
    ),
    (
        r"\b(prescribe medicine|what medicine should i take|give me medical prescription|give me treatment right now)\b",
        "Clinical Safety Guardrail: As an AI assistant, I am not authorized to provide direct medical advice or prescriptions. Please consult a qualified physician.",
    ),
]


def validate_user_query(query: str) -> Tuple[bool, str]:
    """Checks query length, screens for adversarial prompt injections and unsafe topic requests."""
    if not query or not query.strip():
        return (
            False,
            "Query cannot be empty. Please enter a valid enterprise question.",
        )

    if len(query) > 2000:
        return (
            False,
            "Query exceeds maximum allowed length (2000 characters).",
        )

    # 1. Adversarial prompt injection defense
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return (
                False,
                "Safety Control Triggered: Adversarial prompt injection pattern detected.",
            )

    # 2. Unsafe PII / HIPAA / Clinical advice screening (as highlighted in Edureka session & criteria doc)
    for pattern, warning_msg in UNSAFE_REQUEST_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return False, warning_msg

    return True, "Valid"


def validate_uploaded_file(
    filename: str, file_size_bytes: int
) -> Tuple[bool, str]:
    """Validates file format and size before ingestion."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(list(Config.ALLOWED_EXTENSIONS)))
        return (
            False,
            f"Unsupported file format '{ext}' in '{filename}'. Allowed enterprise formats: {allowed_str}.",
        )

    max_bytes = Config.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return (
            False,
            f"File '{filename}' exceeds the {Config.MAX_FILE_SIZE_MB}MB upload limit.",
        )

    return True, "Valid"
