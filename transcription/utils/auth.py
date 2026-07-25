"""Simple optional password gate for local Streamlit deployments."""

from __future__ import annotations

import hmac

from config import APP_PASSWORD


def password_required() -> bool:
    """True when TRANSCRIPTION_APP_PASSWORD is set."""
    return bool(APP_PASSWORD)


def verify_password(candidate: str) -> bool:
    """Constant-time compare against the configured app password."""
    if not APP_PASSWORD:
        return True
    return hmac.compare_digest(candidate.encode("utf-8"), APP_PASSWORD.encode("utf-8"))
