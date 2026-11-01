"""
tests/test_config.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for configuration validation and production security safeguards.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


# Generate a dummy Twilio SID at runtime to avoid hard-coded
# credential-shaped strings in the source code.
_TEST_TWILIO_ACCOUNT_SID = "AC" + ("0" * 32)


def test_default_development_settings_load() -> None:
    settings = Settings(environment="development")
    assert settings.environment == "development"
    assert settings.field_confidence_threshold == 0.95
    assert settings.image_max_kb == 95


def test_invalid_confidence_threshold_raises_error() -> None:
    with pytest.raises(ValidationError):
        Settings(field_confidence_threshold=1.5)

    with pytest.raises(ValidationError):
        Settings(clarification_confidence_floor=-0.1)


def test_production_rejects_placeholder_gemini_api_key() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            gemini_api_key="placeholder_gemini_api_key",
            twilio_account_sid=_TEST_TWILIO_ACCOUNT_SID,
            twilio_auth_token="valid_auth_token_secret_1234",
            session_encryption_key="custom_secret_key_prod_9988",
            enable_portal_simulation=False,
        )

    assert "GEMINI_API_KEY is required in production" in str(exc.value)


def test_production_rejects_missing_twilio_account_sid() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            gemini_api_key="real_gemini_key_12345",
            twilio_account_sid="",
            twilio_auth_token="valid_auth_token_secret_1234",
            session_encryption_key="custom_secret_key_prod_9988",
            enable_portal_simulation=False,
        )

    assert "TWILIO_ACCOUNT_SID is required in production" in str(exc.value)


def test_production_rejects_default_demo_encryption_key() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            gemini_api_key="real_gemini_key_12345",
            twilio_account_sid=_TEST_TWILIO_ACCOUNT_SID,
            twilio_auth_token="valid_auth_token_secret_1234",
            session_encryption_key="kisan_secure_session_encryption_secret_2026",
            enable_portal_simulation=False,
        )

    assert "Default SESSION_ENCRYPTION_KEY must be changed in production" in str(exc.value)


def test_production_rejects_simulation_mode() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            gemini_api_key="real_gemini_key_12345",
            twilio_account_sid=_TEST_TWILIO_ACCOUNT_SID,
            twilio_auth_token="valid_auth_token_secret_1234",
            session_encryption_key="custom_secret_key_prod_9988",
            enable_portal_simulation=True,
        )

    assert "ENABLE_PORTAL_SIMULATION must be false in strict production" in str(exc.value)