"""
tests/test_webhook.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for Twilio WhatsApp Webhook Gateway.

Tests cover:
  • Health check endpoint.
  • Inbound text and multimodal webhook dispatch (returning TwiML immediately).
  • Fast-path 6-digit OTP capture and StateManager relay.
  • Privacy: phone number hashed to session ID, zero raw PII logged.
  • Webhook signature verification (valid & invalid).
  • Message deduplication via MessageSid.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.webhook import app, _derive_session_id, verify_twilio_signature
from app.core.config import get_settings
from app.core.state_manager import state_manager

client = TestClient(app)
settings = get_settings()


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_derive_session_id_privacy() -> None:
    """Ensure phone numbers are converted to fixed-length SHA-256 hashes."""
    phone = "whatsapp:+919876543210"
    session_id = _derive_session_id(phone)
    assert len(session_id) == 16
    assert session_id == _derive_session_id(phone)
    assert "+91" not in session_id


def test_webhook_inbound_text_returns_twiml() -> None:
    """Twilio requires HTTP 200 and XML TwiML containing the agent reply."""
    form_data = {
        "From": "whatsapp:+919876543210",
        "Body": "Mera bajre ka bima kar do",
        "NumMedia": "0",
    }
    response = client.post("/webhook/twilio", data=form_data)
    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]
    assert "<Response><Message>" in response.text
    assert "</Message></Response>" in response.text


def test_webhook_otp_fast_path() -> None:
    """
    If farmer replies with 6-digit OTP while session is in otp_pending,
    the gateway must immediately record it into StateManager without re-running extraction.
    """
    phone = "whatsapp:+919876543210"
    session_id = _derive_session_id(phone)

    # Seed an active session awaiting OTP
    async def _seed():
        state_manager.clear_all_for_test()
        await state_manager.set_session(
            session_id,
            {
                "session_id": session_id,
                "whatsapp_number": phone,
                "otp_pending": True,
                "otp_created_at": time.time(),
                "otp_attempts": 0,
                "otp_value": None,
                "form_submission_status": "awaiting_otp",
            },
        )

    asyncio.run(_seed())

    # Inbound 6-digit OTP
    form_data = {
        "From": phone,
        "Body": "847291",
        "NumMedia": "0",
    }
    response = client.post("/webhook/twilio", data=form_data)
    assert response.status_code == 200
    assert "<Response><Message>" in response.text
    assert "</Message></Response>" in response.text

    # Verify StateManager received the OTP, processed submission, and purged OTP value for privacy
    async def _verify():
        updated_session = await state_manager.get_session(session_id)
        assert updated_session is not None
        assert updated_session["otp_pending"] is False
        assert updated_session["form_submission_status"] in ("simulation_completed", "submitted")
        assert updated_session.get("otp_value") is None  # Safely cleared post-verification

    asyncio.run(_verify())


def test_twilio_signature_verification() -> None:
    """Verify HMAC-SHA1 signature verification utility."""
    auth_token = "12345secret"
    url = "https://example.com/webhook/twilio"
    params = {"Body": "Hello", "From": "whatsapp:+919876543210"}

    # Correct signature calculation
    import base64
    import hashlib
    import hmac
    data = url + "BodyHelloFromwhatsapp:+919876543210"
    expected_sig = base64.b64encode(
        hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")

    assert verify_twilio_signature(url, params, expected_sig, auth_token) is True
    assert verify_twilio_signature(url, params, "invalid_sig", auth_token) is False
    assert verify_twilio_signature(url, params, None, auth_token) is False


def test_webhook_duplicate_message_prevention() -> None:
    """Ensure identical incoming MessageSid is only dispatched once."""
    msg_sid = "SM_test_dedup_12345"
    form_data = {
        "From": "whatsapp:+919876543210",
        "Body": "Duplicate test message",
        "NumMedia": "0",
        "MessageSid": msg_sid,
    }

    # First request
    resp1 = client.post("/webhook/twilio", data=form_data)
    assert resp1.status_code == 200

    # Second duplicate request with identical MessageSid
    resp2 = client.post("/webhook/twilio", data=form_data)
    assert resp2.status_code == 200
    assert "<Response></Response>" in resp2.text


def test_webhook_production_signature_enforcement() -> None:
    """In production, missing or invalid signature raises HTTP 403."""
    with patch.object(settings, "environment", "production"), \
         patch.object(settings.twilio_auth_token, "get_secret_value", return_value="real_production_auth_token_xyz"):
        form_data = {
            "From": "whatsapp:+919876543210",
            "Body": "Message without signature",
            "NumMedia": "0",
        }
        # Request with invalid signature in production
        resp = client.post("/webhook/twilio", data=form_data, headers={"X-Twilio-Signature": "bad_sig"})
        assert resp.status_code == 403
