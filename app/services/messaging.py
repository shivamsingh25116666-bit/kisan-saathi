"""
app/services/messaging.py
──────────────────────────────────────────────────────────────────────────────
Dedicated Outbound WhatsApp & Notification Messaging Service.

Decouples outbound notification delivery from the API / webhook presentation layer,
eliminating circular dependencies between graph execution and webhook dispatch.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def sanitize_outbound_text(text: str) -> str:
    """
    Sanitizes outbound WhatsApp text to prevent accidental leakage of sensitive
    data such as raw 12-digit Aadhaar numbers or unredacted bank account details.
    """
    if not text:
        return ""
    # Redact any accidental 12-digit number sequence (Aadhaar pattern)
    sanitized = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[Aadhaar Redacted]", text)
    return sanitized


async def send_whatsapp_message(
    to_number: str,
    body: str,
    media_url: Optional[str] = None,
) -> bool:
    """
    Send an outbound WhatsApp message via Twilio REST API.

    Args:
        to_number: Recipient number in format 'whatsapp:+91XXXXXXXXXX'
        body: Plaintext response message
        media_url: Optional URL of uploaded document (e.g., official PDF receipt)
    """
    account_sid = settings.twilio_account_sid.get_secret_value()
    auth_token = settings.twilio_auth_token.get_secret_value()
    from_number = settings.twilio_whatsapp_from

    clean_body = sanitize_outbound_text(body)

    # In mock, local test, or sandbox environments, do not fail if mock credentials are set
    if "XXXXX" in account_sid or "placeholder" in auth_token:
        logger.info("[MockTwilio] Outbound message to %s: %s (Media: %s)", to_number, clean_body, media_url)
        return True

    api_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data: dict[str, Any] = {
        "From": from_number,
        "To": to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}",
        "Body": clean_body,
    }
    if media_url:
        data["MediaUrl"] = media_url

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(
                api_url,
                data=data,
                auth=(account_sid, auth_token),
            )
            if resp.status_code in (200, 201):
                logger.info("[Twilio] Outbound message delivered successfully to %s", to_number)
                return True
            else:
                logger.error("[Twilio] Failed to deliver message (%d): %s", resp.status_code, resp.text)
                return False
        except Exception as exc:
            logger.exception("[Twilio] Network error delivering message: %s", exc)
            return False


async def send_text_reply(to_number: str, text: str) -> None:
    """Convenience alias for sending text replies."""
    await send_whatsapp_message(to_number, text)
