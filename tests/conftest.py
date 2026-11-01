"""
tests/conftest.py
──────────────────────────────────────────────────────────────────────────────
Shared fixtures for the Kisan-Agent test suite.

Auto-mocks external dependencies (Twilio outbound, Redis) so unit tests
can run without real credentials or network access.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from app.core.state_manager import state_manager


@pytest.fixture(autouse=True)
def _clean_state_manager():
    """Reset in-memory state before each test to prevent cross-test contamination."""
    state_manager.clear_all_for_test()
    yield
    state_manager.clear_all_for_test()


@pytest.fixture(autouse=True)
def _mock_send_whatsapp():
    """
    Auto-mock outbound WhatsApp messaging across all tests.
    The actual Twilio/send_whatsapp_message function requires credentials
    and network, so we replace it with a no-op async mock.
    """
    with patch("app.services.messaging.send_whatsapp_message", new_callable=AsyncMock, return_value=True), \
         patch("app.api.webhook.send_whatsapp_message", new_callable=AsyncMock, return_value=True):
        yield


@pytest.fixture(autouse=True)
def _ensure_tmp_dirs():
    """Ensure required directories exist for receipt generation and audit logs."""
    from app.core.config import get_settings
    settings = get_settings()
    os.makedirs(settings.tmp_dir, exist_ok=True)
    os.makedirs(settings.audit_log_dir, exist_ok=True)
    yield
