"""
tests/test_state_manager.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive unit tests for StateManager, Redis handling, at-rest encryption,
session expiration, OTP validation, and attempt limit gating.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.state_manager import SessionCipher, StateManager


@pytest.fixture
def clean_state_mgr() -> StateManager:
    """Provides a fresh StateManager instance with clean in-memory state."""
    mgr = StateManager()
    mgr.clear_all_for_test()
    return mgr


class TestSessionCipher:
    def test_encryption_and_decryption_roundtrip(self) -> None:
        cipher = SessionCipher("test_secret_key_12345")
        data = '{"session_id": "sess_1", "farmer_name": "Ramchandra"}'
        encrypted = cipher.encrypt(data)
        assert encrypted != data
        assert "Ramchandra" not in encrypted
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == data

    def test_decrypt_fallback_for_raw_json(self) -> None:
        cipher = SessionCipher("test_secret_key_12345")
        raw = '{"raw": true}'
        # If text is not valid Fernet token, fallback returns text unchanged
        assert cipher.decrypt(raw) == raw


class TestStateManagerPersistence:
    @pytest.mark.asyncio
    async def test_session_save_and_retrieval(self, clean_state_mgr: StateManager) -> None:
        session_id = "test_sess_001"
        data = {
            "session_id": session_id,
            "farmer_language": "hi",
            "form_submission_status": "pending",
            "farmer_data": {"farmer_name": "Kisan"},
        }
        await clean_state_mgr.set_session(session_id, data, ttl_seconds=300)
        retrieved = await clean_state_mgr.get_session(session_id)
        assert retrieved is not None
        assert retrieved["session_id"] == session_id
        assert retrieved["farmer_data"]["farmer_name"] == "Kisan"
        assert "expires_at" in retrieved

    @pytest.mark.asyncio
    async def test_session_expiration(self, clean_state_mgr: StateManager) -> None:
        session_id = "test_exp_002"
        data = {"session_id": session_id, "status": "pending"}
        # Set with 0 TTL to trigger immediate expiration
        await clean_state_mgr.set_session(session_id, data, ttl_seconds=-1)
        retrieved = await clean_state_mgr.get_session(session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_session_deletion(self, clean_state_mgr: StateManager) -> None:
        session_id = "test_del_003"
        await clean_state_mgr.set_session(session_id, {"val": 123}, ttl_seconds=300)
        assert await clean_state_mgr.get_session(session_id) is not None
        deleted = await clean_state_mgr.delete_session(session_id)
        assert deleted is True
        assert await clean_state_mgr.get_session(session_id) is None


class TestOTPValidationAndSecurity:
    @pytest.mark.asyncio
    async def test_otp_submission_valid(self, clean_state_mgr: StateManager) -> None:
        session_id = "otp_sess_004"
        await clean_state_mgr.set_session(
            session_id,
            {"otp_pending": True, "otp_created_at": time.time(), "otp_attempts": 0},
            ttl_seconds=300,
        )
        valid, msg = await clean_state_mgr.record_otp_and_validate(session_id, "123456")
        assert valid is True
        assert msg == "OTP accepted."

        session = await clean_state_mgr.get_session(session_id)
        assert session["otp_pending"] is False
        assert session["otp_value"] == "123456"
        assert session["otp_attempts"] == 1

    @pytest.mark.asyncio
    async def test_otp_submission_invalid_format(self, clean_state_mgr: StateManager) -> None:
        session_id = "otp_sess_005"
        await clean_state_mgr.set_session(session_id, {"otp_pending": True}, ttl_seconds=300)
        # Not 6 digits
        valid, msg = await clean_state_mgr.record_otp_and_validate(session_id, "12")
        assert valid is False
        assert "6 digits" in msg

        # Non-numeric
        valid, msg = await clean_state_mgr.record_otp_and_validate(session_id, "12345A")
        assert valid is False
        assert "6 digits" in msg

    @pytest.mark.asyncio
    async def test_otp_submission_expired(self, clean_state_mgr: StateManager) -> None:
        session_id = "otp_sess_006"
        # Created 10 minutes ago (beyond 300s window)
        old_time = time.time() - 600
        await clean_state_mgr.set_session(
            session_id,
            {"otp_pending": True, "otp_created_at": old_time, "otp_attempts": 0},
            ttl_seconds=3600,
        )
        valid, msg = await clean_state_mgr.record_otp_and_validate(session_id, "654321")
        assert valid is False
        assert "expired" in msg.lower()

        session = await clean_state_mgr.get_session(session_id)
        assert session is not None
        assert session["otp_expired"] is True
        assert session["form_submission_status"] == "failed"

    @pytest.mark.asyncio
    async def test_otp_max_attempts_exceeded(self, clean_state_mgr: StateManager) -> None:
        session_id = "otp_sess_007"
        await clean_state_mgr.set_session(
            session_id,
            {"otp_pending": True, "otp_created_at": time.time(), "otp_attempts": 3},
            ttl_seconds=300,
        )
        # 4th attempt when max is 3
        valid, msg = await clean_state_mgr.record_otp_and_validate(session_id, "888999")
        assert valid is False
        assert "exceeded" in msg.lower()


class TestRedisAvailabilityAndFallback:
    @pytest.mark.asyncio
    async def test_redis_available_read_write(self) -> None:
        """Mock active Redis connection and verify encrypted round-trip."""
        mgr = StateManager()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        store = {}

        async def mock_setex(k, ttl, val):
            store[k] = val
            return True

        async def mock_get(k):
            return store.get(k)

        mock_redis.setex = AsyncMock(side_effect=mock_setex)
        mock_redis.get = AsyncMock(side_effect=mock_get)
        mock_redis.delete = AsyncMock(return_value=1)

        mgr._redis = mock_redis
        mgr._redis_connected = True

        session_id = "redis_test_001"
        data = {"session_id": session_id, "farmer": "Devendra"}
        await mgr.set_session(session_id, data, ttl_seconds=60)

        # Ensure Redis was called with encrypted ciphertext
        assert f"session:{session_id}" in store
        raw_in_redis = store[f"session:{session_id}"]
        assert "Devendra" not in raw_in_redis  # At-rest encryption check

        # Verify retrieval and decryption
        retrieved = await mgr.get_session(session_id)
        assert retrieved is not None
        assert retrieved["farmer"] == "Devendra"

    @pytest.mark.asyncio
    async def test_redis_unavailable_fallback(self) -> None:
        """When Redis connection fails, transparently operates in safe fallback."""
        mgr = StateManager()
        mgr._redis_connected = False
        mgr._last_redis_failure_time = time.time()  # in cooldown

        session_id = "fallback_sess_002"
        await mgr.set_session(session_id, {"key": "val"}, ttl_seconds=60)
        retrieved = await mgr.get_session(session_id)
        assert retrieved is not None
        assert retrieved["key"] == "val"


class TestAdvancedStateAndPrivacyFeatures:
    @pytest.mark.asyncio
    async def test_atomic_update_session(self, clean_state_mgr: StateManager) -> None:
        session_id = "atomic_sess_001"
        await clean_state_mgr.set_session(session_id, {"fieldA": 1, "fieldB": "initial"})
        updated = await clean_state_mgr.update_session(session_id, {"fieldB": "modified", "fieldC": True})
        assert updated is not None
        assert updated["fieldA"] == 1
        assert updated["fieldB"] == "modified"
        assert updated["fieldC"] is True

        retrieved = await clean_state_mgr.get_session(session_id)
        assert retrieved is not None
        assert retrieved["fieldA"] == 1
        assert retrieved["fieldB"] == "modified"

    @pytest.mark.asyncio
    async def test_clear_otp_purges_otp_state(self, clean_state_mgr: StateManager) -> None:
        session_id = "clear_otp_sess_002"
        await clean_state_mgr.set_session(
            session_id,
            {"otp_value": "123456", "otp_pending": False, "status": "submitted"},
        )
        await clean_state_mgr.clear_otp(session_id)
        session = await clean_state_mgr.get_session(session_id)
        assert session is not None
        assert session["otp_value"] is None
        assert session["otp_pending"] is False
        assert session["status"] == "submitted"

    @pytest.mark.asyncio
    async def test_sanitize_state_redacts_aadhaar(self, clean_state_mgr: StateManager) -> None:
        session_id = "sanitize_sess_003"
        await clean_state_mgr.set_session(
            session_id,
            {
                "aadhaar_number": "123456789012",
                "nested": {"id_val": "987654321098"},
                "aadhaar_last4": "1098",
            },
        )
        session = await clean_state_mgr.get_session(session_id)
        assert session is not None
        assert session["aadhaar_number"] == "[Aadhaar Redacted]"
        assert session["nested"]["id_val"] == "[Aadhaar Redacted]"
        assert session["aadhaar_last4"] == "1098"

    @pytest.mark.asyncio
    async def test_message_deduplication_tracking(self, clean_state_mgr: StateManager) -> None:
        msg_id = f"MSG_unique_{int(time.time()*1000)}"
        assert await clean_state_mgr.is_message_processed(msg_id) is False
        await clean_state_mgr.mark_message_processed(msg_id, ttl_seconds=60)
        assert await clean_state_mgr.is_message_processed(msg_id) is True
        r = await clean_state_mgr._get_redis()
        if r:
            await r.delete(f"message:{msg_id}")


