"""
app/core/state_manager.py
──────────────────────────────────────────────────────────────────────────────
Production-Grade Session & OTP State Manager with Redis & Secure Fallback.

RESPONSIBILITIES:
  1. Store active farmer state and serialized browser sessions across workers.
  2. Protect sensitive state at rest in Redis using AES-CBC (Fernet) encryption.
  3. Support reliable cross-message OTP resumption without zombie background tasks.
  4. Enforce strict privacy: OTP values and national IDs are NEVER logged.
  5. Provide resilient Redis connection pooling, auto-reconnection, and safe fallback.
  6. Idempotent webhook deduplication via MessageSid tracking.
  7. Atomic session mutations and per-session concurrency safety.
"""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
import hashlib
import json
import logging
import re
import time
from typing import Any, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken
import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionCipher:
    """Provides transparent AES-CBC encryption at rest using Fernet."""

    def __init__(self, secret: str) -> None:
        key_bytes = hashlib.sha256(secret.encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(key_bytes))

    def encrypt(self, plain_text: str) -> str:
        return self._fernet.encrypt(plain_text.encode("utf-8")).decode("utf-8")

    def decrypt(self, cipher_text: str) -> str:
        try:
            return self._fernet.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            # Fallback in case raw JSON was stored during migration or tests
            return cipher_text


def _sanitize_state(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively redact any raw 12-digit numbers or accidental Aadhaar fields.
    Guarantees strict privacy at rest in both Redis and in-memory cache.
    """
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if key == "aadhaar_number":
            sanitized[key] = "[Aadhaar Redacted]"
        elif isinstance(value, str) and key != "aadhaar_last4":
            # Redact 12-digit number sequence
            if re.fullmatch(r"\d{12}", value.replace(" ", "")):
                sanitized[key] = "[Aadhaar Redacted]"
            else:
                sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_state(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_state(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


class StateManager:
    """
    Unified asynchronous state manager for Kisan-Agent sessions.
    Handles Redis connection lifecycle, at-rest encryption, atomic updates,
    and safe development fallback.
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._pool: aioredis.ConnectionPool | None = None
        self._in_memory_store: dict[str, dict[str, Any]] = {}
        self._expiry_store: dict[str, float] = {}
        self._processed_messages: dict[str, float] = {}
        self._last_redis_failure_time: float = 0.0
        self._redis_connected: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Initialize cipher for encryption at rest
        secret = settings.session_encryption_key.get_secret_value()
        self._cipher = SessionCipher(secret)

    async def _get_redis(self) -> aioredis.Redis | None:
        """
        Get or initialize the Redis client with intelligent reconnection cooldown.
        Avoids permanent disconnection while preventing connection storming.
        """
        now = time.time()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._redis is not None:
            if self._loop is not None and (self._loop.is_closed() or (current_loop is not None and self._loop != current_loop)):
                self._redis = None
                self._pool = None
                self._redis_connected = False
            else:
                return self._redis

        # Cooldown check: do not retry within redis_retry_delay_seconds of a failure
        if (
            not self._redis_connected
            and (now - self._last_redis_failure_time) < settings.redis_retry_delay_seconds
        ):
            return None

        try:
            pool = aioredis.ConnectionPool.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3.0,
                socket_timeout=5.0,
                retry_on_timeout=True,
                max_connections=20,
            )
            client = aioredis.Redis(connection_pool=pool)
            await client.ping()
            self._pool = pool
            self._redis = client
            self._loop = current_loop
            self._redis_connected = True
            logger.info("[StateManager] Successfully connected to Redis at %s", settings.redis_url)
            return self._redis
        except Exception as exc:
            self._last_redis_failure_time = now
            self._redis_connected = False
            self._redis = None
            if self._pool is not None:
                await self._pool.disconnect()
                self._pool = None

            if settings.environment == "production":
                logger.critical("[StateManager] Redis connection failed in strict production: %s", exc)
                raise RuntimeError(f"Redis connection failed in production: {exc}") from exc

            if not settings.use_in_memory_state_fallback:
                logger.critical("[StateManager] Redis connection failed and fallback disabled: %s", exc)
                raise RuntimeError(f"Redis connection failed and fallback disabled: {exc}") from exc

            logger.warning(
                "[StateManager] Redis unavailable (%s). Operating in fallback state mode.",
                exc,
            )
            return None

    async def check_connection(self) -> bool:
        """
        Explicitly check Redis connectivity during application startup or health check.
        Returns True if reachable and healthy, False otherwise.
        """
        try:
            client = await self._get_redis()
            if client is not None:
                pong = await client.ping()
                if pong:
                    self._redis_connected = True
                    logger.info("[StateManager] Redis connectivity verified at %s", settings.redis_url)
                    return True
        except Exception as exc:
            self._redis_connected = False
            logger.warning("[StateManager] Redis health check failed at %s: %s", settings.redis_url, exc)
            return False
        return False

    def is_redis_available(self) -> bool:
        return self._redis_connected

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve state for a session by ID, decrypting data if needed."""
        if not session_id:
            return None

        key = f"session:{session_id}"
        redis_client = await self._get_redis()

        if redis_client:
            try:
                raw = await redis_client.get(key)
                if raw:
                    decrypted = self._cipher.decrypt(raw)
                    data: dict[str, Any] = json.loads(decrypted)
                    # Check logical expiration
                    if "expires_at" in data and time.time() > data["expires_at"]:
                        logger.info("[StateManager] Session %s has logically expired.", session_id)
                        await self.delete_session(session_id)
                        return None
                    return data
                # Key does not exist in Redis
                if not settings.use_in_memory_state_fallback:
                    return None
            except Exception as exc:
                if settings.environment == "production" or not settings.use_in_memory_state_fallback:
                    raise RuntimeError(f"Redis get failed: {exc}") from exc
                logger.warning("[StateManager] Redis get error on %s: %s. Checking fallback store.", session_id, exc)
        elif settings.environment == "production" or not settings.use_in_memory_state_fallback:
            raise RuntimeError(f"Redis unavailable for get_session({session_id}) in production/strict mode.")

        # Fallback to in-memory store (development only)
        self._cleanup_expired()
        item = self._in_memory_store.get(key)
        if item:
            if time.time() > self._expiry_store.get(key, 0.0):
                self._in_memory_store.pop(key, None)
                self._expiry_store.pop(key, None)
                return None
            return item.copy()

        return None

    async def set_session(
        self,
        session_id: str,
        data: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Persist session state with encryption and expiration.
        """
        if not session_id:
            raise ValueError("session_id cannot be empty")

        ttl = ttl_seconds if ttl_seconds is not None else settings.redis_session_ttl_seconds
        key = f"session:{session_id}"
        now = time.time()

        stored_data = _sanitize_state(dict(data))
        stored_data["session_id"] = session_id
        stored_data["updated_at"] = now
        stored_data["expires_at"] = now + ttl

        redis_client = await self._get_redis()
        serialized = json.dumps(stored_data, default=str)
        encrypted = self._cipher.encrypt(serialized)

        if redis_client:
            try:
                if ttl <= 0:
                    await redis_client.delete(key)
                else:
                    await redis_client.setex(key, ttl, encrypted)
                # Keep in-memory store populated for tests/redundancy if fallback enabled
                if settings.use_in_memory_state_fallback:
                    self._in_memory_store[key] = stored_data.copy()
                    self._expiry_store[key] = now + ttl
                return
            except Exception as exc:
                if settings.environment == "production" or not settings.use_in_memory_state_fallback:
                    raise RuntimeError(f"Redis set failed: {exc}") from exc
                logger.warning("[StateManager] Redis set error on %s: %s. Using fallback store.", session_id, exc)
        elif settings.environment == "production" or not settings.use_in_memory_state_fallback:
            raise RuntimeError(f"Redis unavailable for set_session({session_id}) in production/strict mode.")

        # Update in-memory cache for redundancy in development/fallback
        self._in_memory_store[key] = stored_data.copy()
        self._expiry_store[key] = now + ttl

    async def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Atomically updates existing session with specific fields under a per-session lock.
        Prevents race conditions when concurrent requests arrive for the same farmer.
        """
        local_lock = self._session_locks[session_id]
        async with local_lock:
            redis_client = await self._get_redis()
            if redis_client and hasattr(redis_client, "lock"):
                try:
                    async with redis_client.lock(
                        f"lock:session:{session_id}",
                        timeout=10.0,
                        blocking_timeout=5.0,
                    ):
                        current = await self.get_session(session_id)
                        if not current:
                            current = {"session_id": session_id}
                        current.update(updates)
                        await self.set_session(session_id, current, ttl_seconds=ttl_seconds)
                        return current
                except Exception as exc:
                    logger.debug("[StateManager] Distributed lock not used for %s: %s", session_id, exc)

            current = await self.get_session(session_id)
            if not current:
                current = {"session_id": session_id}
            current.update(updates)
            await self.set_session(session_id, current, ttl_seconds=ttl_seconds)
            return current

    async def record_otp_and_validate(
        self,
        session_id: str,
        otp_code: str,
    ) -> Tuple[bool, str]:
        """
        Validates OTP format, limits attempts, checks expiration, and records state.
        Returns: (is_valid, message)
        CRITICAL PRIVACY: The OTP value is never logged.
        """
        lock = self._session_locks[session_id]
        async with lock:
            clean_otp = (otp_code or "").strip()
            if not clean_otp.isdigit() or len(clean_otp) != 6:
                logger.warning("[StateManager] Rejected invalid OTP format for session %s", session_id)
                return False, "OTP must be exactly 6 digits."

            session = await self.get_session(session_id)
            if not session:
                logger.warning("[StateManager] Attempted OTP update on missing/expired session %s", session_id)
                return False, "Session not found or expired. Please submit a new application."

            # Check OTP expiration window
            otp_created = session.get("otp_created_at", session.get("updated_at", time.time()))
            if time.time() - otp_created > settings.redis_otp_ttl_seconds:
                session["otp_expired"] = True
                session["form_submission_status"] = "failed"
                await self.set_session(session_id, session, ttl_seconds=settings.redis_session_ttl_seconds)
                logger.warning("[StateManager] OTP expired for session %s", session_id)
                return False, "OTP verification has expired. Please initiate a new request."

            # Check maximum attempts
            attempts = int(session.get("otp_attempts", 0)) + 1
            session["otp_attempts"] = attempts
            if attempts > settings.max_otp_attempts:
                session["form_submission_status"] = "failed"
                await self.set_session(session_id, session, ttl_seconds=settings.redis_session_ttl_seconds)
                logger.warning(
                    "[StateManager] Max OTP attempts (%d) exceeded for session %s",
                    settings.max_otp_attempts,
                    session_id,
                )
                return False, f"Maximum OTP attempts ({settings.max_otp_attempts}) exceeded."

            # Store validated OTP without logging the digits
            session["otp_value"] = clean_otp
            session["otp_pending"] = False
            await self.set_session(session_id, session, ttl_seconds=settings.redis_session_ttl_seconds)

            logger.info(
                "[StateManager] Valid OTP recorded for session %s (attempt %d/%d)",
                session_id,
                attempts,
                settings.max_otp_attempts,
            )
            return True, "OTP accepted."

    async def set_otp(self, session_id: str, otp_code: str) -> bool:
        """Convenience wrapper for record_otp_and_validate."""
        success, _ = await self.record_otp_and_validate(session_id, otp_code)
        return success

    async def clear_otp(self, session_id: str) -> None:
        """Purge temporary OTP value from storage after verification or completion."""
        session = await self.get_session(session_id)
        if session and "otp_value" in session:
            session["otp_value"] = None
            session["otp_pending"] = False
            await self.set_session(session_id, session, ttl_seconds=settings.redis_session_ttl_seconds)
            logger.debug("[StateManager] Cleared OTP state for session %s", session_id)

    async def is_message_processed(self, message_sid: str) -> bool:
        """Check whether incoming message SID was already processed (deduplication)."""
        if not message_sid:
            return False

        key = f"message:{message_sid}"
        redis_client = await self._get_redis()
        if redis_client:
            try:
                val = await redis_client.get(key)
                return val is not None
            except Exception:
                pass

        # In-memory deduplication check
        exp = self._processed_messages.get(message_sid)
        if exp and time.time() < exp:
            return True
        return False

    async def mark_message_processed(self, message_sid: str, ttl_seconds: int = 3600) -> None:
        """Record message SID to prevent duplicate webhook processing."""
        if not message_sid:
            return

        key = f"message:{message_sid}"
        redis_client = await self._get_redis()
        if redis_client:
            try:
                await redis_client.set(key, "1", ex=max(1, ttl_seconds))
            except Exception:
                pass

        self._processed_messages[message_sid] = time.time() + ttl_seconds

    async def delete_session(self, session_id: str) -> bool:
        """Explicitly purge a session from Redis and memory."""
        key = f"session:{session_id}"
        self._in_memory_store.pop(key, None)
        self._expiry_store.pop(key, None)
        self._session_locks.pop(session_id, None)

        redis_client = await self._get_redis()
        if redis_client:
            try:
                await redis_client.delete(key)
                return True
            except Exception as exc:
                logger.warning("[StateManager] Redis delete error on %s: %s", session_id, exc)
        return True

    def _cleanup_expired(self) -> None:
        """Evict expired in-memory items and stale processed message SIDs."""
        now = time.time()
        expired_keys = [k for k, exp in self._expiry_store.items() if now > exp]
        for k in expired_keys:
            self._in_memory_store.pop(k, None)
            self._expiry_store.pop(k, None)

        expired_msgs = [sid for sid, exp in self._processed_messages.items() if now > exp]
        for sid in expired_msgs:
            self._processed_messages.pop(sid, None)

    async def close(self) -> None:
        """Gracefully disconnect Redis pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None
        self._redis_connected = False
        logger.info("[StateManager] Closed Redis connections.")

    def clear_all_for_test(self) -> None:
        """Utility for test suites to reset in-memory state."""
        self._in_memory_store.clear()
        self._expiry_store.clear()
        self._processed_messages.clear()
        self._session_locks.clear()
        self._last_redis_failure_time = 0.0
        self._redis = None
        self._pool = None
        self._redis_connected = False
        self._loop = None


# Global singleton
state_manager = StateManager()
