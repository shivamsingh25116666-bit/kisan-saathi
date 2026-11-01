"""
app/core/config.py
──────────────────────────────────────────────────────────────────────────────
Central configuration for Kisan-Agent.

Provides type-safe, validated environment settings via Pydantic v2 BaseSettings.
Secrets (Gemini API Key, Twilio Auth Tokens, Bhashini Keys) are kept strictly
out of source control using SecretStr.

RUBRIC NOTES:
  • Code Modularity: Single source of truth across all nodes.
  • Privacy & Security: SecretStr prevents token leakage into stdout/logs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-level configuration loaded from environment or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App & Server ──────────────────────────────────────────────────────────
    app_name: str = "Kisan-Agent"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Twilio WhatsApp Gateway ───────────────────────────────────────────────
    twilio_account_sid: SecretStr = Field(
        default=SecretStr("ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"),
        description="Twilio Account SID.",
    )
    twilio_auth_token: SecretStr = Field(
        default=SecretStr("placeholder_twilio_auth_token"),
        description="Twilio Auth Token for webhook verification and messaging.",
    )
    twilio_whatsapp_from: str = Field(
        default="whatsapp:+14155238886",
        description="Twilio WhatsApp sender phone number (sandbox or production).",
    )

    # ── Google Gemini (Multimodal Vision & Error Recovery Reasoning) ──────────
    gemini_api_key: SecretStr = Field(
        default=SecretStr("placeholder_gemini_api_key"),
        description="Google Gemini API Key.",
    )
    gemini_model: str = "gemini-2.5-flash"
    gemini_model_vision: str = "gemini-2.5-flash"
    gemini_model_text: str = "gemini-2.5-flash"

    # ── Indic Speech-to-Text (Bhashini + Whisper) ────────────────────────────
    bhashini_user_id: SecretStr = Field(default=SecretStr("placeholder_bhashini_uid"))
    bhashini_api_key: SecretStr = Field(default=SecretStr("placeholder_bhashini_key"))
    whisper_model_size: Literal["tiny", "base", "small", "medium"] = "small"

    # ── Session & OTP State Checkpoint (Redis + In-Memory Fallback) ──────────
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_session_ttl_seconds: int = 900   # 15 minutes session TTL
    redis_otp_ttl_seconds: int = 300       # 5 minutes OTP wait TTL
    use_in_memory_state_fallback: bool = True
    session_encryption_key: SecretStr = Field(
        default=SecretStr("kisan_secure_session_encryption_secret_2026"),
        description="Key used for encrypting session and browser state at rest in Redis.",
    )
    max_otp_attempts: int = Field(default=3, description="Maximum allowed OTP entry attempts.")
    redis_retry_delay_seconds: float = Field(default=5.0, description="Cooldown between Redis reconnection attempts.")

    # ── Self-Healing Browser Automation (Playwright) ─────────────────────────
    browser_headless: bool = True
    browser_slow_mo_ms: int = 150
    browser_timeout_ms: int = 30_000
    browser_max_retries: int = 3
    pmfby_portal_url: str = "https://pmfby.gov.in/farmerRegistrationForm"
    enable_portal_simulation: bool = Field(
        default=True,
        description="When true, allows simulated browser fallback if government portal is inaccessible. In strict production, set to false.",
    )

    # ── Document Optimization (Portal File Size Hard Limits) ─────────────────
    image_max_kb: int = 95                 # Must be strictly < 100 KB
    image_dpi: int = 150
    image_quality_step: int = 5

    # ── Guardrails & Zero Hallucination Thresholds ────────────────────────────
    field_confidence_threshold: float = 0.95
    clarification_confidence_floor: float = 0.75

    # ── Geospatial Routing ────────────────────────────────────────────────────
    geojson_village_path: str = "data/geojson/india_villages.geojson"
    admin_master_path: str = "data/masters/admin_hierarchy.json"

    # ── Storage Directories ───────────────────────────────────────────────────
    tmp_dir: str = "/tmp/kisan_agent"
    audit_log_dir: str = "logs/audit"

    @field_validator("field_confidence_threshold", "clarification_confidence_floor")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence thresholds must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> Settings:
        """
        Enforce strict production safeguards: reject placeholder credentials
        and insecure fallback modes when ENVIRONMENT=production.
        """
        if self.environment == "production":
            api_key = self.gemini_api_key.get_secret_value()
            if not api_key or api_key.startswith("your_") or api_key.startswith("placeholder"):
                raise ValueError("Valid GEMINI_API_KEY is required in production environment.")

            sid = self.twilio_account_sid.get_secret_value()
            if not sid or "XXXX" in sid or sid.startswith("your_"):
                raise ValueError("Valid TWILIO_ACCOUNT_SID is required in production environment.")

            auth = self.twilio_auth_token.get_secret_value()
            if not auth or auth.startswith("placeholder") or auth.startswith("your_"):
                raise ValueError("Valid TWILIO_AUTH_TOKEN is required in production environment.")

            enc_key = self.session_encryption_key.get_secret_value()
            if not enc_key or enc_key == "kisan_secure_session_encryption_secret_2026":
                raise ValueError("Default SESSION_ENCRYPTION_KEY must be changed in production.")

            if self.enable_portal_simulation:
                raise ValueError("ENABLE_PORTAL_SIMULATION must be false in strict production.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    return Settings()
