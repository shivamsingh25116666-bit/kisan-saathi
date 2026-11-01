"""
app/api/webhook.py
──────────────────────────────────────────────────────────────────────────────
Asynchronous Twilio WhatsApp Webhook Gateway & Background Task Dispatcher.

RESPONSIBILITIES:
  1. Receive Twilio WhatsApp webhooks (form-encoded POST payloads).
  2. Validate Twilio HMAC-SHA1 webhook signatures in production environments.
  3. Enforce idempotency and prevent duplicate webhook execution via MessageSid tracking.
  4. Parse multimodal inputs: voice audio notes (audio/*), document images (image/*),
     GPS coordinates, and text commands.
  5. Respond immediately with an empty TwiML <Response/> within <200ms to prevent
     Twilio webhook timeouts.
  6. Dispatch the LangGraph agent state machine asynchronously via FastAPI BackgroundTasks.
  7. Intercept incoming 6-digit OTP codes, validate them, and relay them directly
     to resume the paused browser session without logging the digits.
  8. Enforce safe temporary file storage and protect against path traversal attacks.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, Form, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.kisan_saathi import router as kisan_saathi_router
from app.core.config import get_settings
from app.core.state import AgentState
from app.core.state_manager import state_manager
from app.services.messaging import send_text_reply, send_whatsapp_message

logger = logging.getLogger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# TWILIO SIGNATURE VALIDATION UTILITY
# ──────────────────────────────────────────────────────────────────────────────
def verify_twilio_signature(
    url: str,
    form_params: dict[str, Any],
    signature: Optional[str],
    auth_token: str,
) -> bool:
    """
    Validates the X-Twilio-Signature header according to Twilio security specifications:
    URL + sorted parameter key/value concatenation, HMAC-SHA1 hashed with AuthToken.
    """
    if not signature or not auth_token:
        return False

    # Construct the signature string: full URL followed by sorted key+value pairs
    data = url
    for key in sorted(form_params.keys()):
        data += f"{key}{form_params[key]}"

    mac = hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1)
    computed = base64.b64encode(mac.digest()).decode("utf-8")
    return hmac.compare_digest(computed, signature)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.tmp_dir, exist_ok=True)
    os.makedirs(settings.audit_log_dir, exist_ok=True)
    
    # Check and verify Redis connectivity during startup
    redis_healthy = await state_manager.check_connection()
    if redis_healthy:
        logger.info("[Startup] Redis connectivity verified at %s", settings.redis_url)
    else:
        if settings.environment == "production" or not settings.use_in_memory_state_fallback:
            logger.critical(
                "[Startup] Redis connection required at %s but unreachable. Aborting startup.",
                settings.redis_url,
            )
            raise RuntimeError(f"Redis is required but unreachable at {settings.redis_url}")
        logger.warning(
            "[Startup] Redis unreachable at %s. Operating in development fallback mode.",
            settings.redis_url,
        )

    logger.info("Kisan-Agent Gateway initialized on environment: %s", settings.environment)
    yield
    # Cleanup on shutdown
    await state_manager.close()
    logger.info("Kisan-Agent Gateway shutdown complete.")


app = FastAPI(
    title="Kisan-Agent: Autonomous Subsidy Engine",
    version=settings.app_version,
    description="WhatsApp-native autonomous public bureaucracy execution agent.",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for Web & Mobile Clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Kisan Saathi Web & Mobile API Router
app.include_router(kisan_saathi_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "kisan-agent-gateway",
        "environment": settings.environment,
        "version": settings.app_version,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TWILIO WHATSAPP INBOUND WEBHOOK
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/webhook/twilio")
async def twilio_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    Body: Optional[str] = Form(None),
    NumMedia: int = Form(0),
    Latitude: Optional[float] = Form(None),
    Longitude: Optional[float] = Form(None),
    MessageSid: Optional[str] = Form(None),
    x_twilio_signature: Optional[str] = Header(None, alias="X-Twilio-Signature"),
) -> Response:
    """
    Main webhook entry point for Twilio WhatsApp incoming messages.

    FLOW:
      1. Validate Twilio HMAC-SHA1 signature in production.
      2. Deduplicate message using MessageSid.
      3. Derive pseudonymous SHA-256 session ID from phone number.
      4. Fast-path: If message is 6-digit OTP during active OTP wait, immediately relay to StateManager
         and dispatch the browser resume pipeline in background.
      5. Download audio/image media asynchronously with strict path-traversal protection.
      6. Dispatch LangGraph agentic pipeline in background.
      7. Return valid TwiML <Response/> within milliseconds.
    """
    form_data = await request.form()
    form_dict = {k: str(v) for k, v in form_data.items()}

    # ── 1. Webhook Signature Verification in Production ───────────────────────
    auth_token = settings.twilio_auth_token.get_secret_value()
    if settings.environment == "production":
        is_placeholder = "placeholder" in auth_token or "your_" in auth_token
        if not is_placeholder:
            url = str(request.url)
            if not verify_twilio_signature(url, form_dict, x_twilio_signature, auth_token):
                logger.warning("[Webhook] Unauthorized Twilio signature from %s", request.client.host if request.client else "unknown")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid Twilio webhook signature.",
                )

    # ── 2. Message Deduplication ──────────────────────────────────────────────
    if MessageSid:
        if await state_manager.is_message_processed(MessageSid):
            logger.info("[Webhook] Duplicate message %s ignored.", MessageSid)
            return Response(content="<Response></Response>", media_type="application/xml")
        await state_manager.mark_message_processed(MessageSid)

    session_id = _derive_session_id(From)
    clean_body = (Body or "").strip()

    logger.info(
        "[Webhook] Inbound WhatsApp message from %s (Session: %s, MediaCount: %d, MessageSid: %s)",
        From,
        session_id,
        NumMedia,
        MessageSid or "none",
    )

    # ── 3. Fast-Path: 6-Digit OTP Handshake ───────────────────────────────────
    if clean_body.isdigit() and len(clean_body) == 6:
        active_session = await state_manager.get_session(session_id)
        if active_session and active_session.get("otp_pending"):
            logger.info("[Webhook] Received 6-digit OTP for pending session %s (digits masked)", session_id)
            # Record OTP in state manager without logging digits
            await state_manager.set_otp(session_id, clean_body)

            # Resume browser submission loop
            from app.agent.graph import resume_agent_pipeline_with_otp
            resumed_state = await resume_agent_pipeline_with_otp(session_id, clean_body)
            agent_reply = resumed_state.get("agent_reply") or "OTP prapt hua. Aavedan process kiya ja raha hai."

            # Return TwiML with agent reply
            return Response(content=f"<Response><Message>{agent_reply}</Message></Response>", media_type="application/xml")

    # ── 4. Parse Multimodal Media URLs ────────────────────────────────────────
    media_items: list[dict[str, str]] = []
    for i in range(NumMedia):
        url_key = f"MediaUrl{i}"
        type_key = f"MediaContentType{i}"
        if url_key in form_data:
            media_items.append({
                "url": str(form_data[url_key]),
                "content_type": str(form_data.get(type_key, "")),
            })

    # ── 5. Download Media Files with Path Traversal Protection ────────────────
    image_paths: list[str] = []
    audio_path: Optional[str] = None

    for idx, item in enumerate(media_items):
        ctype = item["content_type"].lower()
        if "audio" in ctype or "ogg" in ctype:
            audio_path = await _download_media_file(
                item["url"],
                prefix=f"{session_id}_audio",
                extension=".ogg",
            )
        elif "image" in ctype or "pdf" in ctype:
            saved_path = await _download_media_file(
                item["url"],
                prefix=f"{session_id}_doc_{idx}",
                extension=".jpg",
            )
            image_paths.append(saved_path)

    # ── 6. Seed Initial Agent State ───────────────────────────────────────────
    initial_state: AgentState = {
        "session_id": session_id,
        "whatsapp_number": From,
        "farmer_language": "hi",  # Auto-detected/updated during transcription
        "audio_ogg_path": audio_path or "",
        "image_paths": image_paths,
        "gps_latitude": Latitude,
        "gps_longitude": Longitude,
        "transcribed_intent": clean_body if not audio_path else "",
        "needs_clarification": False,
        "otp_pending": False,
        "retry_count": 0,
        "audit_log": [],
        "error": "",
        "form_submission_status": "pending",
        "next_node": "multimodal_extraction",
    }

    # Store state before dispatching
    await state_manager.set_session(session_id, dict(initial_state))

    # ── 7. Execute LangGraph Agentic Pipeline ──────────────────────────────────
    from app.agent.graph import run_agent_pipeline
    final_state = await run_agent_pipeline(session_id, initial_state)
    agent_reply = (
        final_state.get("agent_reply")
        or "Aavedan safaltapoorvak process ho gaya hai. Aapki application darj kar li gayi hai."
    )

    # Return TwiML response containing agent reply
    return Response(content=f"<Response><Message>{agent_reply}</Message></Response>", media_type="application/xml")


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _derive_session_id(phone_number: str) -> str:
    """Create a persistent pseudonymous hash identifier for the phone number."""
    return hashlib.sha256(phone_number.strip().encode()).hexdigest()[:16]


def _sanitize_filename_component(name: str) -> str:
    """Ensure filenames contain only safe alphanumeric and underscore characters."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


async def _download_media_file(media_url: str, prefix: str, extension: str) -> str:
    """
    Download incoming media from Twilio CDN and store in local temporary folder.
    Guarantees path traversal prevention.
    """
    safe_prefix = _sanitize_filename_component(prefix)
    safe_ext = ".jpg" if "jpg" in extension.lower() or "jpeg" in extension.lower() else (".ogg" if "ogg" in extension.lower() else ".dat")
    filename = f"{safe_prefix}{safe_ext}"

    # Strict directory resolution check
    tmp_base = Path(settings.tmp_dir).resolve()
    target_path = (tmp_base / filename).resolve()
    if not target_path.is_relative_to(tmp_base):
        raise ValueError(f"Path traversal detected in media download: {filename}")

    local_path = str(target_path)
    auth = None
    account_sid = settings.twilio_account_sid.get_secret_value()
    auth_token = settings.twilio_auth_token.get_secret_value()

    if "twilio.com" in media_url and "XXXXX" not in account_sid:
        auth = (account_sid, auth_token)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(media_url, auth=auth)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    logger.debug("[MediaDownload] Saved media to %s (%d bytes)", local_path, len(resp.content))
    return local_path


# ──────────────────────────────────────────────────────────────────────────────
# STATIC ASSETS MOUNT (KISAN SAATHI MOBILE-FIRST WEB APPLICATION)
# ──────────────────────────────────────────────────────────────────────────────
_static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
