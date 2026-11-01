"""
app/agent/graph.py
──────────────────────────────────────────────────────────────────────────────
LangGraph Cyclic State Machine Orchestrator for Kisan-Agent.

GRAPH TOPOLOGY & AGENTIC WORKFLOW:
  START -> Multimodal Extraction (Node 1)
        -> Guardrail Validation Gate (Checksums, master-table lookup, regex)
           ├── [conf < 0.95 or conflict] -> Human Clarification Edge (Voice/Text via WhatsApp)
           │                                 └──> Farmer replies -> loops back to Extraction
           └── [conf >= 0.95]           -> Geospatial Verification (Node 2)
                                              └──> Browser Automation Loop (Node 3)
                                                    └──> OTP Checkpoint (Node 4: Pauses state in Redis)
                                                          └──> Farmer sends OTP via WhatsApp
                                                                └──> Browser Resume (Node 3b)
                                                                      ├── [submitted / simulation_completed] -> Audit & Deliver -> END
                                                                      └── [failed] -> Error Handler -> END
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.browser_worker import browser_automation_node, browser_resume_node
from app.core.config import get_settings
from app.core.state import AgentState
from app.core.state_manager import state_manager
from app.geo.spatial_router import geospatial_verification_node
from app.guardrails.validators import validate_farmer_data
from app.senses.multimodal import multimodal_extraction_node
from app.services.messaging import send_text_reply

logger = logging.getLogger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# GUARDRAIL VALIDATION NODE
# ──────────────────────────────────────────────────────────────────────────────
async def guardrail_validation_node(state: AgentState) -> AgentState:
    """
    Firewall between AI OCR and portal browser execution.
    Runs deterministic regex, checksums, and master table lookups.
    """
    session_id = state.get("session_id", "unknown")
    logger.info("[Guardrail] Validating extracted data for session %s", session_id)

    raw_data: dict[str, Any] = dict(state.get("farmer_data", {}))
    result = validate_farmer_data(raw_data)

    if result.is_valid and result.validated_data:
        state["farmer_data"] = result.validated_data.model_dump()
        state["next_node"] = "geospatial_verification"
        logger.info("[Guardrail] All fields validated deterministically. Routing to geospatial verification.")
    elif result.needs_clarification:
        state["needs_clarification"] = True
        state["clarification_field"] = result.clarification_field or "khasra_plot_no"
        state["next_node"] = "human_clarification"
        logger.warning("[Guardrail] Low confidence detected on '%s'. Routing to Human Clarification Gate.", result.clarification_field)
    else:
        state["error"] = result.error_message or "Deterministic validation failed."
        state["form_submission_status"] = "failed"
        state["next_node"] = "error_handler"
        logger.error("[Guardrail] Hard validation failure: %s", result.error_message)

    return state


# ──────────────────────────────────────────────────────────────────────────────
# HUMAN CLARIFICATION NODE (HUMAN-IN-THE-LOOP EDGE)
# ──────────────────────────────────────────────────────────────────────────────
async def human_clarification_node(state: AgentState) -> AgentState:
    """
    Sends voice/text clarification prompt via WhatsApp in the farmer's dialect.
    Pauses execution safely until farmer responds.
    """
    field = state.get("clarification_field", "khasra_plot_no")
    session_id = state["session_id"]
    whatsapp_number = state.get("whatsapp_number", "")
    language = state.get("farmer_language", "hi")

    messages = {
        "khasra_plot_no": {
            "hi": "🌾 Kisaan bhai, aapka Khasra ya plot number photo mein saaf nahi dikh raha. Kripaya apna Khasra number text mein likhein ya saaf photo bhejein.",
            "mr": "🌾 शेतकरी बंधो, आपला खसरा क्रमांक स्पष्ट दिसत नाही. कृपया तो टाईप करून पाठवा.",
            "en": "🌾 Dear farmer, your Khasra / land plot number was not clearly visible. Please reply with your plot number or send a clearer photo.",
        },
        "ifsc_code": {
            "hi": "🌾 Aapka Bank IFSC code check karein. Passbook par likha 11-akshar ka IFSC code yahan reply karein (Jaise: SBIN0001234).",
            "en": "🌾 Please verify your Bank IFSC code. Reply with the 11-character code (e.g., SBIN0001234).",
        },
        "gps_location": {
            "hi": "🌾 Kripaya apna WhatsApp location share karein taaki aapka gaon aur jila verify ho sake.",
            "en": "🌾 Please share your WhatsApp location to verify your village and district.",
        },
    }

    msg = messages.get(field, {}).get(language, messages["khasra_plot_no"]["hi"])
    state["agent_reply"] = msg
    logger.info("[Human Clarification] Sending prompt for '%s' to %s", field, whatsapp_number)

    await send_text_reply(whatsapp_number, msg)

    # Persist state so the next inbound message can resume the graph
    paused_state = dict(state)
    paused_state["next_node"] = "multimodal_extraction"
    await state_manager.set_session(session_id, paused_state, ttl_seconds=settings.redis_session_ttl_seconds)

    state["next_node"] = END
    return state


# ──────────────────────────────────────────────────────────────────────────────
# STATEFUL OTP CHECKPOINT NODE
# ──────────────────────────────────────────────────────────────────────────────
async def otp_checkpoint_node(state: AgentState) -> AgentState:
    """
    Node 4: Pauses browser execution loop when SMS OTP is triggered.
    Prompts farmer on WhatsApp and safely suspends graph state in StateManager.
    """
    session_id = state["session_id"]
    whatsapp_number = state.get("whatsapp_number", "")
    language = state.get("farmer_language", "hi")

    # If OTP value is already present in state (e.g. from tests or prior node), resume directly
    if state.get("otp_value"):
        state["otp_pending"] = False
        state["next_node"] = "browser_resume"
        return state

    prompt = (
        "🔐 Portal ne aapke mobile number par ek 6-digit OTP bheja hai. "
        "Kripaya woh 6-digit OTP yahan reply karein taaki aapka form submit ho sake."
        if language == "hi" else
        "🔐 A 6-digit OTP has been sent by the PMFBY portal to your mobile. "
        "Please reply with the 6 digits to complete your submission."
    )

    state["agent_reply"] = prompt
    logger.info("[OTP Checkpoint] Prompting farmer for OTP on session %s", session_id)
    await send_text_reply(whatsapp_number, prompt)

    # Mark OTP pending and record timestamp
    state["otp_pending"] = True
    state["form_submission_status"] = "awaiting_otp"
    await state_manager.set_session(session_id, dict(state), ttl_seconds=settings.redis_session_ttl_seconds)

    # Pause graph execution cleanly; resumption is handled when farmer replies via WhatsApp
    state["next_node"] = END
    return state


# ──────────────────────────────────────────────────────────────────────────────
# RESUME PIPELINE WITH OTP (CALLED CROSS-MESSAGE FROM WEBHOOK)
# ──────────────────────────────────────────────────────────────────────────────
async def resume_agent_pipeline_with_otp(session_id: str, otp_code: str) -> AgentState:
    """
    Called when a farmer sends their 6-digit OTP in a separate WhatsApp message.
    Restores complete session state from Redis, validates OTP, executes browser_resume_node,
    and completes delivery or error handling.
    """
    logger.info("[Pipeline] Resuming session %s with OTP", session_id)
    session = await state_manager.get_session(session_id)
    if not session:
        logger.error("[Pipeline] No active session found to resume for %s", session_id)
        error_state: AgentState = {
            "session_id": session_id,
            "error": "Session expired or missing.",
            "form_submission_status": "failed",
            "agent_reply": "Session expired or missing. Kripaya naya aavedan shuru karein.",
        }
        return error_state

    # Duplicate submission prevention
    if session.get("form_submission_status") == "submitted":
        logger.warning("[Pipeline] Session %s already confirmed submitted. Ignoring duplicate OTP.", session_id)
        ref = session.get("application_reference_no", "")
        reply = f"🌾 Aapka aavedan pehle hi safalta-purvak submit ho chuka hai. Reference Number: {ref}"
        await send_text_reply(session.get("whatsapp_number", ""), reply)
        session["agent_reply"] = reply
        return session  # type: ignore[return-value]

    # Validate OTP format, attempts, and expiration
    valid, msg = await state_manager.record_otp_and_validate(session_id, otp_code)
    if not valid:
        logger.warning("[Pipeline] OTP validation failed for session %s: %s", session_id, msg)
        reply = f"⚠️ {msg}"
        await send_text_reply(session.get("whatsapp_number", ""), reply)
        session["error"] = msg
        session["agent_reply"] = reply
        session["form_submission_status"] = "failed"
        session["otp_value"] = None
        session["otp_pending"] = False
        await state_manager.set_session(session_id, session, ttl_seconds=settings.redis_session_ttl_seconds)
        return session  # type: ignore[return-value]

    # Reload validated state
    state: AgentState = await state_manager.get_session(session_id) or session  # type: ignore[assignment]
    state["otp_value"] = otp_code
    state["otp_pending"] = False

    # Execute browser resume node
    state = await browser_resume_node(state)

    # Clear OTP from memory and state after submission attempt (Privacy Directive)
    state["otp_value"] = None
    state["otp_pending"] = False

    # Route based on verified status
    status = state.get("form_submission_status")
    if status in ("submitted", "simulation_completed"):
        state = await audit_and_deliver_node(state)
    else:
        state = await error_handler_node(state)

    # Persist updated session
    await state_manager.set_session(session_id, dict(state), ttl_seconds=settings.redis_session_ttl_seconds)
    await state_manager.clear_otp(session_id)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT & DELIVER NODE
# ──────────────────────────────────────────────────────────────────────────────
async def audit_and_deliver_node(state: AgentState) -> AgentState:
    """
    Builds the source-attribution audit trail and delivers the official receipt to farmer.
    Distinguishes genuine submissions from simulated demonstrations.
    """
    session_id = state["session_id"]
    farmer_data = state.get("farmer_data", {})
    receipt_path = state.get("receipt_pdf_path", "")
    whatsapp_number = state.get("whatsapp_number", "")
    status = state.get("form_submission_status", "submitted")

    if status == "failed":
        return await error_handler_node(state)

    provenance = farmer_data.get("field_provenance", {})
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_id": session_id,
        "session_id": session_id,
        "submission_status": status,
        "is_simulation": state.get("is_simulation", False),
        "application_reference_no": state.get("application_reference_no"),
        "national_id": "[Aadhaar Redacted]",  # Security invariant
        "field_provenance_audit": {
            "farmer_name": {
                "value": farmer_data.get("farmer_name"),
                "source": provenance.get("farmer_name", {}).get("source", "identity_card"),
                "confidence": farmer_data.get("farmer_name_conf", 0.98),
            },
            "khasra_plot_no": {
                "value": farmer_data.get("khasra_plot_no"),
                "source": provenance.get("khasra_plot_no", {}).get("source", "khatauni"),
                "bbox": provenance.get("khasra_plot_no", {}).get("bbox", [120, 340, 280, 360]),
                "confidence": farmer_data.get("khasra_plot_no_conf", 0.98),
            },
            "ifsc_code": {
                "value": farmer_data.get("ifsc_code"),
                "source": provenance.get("ifsc_code", {}).get("source", "bank_passbook"),
                "bbox": provenance.get("ifsc_code", {}).get("bbox", [45, 210, 195, 230]),
                "confidence": farmer_data.get("ifsc_code_conf", 0.99),
            },
        },
        "receipt_file": receipt_path,
    }

    # Write audit log to disk
    os.makedirs(settings.audit_log_dir, exist_ok=True)
    audit_file = os.path.join(settings.audit_log_dir, f"{session_id}.json")
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2)
    logger.info("[Audit] Provenance log written to %s", audit_file)

    ref_no = state.get("application_reference_no") or f"PMFBY/2026/{session_id[:8].upper()}"

    if status == "submitted":
        delivery_msg = (
            "🌾 Badhai ho! Aapka PMFBY Fasal Bima aavedan safalta-purvak portal par submit ho gaya hai!\n\n"
            f"📋 Aavedan Number: {ref_no}\n"
            f"👨‍🌾 Kisaan: {farmer_data.get('farmer_name')}\n"
            f"🌱 Fasal: {farmer_data.get('crop_name', 'Bajra')}\n"
            f"📍 Khasra No: {farmer_data.get('khasra_plot_no')}\n"
            + (f"📄 Official receipt PDF download ho chuka hai: {receipt_path}" if receipt_path else "📄 Application reference number noted.")
        )
    elif status == "simulation_completed":
        delivery_msg = (
            "🌾 [DEMO SIMULATION] Yeh kewal ek demonstration test execution hai.\n\n"
            f"📋 Simulation ID: {ref_no}\n"
            f"👨‍🌾 Kisaan: {farmer_data.get('farmer_name')}\n"
            f"🌱 Fasal: {farmer_data.get('crop_name', 'Bajra')}\n"
            f"📍 Khasra No: {farmer_data.get('khasra_plot_no')}\n"
            f"📄 Demo Receipt: {receipt_path}\n\n"
            "⚠️ NOTICE: DEMO ONLY — NOT A GOVERNMENT RECEIPT. Is test execution se koi bima coverage pradan nahi kiya gaya hai."
        )
    else:
        delivery_msg = f"Aavedan status: {status}."

    state["agent_reply"] = delivery_msg
    await send_text_reply(whatsapp_number, delivery_msg)
    state["next_node"] = END
    return state


async def error_handler_node(state: AgentState) -> AgentState:
    err = state.get("error", "An error occurred during submission.")
    whatsapp_number = state.get("whatsapp_number", "")
    state["form_submission_status"] = "failed"
    state["otp_value"] = None
    state["otp_pending"] = False
    logger.error("[Error Node] Pipeline error on session %s: %s", state.get("session_id"), err)

    msg = f"Khed hai: {err}\nKripaya thodi der baad koshish karein ya PMFBY toll-free helpline 155261 par call karein."
    state["agent_reply"] = msg
    await send_text_reply(whatsapp_number, msg)
    state["next_node"] = END
    return state


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH TOPOLOGY ROUTING
# ──────────────────────────────────────────────────────────────────────────────
def _route_extraction(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("needs_clarification"):
        return "human_clarification"
    return "guardrail_validation"


def _route_guardrail(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("needs_clarification"):
        return "human_clarification"
    return "geospatial_verification"


def _route_geo(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("needs_clarification"):
        return "human_clarification"
    return "browser_automation"


def _route_browser(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("otp_pending"):
        return "otp_checkpoint"
    if state.get("form_submission_status") in ("submitted", "simulation_completed"):
        return "audit_and_deliver"
    return "error_handler"


def _route_otp(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("otp_value"):
        return "browser_resume"
    return END


def build_kisan_graph() -> StateGraph:
    """Build and compile the LangGraph cyclic state machine."""
    workflow = StateGraph(AgentState)

    workflow.add_node("multimodal_extraction", multimodal_extraction_node)
    workflow.add_node("guardrail_validation", guardrail_validation_node)
    workflow.add_node("human_clarification", human_clarification_node)
    workflow.add_node("geospatial_verification", geospatial_verification_node)
    workflow.add_node("browser_automation", browser_automation_node)
    workflow.add_node("otp_checkpoint", otp_checkpoint_node)
    workflow.add_node("browser_resume", browser_resume_node)
    workflow.add_node("audit_and_deliver", audit_and_deliver_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("multimodal_extraction")

    workflow.add_conditional_edges(
        "multimodal_extraction",
        _route_extraction,
        {
            "guardrail_validation": "guardrail_validation",
            "human_clarification": "human_clarification",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "guardrail_validation",
        _route_guardrail,
        {
            "geospatial_verification": "geospatial_verification",
            "human_clarification": "human_clarification",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "geospatial_verification",
        _route_geo,
        {
            "browser_automation": "browser_automation",
            "human_clarification": "human_clarification",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "browser_automation",
        _route_browser,
        {
            "otp_checkpoint": "otp_checkpoint",
            "audit_and_deliver": "audit_and_deliver",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "otp_checkpoint",
        _route_otp,
        {
            "browser_resume": "browser_resume",
            "error_handler": "error_handler",
            END: END,
        },
    )

    workflow.add_edge("browser_resume", "audit_and_deliver")
    workflow.add_edge("audit_and_deliver", END)
    workflow.add_edge("human_clarification", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


_compiled_graph: Any = None


async def run_agent_pipeline(session_id: str, initial_state: AgentState) -> AgentState:
    """Execute compiled LangGraph pipeline for an incoming farmer session."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_kisan_graph()

    try:
        # Duplicate submission check
        existing = await state_manager.get_session(session_id)
        if existing and existing.get("form_submission_status") == "submitted":
            logger.warning("[Pipeline] Duplicate submission prevented for session %s", session_id)
            ref = existing.get("application_reference_no", "")
            msg = f"🌾 Aapka aavedan pehle hi jama ho chuka hai (Ref: {ref})."
            await send_text_reply(initial_state.get("whatsapp_number", ""), msg)
            existing["agent_reply"] = msg
            return existing  # type: ignore[return-value]

        logger.info("[Pipeline] Starting LangGraph execution for session %s", session_id)
        final_state: AgentState = await _compiled_graph.ainvoke(initial_state)
        # Store state in StateManager
        await state_manager.set_session(session_id, dict(final_state), ttl_seconds=settings.redis_session_ttl_seconds)
        logger.info("[Pipeline] Session %s complete. Status: %s", session_id, final_state.get("form_submission_status"))
        return final_state
    except Exception as exc:
        logger.exception("[Pipeline] Unhandled pipeline error on %s: %s", session_id, exc)
        err_msg = "Khed hai: Takneeki samasya ke karan aavedan process nahi ho saka. Kripaya thodi der baad prayas karein."
        return {
            "session_id": session_id,
            "error": str(exc),
            "form_submission_status": "failed",
            "agent_reply": err_msg,
        }
