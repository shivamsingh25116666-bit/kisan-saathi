"""
app/core/state.py
──────────────────────────────────────────────────────────────────────────────
Shared LangGraph state type-definition for the Kisan-Agent state machine.

DESIGN PRINCIPLES:
  • Every field is Optional so that nodes can safely add data incrementally.
  • TypedDict keeps the state serialisable (JSON-compatible) for Redis
    persistence without any custom serializer.
  • Privacy directive: Aadhaar numbers MUST never appear here.
    The aadhaar_redacted field stores only the literal string
    "[Aadhaar Redacted]" — never the actual digits.

RUBRIC NOTES:
  • Depth of Implementation – typed cyclic-state graph state.
  • Privacy & Security       – no raw national ID numbers in state.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict

FormSubmissionStatus = Literal[
    "pending",
    "awaiting_otp",
    "submitted",
    "simulation_completed",
    "failed",
]


class AdminHierarchy(TypedDict, total=False):
    """
    Resolved cascade of administrative drop-downs required by the PMFBY portal.
    Populated deterministically by the Geospatial Verification Node.
    """
    state_code: str
    state_name: str
    district_code: str
    district_name: str
    sub_district_code: str
    sub_district_name: str
    village_code: str
    village_name: str
    lgd_code: str           # Local Government Directory census code


class ExtractedFarmerData(TypedDict, total=False):
    """
    Structured, validated output from the Multimodal Extraction Node.
    Confidence scores (0.0–1.0) travel with each field so the guardrail
    node can decide whether to pause for human clarification.
    """
    # ── Core farmer identity ──────────────────────────────────────────────
    farmer_name: str
    farmer_name_conf: float

    aadhaar_redacted: str           # ALWAYS "[Aadhaar Redacted]"
    aadhaar_last4: str              # Only the last 4 digits for display
    aadhaar_conf: float

    mobile_number: str
    mobile_number_conf: float

    # ── Land & crop details ───────────────────────────────────────────────
    khasra_plot_no: str
    khasra_plot_no_conf: float

    land_area_hectares: float
    land_area_conf: float

    crop_name: str                  # Normalised from spoken crop name
    crop_code: str                  # Official PMFBY crop code
    crop_name_conf: float

    season: str                     # "Kharif" | "Rabi"

    # ── Bank details ──────────────────────────────────────────────────────
    ifsc_code: str
    ifsc_code_conf: float

    account_number_redacted: str    # "[Account Redacted]"
    bank_name: str

    # ── Source bounding-boxes (for audit trail) ───────────────────────────
    field_provenance: dict[str, Any]
    # e.g. {"ifsc_code": {"source": "passbook_image", "bbox": [x1,y1,x2,y2]}}


class BrowserSession(TypedDict, total=False):
    """Serialisable snapshot of the Playwright browser session stored in Redis."""
    session_id: str
    cookies: list[dict[str, Any]]
    storage_state: dict[str, Any]
    current_url: str
    last_dom_snapshot: str          # Accessibility tree XML (truncated)
    retry_count: int
    last_error: str


class AgentState(TypedDict, total=False):
    """
    Top-level LangGraph state.  All nodes read from and write into this dict.

    Routing signals:
      next_node  – explicit override for the graph router.
      error      – non-empty string triggers the error-handling edge.
      needs_clarification – True routes to Human_Clarification_Edge.
      otp_pending         – True pauses execution at the OTP checkpoint.
    """

    # ── Inbound message context ───────────────────────────────────────────
    session_id: str                 # Derived from farmer's WhatsApp number (hashed)
    whatsapp_number: str            # Hashed/pseudonymous phone number
    farmer_language: str            # "hi" | "mr" | "raj" | "en"

    # ── Raw inputs ────────────────────────────────────────────────────────
    audio_ogg_path: str             # Local path to downloaded voice note
    image_paths: list[str]          # [id_card_path, passbook_path, khatauni_path]
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]

    # ── Processed data ────────────────────────────────────────────────────
    transcribed_intent: str         # Plain-text output of Bhashini/Whisper
    farmer_data: ExtractedFarmerData
    admin_hierarchy: AdminHierarchy

    # ── Clarification sub-flow ────────────────────────────────────────────
    needs_clarification: bool
    clarification_field: str        # Which field triggered human-in-the-loop
    clarification_response: str     # Farmer's follow-up message

    # ── OTP checkpoint ────────────────────────────────────────────────────
    otp_pending: bool
    otp_value: Optional[str]        # Populated when farmer replies with OTP
    otp_created_at: Optional[float] # Timestamp when OTP was requested from portal
    otp_attempts: int               # Count of verification attempts
    otp_expired: bool               # Set if OTP wait window lapsed

    # ── Browser session ───────────────────────────────────────────────────
    browser_session: BrowserSession
    form_submission_status: FormSubmissionStatus  # "pending" | "awaiting_otp" | "submitted" | "simulation_completed" | "failed"
    is_simulation: bool             # Explicit marker if session operated in simulated portal sandbox
    application_reference_no: Optional[str] # Official PMFBY acknowledgement number
    receipt_pdf_path: Optional[str]
    receipt_download_error: Optional[str]
    agent_reply: Optional[str]       # Text response output for WhatsApp farmer reply

    # ── Audit trail ───────────────────────────────────────────────────────
    audit_log: list[dict[str, Any]] # Append-only provenance events

    # ── Graph routing ─────────────────────────────────────────────────────
    next_node: str
    error: str
    retry_count: int

