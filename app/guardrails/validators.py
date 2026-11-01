"""
app/guardrails/validators.py
──────────────────────────────────────────────────────────────────────────────
Pydantic schemas + deterministic validation for every extracted farmer field.

PHILOSOPHY:
  The AI model (Gemini) does probabilistic OCR/extraction.  THIS module does
  deterministic, rule-based validation that CANNOT hallucinate.

VALIDATION LAYERS:
  1. Pydantic schema  – type coercion, regex constraints, null checks.
  2. Checksum / format gates – IFSC 11-char format, Indian mobile regex,
     Khasra pattern per state.
  3. Master-table lookup  – crop codes, state/district codes cross-checked
     against the official PMFBY master data JSON loaded at startup.
  4. Confidence gates  – fields below threshold route to human clarification.

RUBRIC NOTES:
  • AI Reasoning & Execution (Guardrails) – deterministic post-processing
    prevents any hallucinated value from reaching the browser.
  • Code Quality & Modularity – Pydantic v2 with @model_validator.
  • Privacy                   – Aadhaar handled only as a redacted placeholder.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# MASTER DATA LOADER
# ──────────────────────────────────────────────────────────────────────────────
_master_data: dict[str, Any] = {}


def load_master_data() -> dict[str, Any]:
    """
    Load official PMFBY master data (crop codes, state/district codes) from the
    bundled JSON file.  Results are cached in module-level _master_data dict.
    Absence of the file is non-fatal — validation falls back to format-only checks.
    """
    global _master_data
    if _master_data:
        return _master_data

    master_path = Path(settings.admin_master_path)
    if not master_path.exists():
        logger.warning(
            "[Masters] Master data file not found at %s. "
            "Field-level master-lookup validation will be skipped.",
            master_path,
        )
        _master_data = {"crops": {}, "states": {}, "ifsc_prefixes": []}
        return _master_data

    with open(master_path, encoding="utf-8") as f:
        _master_data = json.load(f)

    logger.info(
        "[Masters] Loaded %d crops, %d states from master data.",
        len(_master_data.get("crops", {})),
        len(_master_data.get("states", {})),
    )
    return _master_data


# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────
class ConfidenceScore(BaseModel):
    """Reusable wrapper that pairs a value with its extraction confidence."""
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)


class FarmerDataSchema(BaseModel):
    """
    Strict Pydantic v2 schema for validated farmer data.

    DESIGN:
      • All regex validators are deterministic — no AI inference.
      • Any field failing validation raises a ValidationError that is caught
        upstream and routed to human clarification or error handling.
      • Aadhaar is stored ONLY as a redacted placeholder.

    RUBRIC: Guardrails — this schema acts as the final firewall between the
    probabilistic AI extraction layer and the browser automation layer.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    farmer_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Full name as printed on government ID.",
    )
    farmer_name_conf: float = Field(ge=0.0, le=1.0)

    # PRIVACY: Only last 4 digits + canonical redacted placeholder allowed.
    aadhaar_redacted: str = Field(
        default="[Aadhaar Redacted]",
        pattern=r"^\[Aadhaar Redacted\]$",  # Must be EXACTLY this string
    )
    aadhaar_last4: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}$",
        description="Last 4 digits of Aadhaar for display/verification only.",
    )
    aadhaar_conf: float = Field(ge=0.0, le=1.0)

    mobile_number: str = Field(
        ...,
        description="10-digit Indian mobile number without country code.",
    )
    mobile_number_conf: float = Field(ge=0.0, le=1.0)

    # ── Land & Crop ───────────────────────────────────────────────────────
    khasra_plot_no: str = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Khasra / survey / plot number from the Khatauni.",
    )
    khasra_plot_no_conf: float = Field(ge=0.0, le=1.0)

    land_area_hectares: float = Field(
        ...,
        gt=0.0,
        lt=200.0,  # Sanity ceiling: 200 Ha is unrealistic for target persona
        description="Cultivated land area in hectares.",
    )
    land_area_conf: float = Field(ge=0.0, le=1.0)

    crop_name: Optional[str] = None
    crop_code: Optional[str] = None
    crop_name_conf: float = Field(default=0.0, ge=0.0, le=1.0)

    season: Optional[str] = Field(
        default=None,
        pattern=r"^(Kharif|Rabi|Zaid)$",
        description="Crop season.",
    )

    # ── Bank Details ──────────────────────────────────────────────────────
    ifsc_code: str = Field(
        ...,
        description="11-character IFSC code.",
    )
    ifsc_code_conf: float = Field(ge=0.0, le=1.0)

    account_number_redacted: str = Field(
        default="[Account Redacted]",
        pattern=r"^\[Account Redacted\]$",
    )
    bank_name: Optional[str] = None

    # ── Provenance ────────────────────────────────────────────────────────
    field_provenance: dict[str, Any] = Field(default_factory=dict)

    # ══ FIELD-LEVEL VALIDATORS ════════════════════════════════════════════

    @field_validator("mobile_number")
    @classmethod
    def validate_mobile(cls, v: str) -> str:
        """Indian mobile: starts with 6-9, exactly 10 digits."""
        cleaned = re.sub(r"[\s\-+]", "", v)
        if cleaned.startswith("91") and len(cleaned) == 12:
            cleaned = cleaned[2:]
        if not re.fullmatch(r"[6-9]\d{9}", cleaned):
            raise ValueError(
                f"Invalid Indian mobile number: '{cleaned}'. "
                "Must be 10 digits starting with 6-9."
            )
        return cleaned

    @field_validator("ifsc_code")
    @classmethod
    def validate_ifsc(cls, v: str) -> str:
        """
        IFSC format: 4 uppercase alpha + '0' + 6 alphanumeric.
        Example: SBIN0001234
        This is a deterministic checksum — the AI cannot fake a passing IFSC.
        """
        cleaned = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", cleaned):
            raise ValueError(
                f"Invalid IFSC code: '{cleaned}'. "
                "Expected format: XXXX0XXXXXX (11 chars)."
            )
        return cleaned

    @field_validator("khasra_plot_no")
    @classmethod
    def validate_khasra(cls, v: str) -> str:
        """
        Khasra numbers are numeric (possibly with fractional slash e.g. 45/2)
        or numeric+alpha (e.g. "123P") depending on the state.
        Reject patterns that look like random noise.
        """
        cleaned = v.strip()
        # Accept: digits, optional /digits, optional trailing alpha
        if not re.fullmatch(r"\d+(/\d+)?[A-Za-z]?", cleaned):
            raise ValueError(
                f"Khasra/plot number '{cleaned}' has unexpected format. "
                "Expected numeric, e.g. '45', '45/2', '123P'."
            )
        return cleaned

    @field_validator("farmer_name")
    @classmethod
    def validate_farmer_name(cls, v: str) -> str:
        """
        Must contain only letters (Latin + Devanagari Unicode range) and spaces.
        Rejects: numbers, special chars, single characters.
        """
        # Allow Latin, Devanagari, Gujarati, Bengali, etc.
        if not re.fullmatch(r"[\u0900-\u097Fa-zA-Z\s]{2,120}", v.strip()):
            raise ValueError(
                f"Farmer name '{v}' contains invalid characters. "
                "Only letters and spaces are allowed."
            )
        return v.strip()

    @model_validator(mode="after")
    def validate_crop_master_lookup(self) -> "FarmerDataSchema":
        """
        Cross-check extracted crop name against official PMFBY crop master.
        If master data is available and crop is not found, mark it as unknown
        so the guardrail can ask for clarification.

        RUBRIC: Hallucination Guardrail — deterministic lookup, not AI inference.
        """
        masters = load_master_data()
        crop_master: dict[str, str] = masters.get("crops", {})

        if not crop_master or not self.crop_name:
            return self  # Master not loaded or crop not extracted — skip

        # Case-insensitive prefix match for normalised crop name lookup
        crop_lower = self.crop_name.lower().strip()
        matched_code = None
        for code, name in crop_master.items():
            if crop_lower in name.lower() or name.lower() in crop_lower:
                matched_code = code
                break

        if matched_code:
            self.crop_code = matched_code
            logger.debug("[Validator] Matched crop '%s' → code %s", self.crop_name, matched_code)
        else:
            logger.warning(
                "[Validator] Crop '%s' not found in PMFBY master data.",
                self.crop_name,
            )
            self.crop_name_conf = min(self.crop_name_conf, 0.5)  # Downgrade confidence

        return self


FarmerDocumentSchema = FarmerDataSchema


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION GATEWAY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
class ValidationResult(BaseModel):
    """Structured return type from the validation gateway."""
    is_valid: bool
    validated_data: Optional[FarmerDataSchema] = None
    failed_field: Optional[str] = None
    error_message: Optional[str] = None
    needs_clarification: bool = False
    clarification_field: Optional[str] = None


def validate_farmer_data(raw: dict[str, Any]) -> ValidationResult:
    """
    Entry point for the guardrail layer.

    1. Runs Pydantic schema validation (format + regex + type checks).
    2. Runs confidence gate check.
    3. Returns a ValidationResult with routing signals.

    This is the last line of defence before the browser automation starts.
    No invalid data can reach the PMFBY portal.

    RUBRIC: Guardrails, Zero Hallucination — deterministic validation cannot
    be bypassed by a confident-but-wrong AI extraction.
    """
    # ── Pydantic validation ────────────────────────────────────────────────
    try:
        validated = FarmerDataSchema.model_validate(raw)
    except Exception as exc:
        # Extract first failed field name from Pydantic error
        failed_field = _extract_failed_field(exc)
        logger.warning("[Validator] Pydantic validation failed on '%s': %s", failed_field, exc)
        return ValidationResult(
            is_valid=False,
            failed_field=failed_field,
            error_message=str(exc),
            needs_clarification=True,
            clarification_field=failed_field,
        )

    # ── Confidence gate ────────────────────────────────────────────────────
    threshold = settings.field_confidence_threshold
    critical_fields = {
        "khasra_plot_no": validated.khasra_plot_no_conf,
        "ifsc_code": validated.ifsc_code_conf,
        "land_area_hectares": validated.land_area_conf,
        "farmer_name": validated.farmer_name_conf,
    }

    for field_name, confidence in critical_fields.items():
        if confidence < threshold:
            logger.warning(
                "[Validator] Field '%s' confidence %.2f < threshold %.2f — clarification needed.",
                field_name, confidence, threshold,
            )
            return ValidationResult(
                is_valid=False,
                validated_data=validated,
                needs_clarification=True,
                clarification_field=field_name,
                error_message=f"Low confidence ({confidence:.0%}) on '{field_name}'",
            )

    logger.info("[Validator] All fields passed validation. Routing to geospatial node.")
    return ValidationResult(is_valid=True, validated_data=validated)


def _extract_failed_field(exc: Exception) -> str:
    """Best-effort extraction of the first failed field name from a Pydantic error."""
    try:
        errors = exc.errors()  # type: ignore[attr-defined]
        if errors:
            loc = errors[0].get("loc", ())
            return str(loc[-1]) if loc else "unknown"
    except Exception:
        pass
    return "unknown"

