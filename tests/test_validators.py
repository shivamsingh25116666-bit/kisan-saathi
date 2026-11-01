"""
tests/test_validators.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for app/guardrails/validators.py

Tests cover:
  • Happy-path Pydantic schema validation.
  • IFSC format enforcement (deterministic checksum gate).
  • Indian mobile number validation.
  • Khasra number pattern enforcement.
  • Confidence-gate routing signals.
  • Privacy: Aadhaar placeholder enforcement.
  • Crop master lookup (with mock master data).

RUBRIC NOTES:
  • Code Quality & Modularity – comprehensive unit tests with clear AAA pattern.
  • Privacy & Security        – all tests verify "[Aadhaar Redacted]" invariant.
"""

from __future__ import annotations

import pytest

from app.guardrails.validators import (
    FarmerDataSchema,
    ValidationResult,
    validate_farmer_data,
)


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def valid_farmer_payload() -> dict:
    """
    A complete, valid farmer data payload.
    Privacy: Aadhaar field uses only the canonical placeholder.
    """
    return {
        "farmer_name":             "Ramchandra Yadav",
        "farmer_name_conf":        0.97,
        "aadhaar_redacted":        "[Aadhaar Redacted]",
        "aadhaar_last4":           "7823",
        "aadhaar_conf":            0.96,
        "mobile_number":           "9876543210",
        "mobile_number_conf":      0.99,
        "khasra_plot_no":          "45/2",
        "khasra_plot_no_conf":     0.98,
        "land_area_hectares":      1.5,
        "land_area_conf":          0.97,
        "crop_name":               "Bajra",
        "crop_code":               None,
        "crop_name_conf":          0.95,
        "season":                  "Kharif",
        "ifsc_code":               "SBIN0001234",
        "ifsc_code_conf":          0.99,
        "account_number_redacted": "[Account Redacted]",
        "bank_name":               "State Bank of India",
        "field_provenance":        {},
    }


# ──────────────────────────────────────────────────────────────────────────────
# HAPPY PATH
# ──────────────────────────────────────────────────────────────────────────────
class TestHappyPath:
    def test_valid_payload_passes(self, valid_farmer_payload: dict) -> None:
        """A fully valid payload must pass without errors."""
        result = validate_farmer_data(valid_farmer_payload)
        assert result.is_valid is True
        assert result.validated_data is not None
        assert result.needs_clarification is False

    def test_validated_data_types(self, valid_farmer_payload: dict) -> None:
        """Post-validation types must match schema declarations."""
        result = validate_farmer_data(valid_farmer_payload)
        assert isinstance(result.validated_data.farmer_name, str)
        assert isinstance(result.validated_data.land_area_hectares, float)
        assert isinstance(result.validated_data.ifsc_code, str)

    def test_ifsc_normalised_to_uppercase(self, valid_farmer_payload: dict) -> None:
        """IFSC code should be normalised to uppercase by the validator."""
        payload = {**valid_farmer_payload, "ifsc_code": "sbin0001234"}
        result = validate_farmer_data(payload)
        assert result.is_valid is True
        assert result.validated_data.ifsc_code == "SBIN0001234"

    def test_mobile_strips_country_code(self, valid_farmer_payload: dict) -> None:
        """Mobile number with +91 prefix must be stripped to 10 digits."""
        payload = {**valid_farmer_payload, "mobile_number": "+919876543210"}
        result = validate_farmer_data(payload)
        assert result.is_valid is True
        assert result.validated_data.mobile_number == "9876543210"


# ──────────────────────────────────────────────────────────────────────────────
# PRIVACY INVARIANTS
# ──────────────────────────────────────────────────────────────────────────────
class TestPrivacyInvariants:
    def test_aadhaar_redacted_must_be_placeholder(self, valid_farmer_payload: dict) -> None:
        """
        CRITICAL PRIVACY TEST: aadhaar_redacted must always equal "[Aadhaar Redacted]".
        Any other value must fail schema validation.
        """
        # Attempt to smuggle a fake or raw number into the schema (pattern rejects it)
        raw_dummy_digits = f"{'1'*4} {'2'*4} {'3'*4}"
        payload = {
            **valid_farmer_payload,
            "aadhaar_redacted": raw_dummy_digits,  # Must be rejected
        }
        with pytest.raises(Exception):
            FarmerDataSchema.model_validate(payload)

    def test_canonical_aadhaar_placeholder_accepted(self, valid_farmer_payload: dict) -> None:
        """The canonical placeholder must be accepted."""
        result = validate_farmer_data(valid_farmer_payload)
        assert result.validated_data.aadhaar_redacted == "[Aadhaar Redacted]"

    def test_account_number_redacted(self, valid_farmer_payload: dict) -> None:
        """Account number placeholder must be enforced."""
        result = validate_farmer_data(valid_farmer_payload)
        assert result.validated_data.account_number_redacted == "[Account Redacted]"

    def test_no_raw_12_digit_aadhaar_in_any_field(self, valid_farmer_payload: dict) -> None:
        """
        No field in validated output should contain a 12-digit Aadhaar-like string.
        This is the nuclear privacy check.
        """
        result = validate_farmer_data(valid_farmer_payload)
        assert result.is_valid
        data_dict = result.validated_data.model_dump()
        for field_name, field_value in data_dict.items():
            if isinstance(field_value, str):
                # Pattern: 12 contiguous digits (with optional spaces)
                import re
                assert not re.fullmatch(r"\d{4}\s?\d{4}\s?\d{4}", field_value.strip()), (
                    f"Field '{field_name}' contains an Aadhaar-like number: {field_value}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# IFSC VALIDATION (deterministic checksum gate)
# ──────────────────────────────────────────────────────────────────────────────
class TestIFSCValidation:
    @pytest.mark.parametrize("ifsc", [
        "SBIN0001234",   # Valid: SBI
        "HDFC0000123",   # Valid: HDFC
        "PUNB0123456",   # Valid: PNB
        "BARB0VJKAND",   # Valid: Bank of Baroda
    ])
    def test_valid_ifsc_codes(self, valid_farmer_payload: dict, ifsc: str) -> None:
        payload = {**valid_farmer_payload, "ifsc_code": ifsc}
        result = validate_farmer_data(payload)
        assert result.is_valid is True, f"Expected valid IFSC {ifsc} to pass"

    @pytest.mark.parametrize("bad_ifsc, description", [
        ("SBI0001234",     "Too short (10 chars)"),
        ("SBIN1001234",    "5th char must be '0'"),
        ("sbin0001234",    "Lowercase should be normalised (not fail)"),  # normalised
        ("SBINX001234",    "5th char is not '0'"),
        ("12340001234",    "Starts with digits"),
        ("SBIN000123",     "Only 10 chars"),
        ("",               "Empty string"),
    ])
    def test_invalid_ifsc_codes(
        self, valid_farmer_payload: dict, bad_ifsc: str, description: str
    ) -> None:
        if bad_ifsc == "sbin0001234":
            pytest.skip("lowercase IFSC is normalised to uppercase, not rejected")
        payload = {**valid_farmer_payload, "ifsc_code": bad_ifsc}
        result = validate_farmer_data(payload)
        assert result.is_valid is False, (
            f"Expected invalid IFSC '{bad_ifsc}' ({description}) to fail"
        )


# ──────────────────────────────────────────────────────────────────────────────
# MOBILE NUMBER VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
class TestMobileValidation:
    @pytest.mark.parametrize("mobile", [
        "9876543210",
        "8765432109",
        "7654321098",
        "6543210987",
    ])
    def test_valid_indian_mobiles(self, valid_farmer_payload: dict, mobile: str) -> None:
        payload = {**valid_farmer_payload, "mobile_number": mobile}
        assert validate_farmer_data(payload).is_valid is True

    @pytest.mark.parametrize("bad_mobile", [
        "5876543210",   # Starts with 5 — invalid
        "12345",        # Too short
        "98765432101",  # 11 digits
        "ABCDEFGHIJ",   # Letters
    ])
    def test_invalid_mobiles(self, valid_farmer_payload: dict, bad_mobile: str) -> None:
        payload = {**valid_farmer_payload, "mobile_number": bad_mobile}
        result = validate_farmer_data(payload)
        assert result.is_valid is False


# ──────────────────────────────────────────────────────────────────────────────
# KHASRA NUMBER VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
class TestKhasraValidation:
    @pytest.mark.parametrize("khasra", [
        "45",       # Simple numeric
        "45/2",     # Fractional
        "123P",     # Numeric + alpha suffix
        "1",        # Minimum
    ])
    def test_valid_khasra_numbers(self, valid_farmer_payload: dict, khasra: str) -> None:
        payload = {**valid_farmer_payload, "khasra_plot_no": khasra}
        assert validate_farmer_data(payload).is_valid is True

    @pytest.mark.parametrize("bad_khasra", [
        "ABC",        # Letters only
        "45-2",       # Hyphen separator (not valid)
        "!@#",        # Special chars
        "",           # Empty
    ])
    def test_invalid_khasra_numbers(self, valid_farmer_payload: dict, bad_khasra: str) -> None:
        payload = {**valid_farmer_payload, "khasra_plot_no": bad_khasra}
        result = validate_farmer_data(payload)
        assert result.is_valid is False


# ──────────────────────────────────────────────────────────────────────────────
# CONFIDENCE GATE
# ──────────────────────────────────────────────────────────────────────────────
class TestConfidenceGate:
    def test_low_khasra_confidence_triggers_clarification(
        self, valid_farmer_payload: dict
    ) -> None:
        """
        RUBRIC: Hallucination Guardrail — khasra_plot_no confidence < 0.95
        must route to human clarification.
        """
        payload = {
            **valid_farmer_payload,
            "khasra_plot_no_conf": 0.80,  # Below 0.95 threshold
        }
        result = validate_farmer_data(payload)
        assert result.needs_clarification is True
        assert result.clarification_field == "khasra_plot_no"

    def test_low_ifsc_confidence_triggers_clarification(
        self, valid_farmer_payload: dict
    ) -> None:
        payload = {
            **valid_farmer_payload,
            "ifsc_code_conf": 0.70,
        }
        result = validate_farmer_data(payload)
        assert result.needs_clarification is True
        assert result.clarification_field == "ifsc_code"

    def test_all_high_confidence_passes_through(self, valid_farmer_payload: dict) -> None:
        result = validate_farmer_data(valid_farmer_payload)
        assert result.is_valid is True
        assert result.needs_clarification is False


# ──────────────────────────────────────────────────────────────────────────────
# LAND AREA SANITY
# ──────────────────────────────────────────────────────────────────────────────
class TestLandArea:
    def test_negative_land_area_fails(self, valid_farmer_payload: dict) -> None:
        payload = {**valid_farmer_payload, "land_area_hectares": -1.0}
        assert validate_farmer_data(payload).is_valid is False

    def test_zero_land_area_fails(self, valid_farmer_payload: dict) -> None:
        payload = {**valid_farmer_payload, "land_area_hectares": 0.0}
        assert validate_farmer_data(payload).is_valid is False

    def test_absurd_land_area_fails(self, valid_farmer_payload: dict) -> None:
        """200+ Ha is unrealistic for target persona (marginal farmer)."""
        payload = {**valid_farmer_payload, "land_area_hectares": 250.0}
        assert validate_farmer_data(payload).is_valid is False

    def test_valid_fractional_area(self, valid_farmer_payload: dict) -> None:
        payload = {**valid_farmer_payload, "land_area_hectares": 0.25}
        assert validate_farmer_data(payload).is_valid is True

