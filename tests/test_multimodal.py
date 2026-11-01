"""
tests/test_multimodal.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for app/senses/multimodal.py

Tests cover:
  • Aadhaar redaction invariant (the nuclear privacy test).
  • Image pre-processing: compression stays below IMAGE_MAX_KB.
  • _looks_like_aadhaar() helper coverage.
  • _find_low_confidence_field() routing logic.

NOTE: Gemini and Bhashini API calls are mocked — no network required.
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pytest

from app.senses.multimodal import (
    _find_low_confidence_field,
    _looks_like_aadhaar,
    _redact_aadhaar,
    preprocess_document_image,
)
from app.core.config import get_settings

settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# PRIVACY: AADHAAR REDACTION
# ──────────────────────────────────────────────────────────────────────────────
class TestAadhaarRedaction:
    def test_canonical_placeholder_preserved(self) -> None:
        """Input with canonical placeholder must pass through unchanged."""
        data = {
            "aadhaar_redacted": "[Aadhaar Redacted]",
            "aadhaar_last4": "7823",
            "aadhaar_conf": 0.96,
        }
        result = _redact_aadhaar(data)
        assert result["aadhaar_redacted"] == "[Aadhaar Redacted]"

    def test_raw_aadhaar_in_any_field_gets_redacted(self) -> None:
        """
        Even if the model (hypothetically) returns a 12-digit number in any
        field, _redact_aadhaar must replace it with the placeholder.
        PRIVACY: This test is the red-team check for model misbehaviour.
        """
        raw_dummy_digits = f"{'1'*4} {'2'*4} {'3'*4}"
        data = {
            "aadhaar_redacted": "[Aadhaar Redacted]",  # Correct
            "some_other_field": raw_dummy_digits,       # Should be redacted
            "aadhaar_conf": 0.96,
        }
        result = _redact_aadhaar(data)
        assert result["some_other_field"] == "[Aadhaar Redacted]"

    def test_account_placeholder_enforced(self) -> None:
        data = {"aadhaar_redacted": "[Aadhaar Redacted]"}
        result = _redact_aadhaar(data)
        assert result["account_number_redacted"] == "[Account Redacted]"


# ──────────────────────────────────────────────────────────────────────────────
# _looks_like_aadhaar() HELPER
# ──────────────────────────────────────────────────────────────────────────────
class TestLooksLikeAadhaar:
    @pytest.mark.parametrize("value", [
        f"{'1'*4} {'2'*4} {'3'*4}",
        f"{'1'*12}",
        f"{'1'*4} {'2'*8}",
    ])
    def test_detects_aadhaar_patterns(self, value: str) -> None:
        assert _looks_like_aadhaar(value) is True

    @pytest.mark.parametrize("value", [
        "[Aadhaar Redacted]",
        "7823",               # Last 4 only
        "SBIN0001234",        # IFSC
        "9876543210",         # Mobile
        "45/2",               # Khasra
        "",
    ])
    def test_rejects_non_aadhaar(self, value: str) -> None:
        assert _looks_like_aadhaar(value) is False


# ──────────────────────────────────────────────────────────────────────────────
# CONFIDENCE FIELD DETECTION
# ──────────────────────────────────────────────────────────────────────────────
class TestFindLowConfidenceField:
    def test_all_high_confidence_returns_none(self) -> None:
        data = {
            "khasra_plot_no":      "45/2",
            "khasra_plot_no_conf": 0.97,
            "land_area_hectares":  1.5,
            "land_area_conf":      0.96,
            "ifsc_code":           "SBIN0001234",
            "ifsc_code_conf":      0.98,
            "farmer_name":         "Ramchandra Yadav",
            "farmer_name_conf":    0.97,
        }
        assert _find_low_confidence_field(data) is None  # type: ignore

    def test_low_khasra_conf_detected_first(self) -> None:
        """khasra_plot_no is highest priority — should be caught before other fields."""
        data = {
            "khasra_plot_no":      "45/2",
            "khasra_plot_no_conf": 0.80,   # Low
            "land_area_hectares":  1.5,
            "land_area_conf":      0.75,   # Also low, but khasra is higher priority
            "ifsc_code":           "SBIN0001234",
            "ifsc_code_conf":      0.99,
            "farmer_name":         "Ram",
            "farmer_name_conf":    0.99,
        }
        result = _find_low_confidence_field(data)  # type: ignore
        assert result == "khasra_plot_no"

    def test_null_field_skipped(self) -> None:
        """If a field value is None, skip it (it was not found in the image)."""
        data = {
            "khasra_plot_no":      None,  # Not found — validator handles this
            "khasra_plot_no_conf": 0.0,
            "ifsc_code":           "SBIN0001234",
            "ifsc_code_conf":      0.99,
            "farmer_name":         "Ram",
            "farmer_name_conf":    0.97,
            "land_area_hectares":  1.0,
            "land_area_conf":      0.96,
        }
        result = _find_low_confidence_field(data)  # type: ignore
        assert result is None  # None field is skipped; no routing to clarification


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE PRE-PROCESSING (uses temp file, no API calls)
# ──────────────────────────────────────────────────────────────────────────────
class TestImagePreprocessing:
    def _create_test_image(self, size_kb: int = 200) -> str:
        """Create a synthetic large JPEG image for testing compression."""
        import cv2

        # Generate a random 800x600 BGR image
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        # Draw some fake document-like lines
        cv2.line(img, (0, 100), (800, 100), (0, 0, 0), 2)
        cv2.line(img, (0, 200), (800, 200), (0, 0, 0), 2)
        cv2.putText(img, "TEST DOCUMENT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        tmp.close()
        return tmp.name

    def test_output_below_max_kb(self) -> None:
        """Compressed output must be below IMAGE_MAX_KB."""
        input_path = self._create_test_image()
        try:
            output_path = preprocess_document_image(input_path)
            output_size_kb = os.path.getsize(output_path) / 1024
            assert output_size_kb <= settings.image_max_kb, (
                f"Output size {output_size_kb:.1f} KB exceeds limit {settings.image_max_kb} KB"
            )
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_output_is_valid_jpeg(self) -> None:
        """Output must be a valid JPEG file."""
        import cv2

        input_path = self._create_test_image()
        try:
            output_path = preprocess_document_image(input_path)
            loaded = cv2.imread(output_path)
            assert loaded is not None, "Output JPEG is not a valid image"
            assert loaded.shape[0] > 0 and loaded.shape[1] > 0
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_invalid_image_path_raises(self) -> None:
        """Non-existent image path must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot read image"):
            preprocess_document_image("/nonexistent/path/to/image.jpg")

