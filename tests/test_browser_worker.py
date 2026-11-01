"""
tests/test_browser_worker.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for browser_worker.py covering:
  • Self-healing dropdown fuzzy matching without blind guesses.
  • Genuine vs. Simulation vs. Failed status differentiation.
  • Portal confirmation verification.
  • Honest receipt download failure reporting.
  • Watermark assertion on generated simulation receipts.
  • Safe document upload handling (missing/invalid files).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.agent.browser_worker import (
    _fuzzy_match_option,
    _generate_mock_pdf_receipt,
    _simulate_browser_otp_checkpoint,
    _verify_submission_confirmation,
    browser_automation_node,
    browser_resume_node,
)
from app.core.config import get_settings
from app.core.state import AgentState

settings = get_settings()


@pytest.fixture
def base_test_state() -> AgentState:
    return {
        "session_id": "test_bw_001",
        "whatsapp_number": "whatsapp:+919876543210",
        "farmer_language": "hi",
        "farmer_data": {
            "farmer_name": "Ramchandra Yadav",
            "mobile_number": "9876543210",
            "khasra_plot_no": "45/2",
            "land_area_hectares": 1.5,
            "crop_name": "Bajra",
            "season": "Kharif",
            "ifsc_code": "SBIN0001234",
            "aadhaar_redacted": "[Aadhaar Redacted]",
            "account_number_redacted": "[Account Redacted]",
        },
        "admin_hierarchy": {
            "state_name": "Rajasthan",
            "district_name": "Barmer",
            "sub_district_name": "Sindhari",
            "village_name": "Sindhari",
        },
        "image_paths": [],
        "form_submission_status": "pending",
        "is_simulation": False,
        "otp_pending": False,
        "retry_count": 0,
        "audit_log": [],
        "error": "",
    }


class TestDropdownMatchingWithoutBlindGuessing:
    def test_exact_match(self) -> None:
        options = ["Sindhari", "Mohol", "Fatehganj"]
        assert _fuzzy_match_option("Sindhari", options) == "Sindhari"
        assert _fuzzy_match_option("sindhari", options) == "Sindhari"

    def test_unique_substring_match(self) -> None:
        options = ["Sindhari (Gram)", "Mohol (Rural)", "Fatehganj West"]
        assert _fuzzy_match_option("Sindhari", options) == "Sindhari (Gram)"
        assert _fuzzy_match_option("Mohol", options) == "Mohol (Rural)"

    def test_ambiguous_or_unmatched_never_picks_first(self) -> None:
        """
        CRITICAL ARCHITECTURAL RULE:
        Do not automatically select the first village or option when unmatched.
        """
        options = ["Sindhari North", "Sindhari South", "Mohol"]
        # Multiple matches for "Sindhari" -> ambiguous, must return None
        assert _fuzzy_match_option("Sindhari", options) is None

        # Zero matches -> must return None, NEVER options[0]!
        assert _fuzzy_match_option("Unknown Village", options) is None
        assert _fuzzy_match_option("", options) is None
        assert _fuzzy_match_option("Target", []) is None


class TestSimulationWatermarkingAndHonestStatuses:
    def test_simulation_pdf_receipt_watermark(self, base_test_state: AgentState) -> None:
        """
        Verifies that mock receipts contain the required prominent watermark
        and do NOT claim to be official government documents.
        """
        pdf_path = _generate_mock_pdf_receipt("demo_sess_watermark", base_test_state["farmer_data"])
        assert os.path.exists(pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Check required disclaimer watermark
        assert b"SIMULATION / NOT AN OFFICIAL GOVERNMENT RECEIPT" in pdf_bytes
        assert b"DISCLAIMER: Generated in sandbox simulation mode" in pdf_bytes
        # Invariant: Privacy
        assert b"[Aadhaar Redacted]" in pdf_bytes

    @pytest.mark.asyncio
    async def test_simulated_session_completion_sets_honest_status(self, base_test_state: AgentState) -> None:
        """
        A simulated demo flow must be marked as 'simulation_completed', NEVER 'submitted'.
        """
        state = dict(base_test_state)
        state = _simulate_browser_otp_checkpoint(state)
        assert state["form_submission_status"] == "awaiting_otp"
        assert state["is_simulation"] is True

        # Provide valid OTP to resume simulated session
        state["otp_value"] = "998877"
        resumed = await browser_resume_node(state)

        assert resumed["form_submission_status"] == "simulation_completed"
        assert resumed["is_simulation"] is True
        assert resumed["receipt_pdf_path"] is not None
        assert resumed["next_node"] == "audit_and_deliver"

    @pytest.mark.asyncio
    async def test_invalid_otp_format_fails_honestly(self, base_test_state: AgentState) -> None:
        state = dict(base_test_state)
        state["otp_value"] = "12"  # Invalid: not 6 digits
        resumed = await browser_resume_node(state)
        assert resumed["form_submission_status"] == "failed"
        assert "Invalid" in resumed["error"]
        assert resumed["next_node"] == "error_handler"


class TestPortalConfirmationVerification:
    @pytest.mark.asyncio
    async def test_verify_submission_confirmation_success(self) -> None:
        mock_page = MagicMock()
        mock_el = MagicMock()
        mock_el.is_visible = AsyncMock(return_value=True)
        mock_el.text_content = AsyncMock(return_value="PMFBY-RJ-2026-99238")
        mock_page.wait_for_selector = AsyncMock(return_value=mock_el)

        confirmed, ref = await _verify_submission_confirmation(mock_page)
        assert confirmed is True
        assert ref == "PMFBY-RJ-2026-99238"

    @pytest.mark.asyncio
    async def test_verify_submission_confirmation_error_banner(self) -> None:
        mock_page = MagicMock()

        async def selector_side_effect(sel, timeout=1000):
            if "#application_success" in sel:
                return None
            if "#otp_error" in sel:
                err_el = MagicMock()
                err_el.is_visible = AsyncMock(return_value=True)
                err_el.text_content = AsyncMock(return_value="Invalid OTP entered")
                return err_el
            return None

        mock_page.wait_for_selector = AsyncMock(side_effect=selector_side_effect)

        confirmed, ref = await _verify_submission_confirmation(mock_page)
        assert confirmed is False
        assert ref is None


class TestBrowserLaunchAndNavigationFailures:
    @pytest.mark.asyncio
    async def test_browser_automation_handles_portal_unreachable_in_simulation_mode(
        self, base_test_state: AgentState
    ) -> None:
        state = dict(base_test_state)
        # In demo/simulation mode (enable_portal_simulation=True), falling back to simulation is allowed
        with patch.object(settings, "enable_portal_simulation", True):
            # When playwright fails launch or navigation
            with patch("app.agent.browser_worker._navigate_with_retry", side_effect=Exception("Portal unreachable")):
                res = await browser_automation_node(state)
                assert res["form_submission_status"] == "awaiting_otp"
                assert res["is_simulation"] is True
                assert res["otp_pending"] is True
                assert res["next_node"] == "otp_checkpoint"

    @pytest.mark.asyncio
    async def test_browser_automation_fails_honestly_when_simulation_disabled(
        self, base_test_state: AgentState
    ) -> None:
        state = dict(base_test_state)
        with patch.object(settings, "enable_portal_simulation", False):
            with patch("app.agent.browser_worker.async_playwright") as mock_pw:
                mock_pw.side_effect = RuntimeError("Headless chromium binary missing")
                res = await browser_automation_node(state)
                assert res["form_submission_status"] == "failed"
                assert "failed" in res["error"].lower() and "browser" in res["error"].lower()
                assert res["next_node"] == "error_handler"


class TestDocumentUploadValidation:
    @pytest.mark.asyncio
    async def test_upload_missing_file_raises_error(self) -> None:
        from app.agent.browser_worker import _upload_documents
        mock_page = MagicMock()
        with pytest.raises(FileNotFoundError):
            await _upload_documents(mock_page, ["/tmp/non_existent_file_xyz.jpg"])

    @pytest.mark.asyncio
    async def test_upload_oversized_file_raises_error(self, tmp_path: Any) -> None:
        from app.agent.browser_worker import _upload_documents
        mock_page = MagicMock()
        oversized = tmp_path / "huge_doc.jpg"
        # Write > 95 KB
        oversized.write_bytes(b"0" * (120 * 1024))
        with pytest.raises(ValueError) as exc:
            await _upload_documents(mock_page, [str(oversized)])
        assert "exceeds maximum portal upload limit" in str(exc.value)

