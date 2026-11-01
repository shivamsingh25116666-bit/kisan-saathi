"""
tests/test_graph.py
──────────────────────────────────────────────────────────────────────────────
Unit and Integration tests for LangGraph state machine, OTP resumption pipeline,
duplicate submission prevention, and human clarification routing.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch
import pytest

from app.agent.graph import (
    build_kisan_graph,
    _route_extraction,
    _route_guardrail,
    _route_geo,
    _route_browser,
    _route_otp,
    audit_and_deliver_node,
    error_handler_node,
    human_clarification_node,
    resume_agent_pipeline_with_otp,
    run_agent_pipeline,
)
from app.agent.browser_worker import (
    _fuzzy_match_option,
    _generate_mock_pdf_receipt,
    _simulate_browser_otp_checkpoint,
    browser_resume_node,
)
from app.core.state import AgentState
from app.core.state_manager import state_manager


@pytest.fixture
def mock_base_state() -> AgentState:
    """Canonical test state adhering to the security directive."""
    return {
        "session_id": "test_session_graph_1234",
        "whatsapp_number": "whatsapp:+919876543210",
        "farmer_language": "hi",
        "audio_ogg_path": "",
        "image_paths": [],
        "gps_latitude": 25.7,
        "gps_longitude": 71.3,
        "transcribed_intent": "PMFBY Bajra",
        "farmer_data": {
            "farmer_name": "Ramchandra Yadav",
            "farmer_name_conf": 0.98,
            "aadhaar_redacted": "[Aadhaar Redacted]",
            "aadhaar_last4": "7823",
            "aadhaar_conf": 0.97,
            "mobile_number": "9876543210",
            "mobile_number_conf": 0.99,
            "khasra_plot_no": "45/2",
            "khasra_plot_no_conf": 0.98,
            "land_area_hectares": 1.5,
            "land_area_conf": 0.96,
            "crop_name": "Bajra",
            "crop_code": "CS001",
            "crop_name_conf": 0.95,
            "season": "Kharif",
            "ifsc_code": "SBIN0001234",
            "ifsc_code_conf": 0.99,
            "account_number_redacted": "[Account Redacted]",
            "bank_name": "State Bank of India",
            "field_provenance": {},
        },
        "needs_clarification": False,
        "otp_pending": False,
        "retry_count": 0,
        "audit_log": [],
        "error": "",
        "form_submission_status": "pending",
        "next_node": "multimodal_extraction",
    }


class TestGraphCompilation:
    def test_build_kisan_graph_compiles(self) -> None:
        """Verify the LangGraph cyclic state machine builds and compiles successfully."""
        graph = build_kisan_graph()
        assert graph is not None
        expected_nodes = {
            "multimodal_extraction",
            "guardrail_validation",
            "human_clarification",
            "geospatial_verification",
            "browser_automation",
            "otp_checkpoint",
            "browser_resume",
            "audit_and_deliver",
            "error_handler",
        }
        assert expected_nodes.issubset(set(graph.nodes.keys()))


class TestRoutingLogic:
    def test_route_extraction_happy_path(self, mock_base_state: AgentState) -> None:
        assert _route_extraction(mock_base_state) == "guardrail_validation"

    def test_route_extraction_low_confidence(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "needs_clarification": True}
        assert _route_extraction(state) == "human_clarification"

    def test_route_extraction_error(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "error": "Something went wrong"}
        assert _route_extraction(state) == "error_handler"

    def test_route_guardrail_happy_path(self, mock_base_state: AgentState) -> None:
        assert _route_guardrail(mock_base_state) == "geospatial_verification"

    def test_route_guardrail_clarification(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "needs_clarification": True}
        assert _route_guardrail(state) == "human_clarification"

    def test_route_geo_happy_path(self, mock_base_state: AgentState) -> None:
        assert _route_geo(mock_base_state) == "browser_automation"

    def test_route_browser_otp_pending(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "otp_pending": True}
        assert _route_browser(state) == "otp_checkpoint"

    def test_route_browser_submitted(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "form_submission_status": "submitted"}
        assert _route_browser(state) == "audit_and_deliver"

    def test_route_browser_simulation_completed(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "form_submission_status": "simulation_completed"}
        assert _route_browser(state) == "audit_and_deliver"

    def test_route_otp_resumes(self, mock_base_state: AgentState) -> None:
        state = {**mock_base_state, "otp_value": "123456"}
        assert _route_otp(state) == "browser_resume"


class TestHumanClarificationWorkflow:
    @pytest.mark.asyncio
    async def test_human_clarification_pauses_and_persists_state(self, mock_base_state: AgentState) -> None:
        state = dict(mock_base_state)
        state["clarification_field"] = "khasra_plot_no"
        res = await human_clarification_node(state)
        assert res["next_node"] == "__end__"

        persisted = await state_manager.get_session(state["session_id"])
        assert persisted is not None
        assert persisted["clarification_field"] == "khasra_plot_no"


class TestCrossMessageOTPResumptionPipeline:
    @pytest.mark.asyncio
    async def test_resume_pipeline_with_otp_success(self, mock_base_state: AgentState) -> None:
        import time
        session_id = "cross_msg_sess_001"
        state = dict(mock_base_state)
        state["session_id"] = session_id
        state["form_submission_status"] = "awaiting_otp"
        state["otp_pending"] = True
        state["otp_created_at"] = time.time()
        state["otp_attempts"] = 0
        state["is_simulation"] = True
        await state_manager.set_session(session_id, state, ttl_seconds=300)

        resumed_state = await resume_agent_pipeline_with_otp(session_id, "654321")
        assert resumed_state["form_submission_status"] in ("submitted", "simulation_completed")
        assert resumed_state["next_node"] == "__end__"
        assert resumed_state["receipt_pdf_path"] is not None

    @pytest.mark.asyncio
    async def test_duplicate_submission_prevention(self, mock_base_state: AgentState) -> None:
        """
        If session is already marked submitted, do not trigger duplicate browser or OTP workflows.
        """
        session_id = "duplicate_sess_002"
        state = dict(mock_base_state)
        state["session_id"] = session_id
        state["form_submission_status"] = "submitted"
        state["application_reference_no"] = "PMFBY/2026/CONFIRMED"
        await state_manager.set_session(session_id, state, ttl_seconds=300)

        # Attempt to resume with OTP on already submitted application
        res = await resume_agent_pipeline_with_otp(session_id, "654321")
        assert res["form_submission_status"] == "submitted"
        # Verify it remained submitted without changing
        assert res["application_reference_no"] == "PMFBY/2026/CONFIRMED"


class TestAuditAndReceiptDelivery:
    def test_generate_mock_pdf_receipt(self, mock_base_state: AgentState) -> None:
        pdf_path = _generate_mock_pdf_receipt("demo_sess_1", mock_base_state["farmer_data"])
        assert os.path.exists(pdf_path)
        with open(pdf_path, "rb") as f:
            raw_pdf = f.read()
        assert b"%PDF" in raw_pdf
        assert b"[Aadhaar Redacted]" in raw_pdf
        assert b"SIMULATION / NOT AN OFFICIAL GOVERNMENT RECEIPT" in raw_pdf

    @pytest.mark.asyncio
    async def test_audit_and_deliver_node_submitted(self, mock_base_state: AgentState) -> None:
        state = dict(mock_base_state)
        state["form_submission_status"] = "submitted"
        state["receipt_pdf_path"] = _generate_mock_pdf_receipt(state["session_id"], state["farmer_data"])
        result = await audit_and_deliver_node(state)
        assert result["next_node"] == "__end__"

    @pytest.mark.asyncio
    async def test_audit_and_deliver_node_simulation_completed(self, mock_base_state: AgentState) -> None:
        state = dict(mock_base_state)
        state["form_submission_status"] = "simulation_completed"
        state["receipt_pdf_path"] = _generate_mock_pdf_receipt(state["session_id"], state["farmer_data"])
        result = await audit_and_deliver_node(state)
        assert result["next_node"] == "__end__"
