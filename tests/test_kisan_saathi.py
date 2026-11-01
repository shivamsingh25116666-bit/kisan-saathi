"""
tests/test_kisan_saathi.py
──────────────────────────────────────────────────────────────────────────────
Unit and Integration tests for Kisan Saathi Web & Mobile Application APIs.

Tests:
  1. Multilingual language detection across Indian scripts and regional dialects.
  2. Dynamic task planner decomposition (no hardcoded chatbots).
  3. Extensible tool registry execution.
  4. Services discovery catalog API.
  5. Conversational chat API with structured planning response.
  6. Document upload, OCR parsing, and strict Aadhaar privacy redaction.
  7. End-to-end PMFBY crop insurance application with OTP checkpoint.
  8. Static asset serving for web frontend.
"""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from app.agent.planner import detect_language, planner
from app.agent.tools import registry
from app.api.webhook import app

client = TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# 1. MULTILINGUAL LANGUAGE & DIALECT DETECTION
# ──────────────────────────────────────────────────────────────────────────────
class TestMultilingualDetection:
    def test_detect_punjabi_gurmukhi(self):
        text = "ਮੈਨੂੰ ਆਪਣੀ ਕਣਕ ਦੀ ਫਸਲ ਦਾ ਬੀਮਾ ਕਰਵਾਉਣਾ ਹੈ"
        assert detect_language(text) == "pa"

    def test_detect_bengali(self):
        text = "আমি আমার ধানের বীমা করতে চাই"
        assert detect_language(text) == "bn"

    def test_detect_marathi(self):
        text = "मला माझ्या पिकाचा विमा उतरवायचा आहे"
        assert detect_language(text) == "mr"

    def test_detect_gujarati(self):
        text = "મારે મારા પાકનો વીમો કરાવવો છે"
        assert detect_language(text) == "gu"

    def test_detect_tamil(self):
        text = "எனது பயிர் காப்பீடு செய்ய வேண்டும்"
        assert detect_language(text) == "ta"

    def test_detect_telugu(self):
        text = "నా పంట బీమా చేయాలనుకుంటున్నాను"
        assert detect_language(text) == "te"

    def test_detect_kannada(self):
        text = "ನನ್ನ ಬೆಳೆ ವಿಮೆ ಮಾಡಿಸಬೇಕು"
        assert detect_language(text) == "kn"

    def test_detect_malayalam(self):
        text = "എനിക്ക് വിള ഇൻഷുറൻസ് എടുക്കണം"
        assert detect_language(text) == "ml"

    def test_detect_odia(self):
        text = "ମୋର ଫସଲ ବୀମା କରିବାକୁ ଚାହେଁ"
        assert detect_language(text) == "or"

    def test_detect_garhwali_dialect(self):
        text = "म्यार बाजरा का खेत मा कीड़ा लगि गे, कख जाऊं?"
        assert detect_language(text) == "gar"

    def test_detect_hindi_default(self):
        text = "मुझे अपनी बाजरे की फसल का बीमा करवाना है"
        assert detect_language(text) == "hi"

    def test_detect_english(self):
        text = "I need to apply for crop insurance for my wheat field"
        assert detect_language(text) == "en"


# ──────────────────────────────────────────────────────────────────────────────
# 2. DYNAMIC TASK PLANNER & EXTENSIBLE TOOL REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
class TestDynamicTaskPlanner:
    @pytest.mark.asyncio
    async def test_plan_crop_insurance_intent(self):
        plan = await planner.create_plan(
            "मुझे राजस्थान में अपने बाजरे का बीमा करवाना है",
            task_id="plan_test_001",
        )
        assert plan.category == "crop_insurance"
        assert plan.extracted_facts.get("crop_name") == "Bajra"
        assert plan.extracted_facts.get("state") == "Rajasthan"
        assert len(plan.steps) >= 4
        # Verify first step has PMFBY details tool
        assert plan.steps[0].tool_to_execute == "crop_insurance_details"

    @pytest.mark.asyncio
    async def test_plan_mandi_prices_intent(self):
        plan = await planner.create_plan(
            "आज गेहूं का मंडी भाव क्या चल रहा है?",
            task_id="plan_test_002",
        )
        assert plan.category == "market_prices"
        assert plan.steps[0].tool_to_execute == "mandi_prices"

    @pytest.mark.asyncio
    async def test_plan_pm_kisan_intent(self):
        plan = await planner.create_plan(
            "मेरी पीएम किसान सम्मान निधि की 2000 की किस्त कब आएगी?",
            task_id="plan_test_003",
        )
        assert plan.category == "pm_kisan"
        assert plan.steps[0].tool_to_execute == "pm_kisan_advisory"

    @pytest.mark.asyncio
    async def test_plan_kcc_loan_intent(self):
        plan = await planner.create_plan(
            "मुझे 2 हेक्टेयर जमीन पर केसीसी ऋण चाहिए",
            task_id="plan_test_004",
        )
        assert plan.category == "credit_and_loans"
        assert plan.steps[0].tool_to_execute == "kcc_loan_advisory"

    @pytest.mark.asyncio
    async def test_tool_registry_execution(self):
        tool = registry.get("crop_insurance_details")
        assert tool is not None
        res = await tool.execute(crop_name="Bajra", state="Rajasthan", season="Kharif")
        assert res["success"] is True
        assert res["data"]["crop"] == "Bajra"
        assert res["data"]["farmer_premium_share_percent"] == 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. KISAN SAATHI REST APIS
# ──────────────────────────────────────────────────────────────────────────────
class TestKisanSaathiApis:
    def test_languages_endpoint(self):
        resp = client.get("/api/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert "hi" in data["languages"]
        assert "pa" in data["languages"]
        assert "bn" in data["languages"]
        assert "mr" in data["languages"]

    def test_services_catalog_endpoint(self):
        resp = client.get("/api/services")
        assert resp.status_code == 200
        services = resp.json()["services"]
        assert len(services) >= 6
        ids = [s["id"] for s in services]
        assert "pmfby_crop_insurance" in ids
        assert "pm_kisan" in ids
        assert "kisan_credit_card" in ids
        assert "mandi_prices" in ids

    def test_chat_endpoint_generates_plan_and_reply(self):
        payload = {
            "query": "मुझे अपनी बाजरे की फसल का बीमा करवाना है",
            "session_id": "test_web_chat_001",
            "language": "hi",
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "crop_insurance"
        assert "प्रीमियम" in data["reply_text"] or "बीमा" in data["reply_text"]
        assert data["plan"] is not None
        assert len(data["plan"]["steps"]) >= 3

    def test_document_upload_and_privacy_redaction(self):
        # Create a mock image
        mock_file = io.BytesIO(b"\xFF\xD8\xFF\xE0" + b"\x00" * 200)
        files = {"file": ("khasra_doc.jpg", mock_file, "image/jpeg")}
        data = {
            "session_id": "test_vault_sess_001",
            "doc_type": "khatauni",
        }

        resp = client.post("/api/documents/upload", files=files, data=data)
        assert resp.status_code == 200
        res = resp.json()
        assert res["success"] is True
        doc = res["document"]
        assert doc["extracted_data"]["national_id"] == "[Aadhaar Redacted]"
        assert len(doc["extracted_data"]["aadhaar_last4"]) == 4

    def test_document_vault_listing(self):
        resp = client.get("/api/documents/vault?session_id=test_vault_sess_001")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert len(data["documents"]) >= 1

    def test_insurance_start_application_and_otp_workflow(self):
        session_id = "test_ins_flow_001"
        start_payload = {
            "session_id": session_id,
            "farmer_name": "रामचन्द्र वर्मा",
            "crop_name": "Bajra",
            "khasra_plot_no": "142/2",
            "land_area_hectares": 1.85,
            "ifsc_code": "SBIN0001234",
            "bank_account_last4": "9876",
            "state": "Rajasthan",
            "district": "Barmer",
            "sub_district": "Sindhari",
            "village": "Sindhari",
            "aadhaar_last4": "7823",
        }

        # Step 1: Initiate PMFBY application
        resp1 = client.post("/api/insurance/start-application", json=start_payload)
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "awaiting_otp"

        # Step 2: Submit 6-digit OTP
        otp_payload = {
            "session_id": session_id,
            "otp": "847291",
        }
        resp2 = client.post("/api/insurance/submit-otp", json=otp_payload)
        assert resp2.status_code == 200
        result = resp2.json()
        assert result["success"] is True
        assert result["submission_status"] in ("simulation_completed", "submitted")
        assert "application_reference_no" in result
        if result["is_simulation"]:
            assert "NOTICE: DEMO SIMULATION ONLY" in result["watermark_notice"]

    def test_static_web_frontend_served(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "किसान साथी" in resp.text
        assert "Kisan Saathi" in resp.text

