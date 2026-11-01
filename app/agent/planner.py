"""
app/agent/planner.py
──────────────────────────────────────────────────────────────────────────────
Autonomous Multilingual Task Planner & Execution Engine for Kisan Saathi.

Responsibilities:
1. Understand farmer intent across languages, dialects and informal speech.
2. Detect supported Indian languages.
3. Extract crop, location and land-area information.
4. Generate structured task plans.
5. Execute registered agricultural tools.
6. Maintain farmer approval checkpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
from typing import Any, Optional

from app.agent.tools import registry
from app.core.config import get_settings
from app.core.state import AgentState
from app.core.state_manager import state_manager

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "hi": {"name": "Hindi", "native": "हिन्दी", "locale": "hi-IN"},
    "en": {"name": "English", "native": "English", "locale": "en-IN"},
    "gar": {"name": "Garhwali", "native": "गढ़वाली", "locale": "hi-IN"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "locale": "pa-IN"},
    "mr": {"name": "Marathi", "native": "मराठी", "locale": "mr-IN"},
    "bn": {"name": "Bengali", "native": "বাংলা", "locale": "bn-IN"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "locale": "gu-IN"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "locale": "ta-IN"},
    "te": {"name": "Telugu", "native": "తెలుగు", "locale": "te-IN"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "locale": "kn-IN"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "locale": "ml-IN"},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ", "locale": "or-IN"},
    "as": {"name": "Assamese", "native": "অসমীয়া", "locale": "as-IN"},
}


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language using Indian script ranges and simple word markers."""

    if not text:
        return "hi"

    # Punjabi: Gurmukhi
    for char in text:
        code = ord(char)

        if 0x0A00 <= code <= 0x0A7F:
            return "pa"

        if 0x0980 <= code <= 0x09FF:
            if any(word in text for word in ("অসম", "ভাল", "কৰা")):
                return "as"
            return "bn"

        if 0x0A80 <= code <= 0x0AFF:
            return "gu"

        if 0x0B00 <= code <= 0x0B7F:
            return "or"

        if 0x0B80 <= code <= 0x0BFF:
            return "ta"

        if 0x0C00 <= code <= 0x0C7F:
            return "te"

        if 0x0C80 <= code <= 0x0CFF:
            return "kn"

        if 0x0D00 <= code <= 0x0D7F:
            return "ml"

    text_lower = text.lower()

    # Garhwali markers
    garhwali_words = (
        "कख", "कैंक", "म्यार", "कथगा",
        "ह्वै", "रौं", "दाज्यू", "भुलू",
    )

    if any(word in text_lower for word in garhwali_words):
        return "gar"

    # Marathi markers
    marathi_words = (
        "आहे", "नाही", "पाहिजे",
        "शेतकरी", "पिक", "करा", "सांगा",
    )

    if any(word in text_lower for word in marathi_words):
        return "mr"

    # English detection
    has_devanagari = any(
        0x0900 <= ord(char) <= 0x097F
        for char in text
    )

    if not has_devanagari:
        english_words = {
            "insurance", "claim", "crop", "farmer",
            "subsidy", "loan", "weather", "soil",
            "price", "hello", "help", "need",
        }

        words = set(re.findall(r"\b[a-zA-Z]+\b", text_lower))

        if words:
            matching_words = words.intersection(english_words)

            if len(matching_words) >= 2:
                return "en"

            if len(matching_words) / len(words) > 0.4:
                return "en"

    return "hi"


# ─────────────────────────────────────────────────────────────────────────────
# TASK STEP
# ─────────────────────────────────────────────────────────────────────────────

class TaskStep:

    def __init__(
        self,
        step_id: int,
        title: str,
        description: str,
        status: str = "pending",
        tool_to_execute: Optional[str] = None,
        tool_params: Optional[dict[str, Any]] = None,
        requires_farmer_approval: bool = False,
    ):
        self.step_id = step_id
        self.title = title
        self.description = description
        self.status = status
        self.tool_to_execute = tool_to_execute
        self.tool_params = tool_params or {}
        self.requires_farmer_approval = requires_farmer_approval
        self.result: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "tool_to_execute": self.tool_to_execute,
            "tool_params": self.tool_params,
            "requires_farmer_approval": self.requires_farmer_approval,
            "result": self.result,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION PLAN
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionPlan:

    def __init__(
        self,
        task_id: str,
        goal: str,
        category: str,
        language: str,
        steps: list[TaskStep],
        extracted_facts: dict[str, Any],
        missing_info: list[str],
        clarification_prompt: Optional[str] = None,
    ):
        self.task_id = task_id
        self.goal = goal
        self.category = category
        self.language = language
        self.steps = steps
        self.extracted_facts = extracted_facts
        self.missing_info = missing_info
        self.clarification_prompt = clarification_prompt
        self.status = "active"
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "category": self.category,
            "language": self.language,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "extracted_facts": self.extracted_facts,
            "missing_info": self.missing_info,
            "clarification_prompt": self.clarification_prompt,
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL AGENT PLANNER
# ─────────────────────────────────────────────────────────────────────────────

class GeneralAgentPlanner:

    def __init__(self):
        self._active_plans: dict[str, ExecutionPlan] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # CROP, STATE AND LAND-AREA EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def extract_crop_and_location(
        self,
        query: str,
    ) -> dict[str, Any]:

        q = query.lower()
        facts: dict[str, Any] = {}

        crops = {
            "wheat": [
                "wheat", "gehun", "gehu", "कनक", "गेहूं", "गेहूँ",
            ],
            "bajra": [
                "bajra", "pearl millet", "बाजरा", "बाजरी", "बाजरे",
            ],
            "paddy": [
                "paddy", "rice", "dhan", "चावल", "धान", "चांवल",
            ],
            "mustard": [
                "mustard", "sarson", "sarso", "सरसों", "राई", "तोरी",
            ],
            "cotton": [
                "cotton", "kapas", "कापूस", "कपास",
            ],
            "maize": [
                "maize", "corn", "makka", "मक्का", "मका", "मक्के",
            ],
            "soybean": [
                "soybean", "soya", "सोयाबीन",
            ],
            "gram": [
                "gram", "chana", "चना", "हरभरा", "चने",
            ],
            "onion": [
                "onion", "pyaz", "कांदा", "प्याज",
            ],
            "potato": [
                "potato", "aloo", "बटाटा", "आलू",
            ],
        }

        for canonical_name, synonyms in crops.items():
            if any(word in q for word in synonyms):
                facts["crop_name"] = canonical_name.title()
                break

        states = {
            "Rajasthan": ["rajasthan", "राजस्थान"],
            "Maharashtra": ["maharashtra", "महाराष्ट्र"],
            "Uttar Pradesh": ["uttar pradesh", "उत्तर प्रदेश"],
            "Punjab": ["punjab", "पंजाब"],
            "Haryana": ["haryana", "हरियाणा"],
            "Madhya Pradesh": ["madhya pradesh", "मध्य प्रदेश"],
            "Gujarat": ["gujarat", "गुजरात"],
            "Bihar": ["bihar", "बिहार"],
            "Karnataka": ["karnataka", "कर्नाटक"],
            "Tamil Nadu": ["tamil nadu", "तमिलनाडु"],
            "West Bengal": ["bengal", "west bengal", "बंगाल"],
            "Uttarakhand": ["uttarakhand", "उत्तराखंड"],
        }

        for canonical_state, synonyms in states.items():
            if any(word in q for word in synonyms):
                facts["state"] = canonical_state
                break

        area_match = re.search(
            r"(\d+(?:\.\d+)?)\s*"
            r"(bigha|hectare|ha|acre|बीघा|एकड़|हेक्टेयर)",
            q,
        )

        if area_match:
            value = float(area_match.group(1))
            unit = area_match.group(2).lower()

            if unit in ("bigha", "बीघा"):
                facts["land_area_ha"] = round(value * 0.25, 2)

            elif unit in ("acre", "एकड़"):
                facts["land_area_ha"] = round(value * 0.4047, 2)

            else:
                facts["land_area_ha"] = value

        return facts

    # ─────────────────────────────────────────────────────────────────────────
    # CREATE PLAN
    # ─────────────────────────────────────────────────────────────────────────

    async def create_plan(
        self,
        query: str,
        task_id: str,
        language: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ExecutionPlan:

        """Understand a farmer query and generate an executable task plan."""

        lang = language or detect_language(query)
        q_lower = query.lower()

        # Normalize spaces, hyphens, underscores and slashes.
        # This makes PM-KISAN, PM KISAN and PM_KISAN match consistently.
        intent_text = re.sub(
            r"[\s\-_/]+",
            " ",
            q_lower,
        ).strip()

        extracted = self.extract_crop_and_location(query)

        if context:
            extracted.update({
                key: value
                for key, value in context.items()
                if value
            })

        missing_info: list[str] = []
        steps: list[TaskStep] = []
        clarification: Optional[str] = None
        category = "general"

        # ─────────────────────────────────────────────────────────────────────
        # 1. PMFBY CROP INSURANCE
        # ─────────────────────────────────────────────────────────────────────

        if any(term in intent_text for term in (
            "insurance",
            "bima",
            "बीमा",
            "pmfby",
            "fasal bima",
            "क्लेम",
            "claim",
            "फसल बीमा",
        )):

            category = "crop_insurance"

            crop = extracted.get("crop_name")
            state = extracted.get("state")

            if not crop:
                missing_info.append("crop_name")

            if not state:
                missing_info.append("state")

            steps.append(TaskStep(
                step_id=1,
                title="Scheme Rules & Premium Calculation",
                description=(
                    "Retrieve verified PMFBY insurance terms, farmer premium "
                    "share (1.5-2%), and official cutoff date."
                ),
                status="pending",
                tool_to_execute="crop_insurance_details",
                tool_params={
                    "crop_name": crop or "Wheat",
                    "state": state or "Rajasthan",
                    "season": "Kharif",
                },
            ))

            steps.append(TaskStep(
                step_id=2,
                title="Required Document Intake & OCR Verification",
                description=(
                    "Upload Aadhaar, Bank Passbook, and Khatauni land record. "
                    "Auto-extract plot number, area, and IFSC code."
                ),
            ))

            steps.append(TaskStep(
                step_id=3,
                title="Review Application & Obtain Farmer Approval",
                description=(
                    "Present calculated premium, land details, and official "
                    "terms for farmer verification. Require explicit approval."
                ),
                requires_farmer_approval=True,
            ))

            steps.append(TaskStep(
                step_id=4,
                title="Official Portal Registration & OTP Verification",
                description=(
                    "Launch authorized PMFBY portal registration, fill "
                    "verified fields, and request legitimate OTP verification."
                ),
                requires_farmer_approval=True,
            ))

            steps.append(TaskStep(
                step_id=5,
                title="Submission Confirmation & Receipt Delivery",
                description=(
                    "Verify official government portal acknowledgment number "
                    "and generate an audit-trail summary."
                ),
            ))

            if "crop_name" in missing_info:
                clarification = (
                    "🌾 आप कौन सी फसल का बीमा करवाना चाहते हैं? "
                    "जैसे: बाजरा, गेहूं या धान।"
                    if lang in ("hi", "gar")
                    else
                    "🌾 Which crop do you wish to insure? "
                    "For example: Bajra, Wheat or Paddy."
                )

        # ─────────────────────────────────────────────────────────────────────
        # 2. PM-KISAN
        # ─────────────────────────────────────────────────────────────────────

        elif any(term in intent_text for term in (
            "kisan samman",
            "kisan samman nidhi",
            "pm kisan",
            "pmkisan",
            "2000",
            "installment",
            "installments",
            "kisaan",
            "किस्त",
            "सम्मान निधि",
            "पीएम किसान",
            "पीएम-किसान",
            "प्रधानमंत्री किसान",
            "प्रधानमंत्री सम्मान निधि",
            "किसान सम्मान निधि",
        )):

            category = "pm_kisan"

            steps.append(TaskStep(
                step_id=1,
                title="Check PM-KISAN Scheme Guidelines & Eligibility",
                description=(
                    "Retrieve PM-KISAN scheme guidelines, annual income "
                    "support details, DBT bank seeding rules and e-KYC steps."
                ),
                status="pending",
                tool_to_execute="pm_kisan_advisory",
                tool_params={
                    "state": extracted.get("state", "Rajasthan"),
                },
            ))

            steps.append(TaskStep(
                step_id=2,
                title="Verify Land Seeding & Aadhaar DBT Linkage",
                description=(
                    "Guide the farmer to verify active NPCI bank mapping "
                    "and land seeding on the official PM-KISAN portal."
                ),
            ))

        # ─────────────────────────────────────────────────────────────────────
        # 3. KISAN CREDIT CARD
        # ─────────────────────────────────────────────────────────────────────

        elif any(term in intent_text for term in (
            "kcc",
            "loan",
            "kredit",
            "credit card",
            "ऋण",
            "लोन",
            "कर्ज",
            "किसान क्रेडिट कार्ड",
        )):

            category = "credit_and_loans"

            steps.append(TaskStep(
                step_id=1,
                title="Calculate KCC Loan Limit",
                description=(
                    "Calculate the indicative agricultural credit limit "
                    "using crop, land area and applicable scheme guidelines."
                ),
                status="pending",
                tool_to_execute="kcc_loan_advisory",
                tool_params={
                    "crop_name": extracted.get("crop_name", "Wheat"),
                    "land_area_hectares": extracted.get(
                        "land_area_ha",
                        1.0,
                    ),
                    "state": extracted.get("state", "Rajasthan"),
                },
            ))

            steps.append(TaskStep(
                step_id=2,
                title="Prepare Bank Application Checklist",
                description=(
                    "List required land records, identity documents and "
                    "bank requirements for the KCC application."
                ),
            ))

        # ─────────────────────────────────────────────────────────────────────
        # 4. MANDI PRICES
        # ─────────────────────────────────────────────────────────────────────

        elif any(term in intent_text for term in (
            "mandi",
            "bhav",
            "price",
            "rate",
            "भाव",
            "मंडी",
            "दाम",
            "msp",
            "बाजार भाव",
        )):

            category = "market_prices"

            crop = extracted.get("crop_name", "Bajra")

            steps.append(TaskStep(
                step_id=1,
                title=f"Fetch Mandi Prices for {crop}",
                description=(
                    "Retrieve mandi price information and compare the "
                    "available price information with the applicable MSP."
                ),
                status="pending",
                tool_to_execute="mandi_prices",
                tool_params={
                    "crop_name": crop,
                    "state": extracted.get("state"),
                },
            ))

        # ─────────────────────────────────────────────────────────────────────
        # 5. SOIL AND FERTILIZER ADVISORY
        # ─────────────────────────────────────────────────────────────────────

        elif any(term in intent_text for term in (
            "khad",
            "fertilizer",
            "soil",
            "urea",
            "dap",
            "खाद",
            "मिट्टी",
            "यूरिया",
            "उर्वरक",
            "मृदा",
        )):

            category = "agronomy"

            crop = extracted.get("crop_name", "Wheat")

            steps.append(TaskStep(
                step_id=1,
                title=f"Soil Health Recommendation for {crop}",
                description=(
                    "Retrieve relevant crop nutrient and soil-health "
                    "recommendations. Confirm local soil-test details "
                    "before applying fertilizer."
                ),
                status="pending",
                tool_to_execute="soil_health_advisory",
                tool_params={
                    "crop_name": crop,
                },
            ))

        # ─────────────────────────────────────────────────────────────────────
        # 6. CROP DISEASE AND PEST DIAGNOSIS
        # ─────────────────────────────────────────────────────────────────────

        elif any(term in intent_text for term in (
            "keeda",
            "pest",
            "disease",
            "peela",
            "sundee",
            "कीड़ा",
            "कीट",
            "रोग",
            "इल्ली",
            "फंगस",
            "पिला",
            "बीमारी",
        )):

            category = "crop_health"

            crop = extracted.get("crop_name", "Wheat")

            steps.append(TaskStep(
                step_id=1,
                title=f"Diagnose Crop Symptoms for {crop}",
                description=(
                    "Analyze the reported crop symptoms and retrieve "
                    "appropriate crop-health management guidance."
                ),
                status="pending",
                tool_to_execute="pest_disease_diagnostic",
                tool_params={
                    "crop_name": crop,
                    "symptoms": query,
                },
            ))

        # ─────────────────────────────────────────────────────────────────────
        # 7. GENERAL AGRICULTURAL ADVISORY
        # ─────────────────────────────────────────────────────────────────────

        else:

            category = "general_advisory"

            steps.append(TaskStep(
                step_id=1,
                title="Understand Farmer Requirement",
                description=(
                    "Analyze the farmer's requirement and identify the "
                    "relevant agricultural advisory or government service."
                ),
                status="pending",
            ))

            steps.append(TaskStep(
                step_id=2,
                title="Prepare Action Plan",
                description=(
                    "Provide relevant next steps, required information "
                    "or an appropriate official agricultural helpline."
                ),
                status="pending",
            ))

        # ─────────────────────────────────────────────────────────────────────
        # BUILD EXECUTION PLAN
        # ─────────────────────────────────────────────────────────────────────

        plan = ExecutionPlan(
            task_id=task_id,
            goal=query,
            category=category,
            language=lang,
            steps=steps,
            extracted_facts=extracted,
            missing_info=missing_info,
            clarification_prompt=clarification,
        )

        self._active_plans[task_id] = plan

        logger.info(
            "[Planner] Created plan %s with category %s and %d steps",
            task_id,
            category,
            len(steps),
        )

        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTE PLAN STEP
    # ─────────────────────────────────────────────────────────────────────────

    async def execute_step(
        self,
        task_id: str,
        step_id: int,
    ) -> dict[str, Any]:

        """Execute a registered tool for a selected plan step."""

        plan = self._active_plans.get(task_id)

        if not plan:
            return {
                "success": False,
                "error": f"Plan {task_id} not found.",
            }

        target_step: Optional[TaskStep] = None

        for step in plan.steps:
            if step.step_id == step_id:
                target_step = step
                break

        if target_step is None:
            return {
                "success": False,
                "error": f"Step {step_id} not found in plan {task_id}.",
            }

        target_step.status = "in_progress"

        if target_step.tool_to_execute:

            tool = registry.get(target_step.tool_to_execute)

            if not tool:
                target_step.status = "failed"

                return {
                    "success": False,
                    "error": (
                        f"Tool '{target_step.tool_to_execute}' "
                        "not registered."
                    ),
                }

            try:
                result = await tool.execute(**target_step.tool_params)

                target_step.result = result

                target_step.status = (
                    "completed"
                    if result.get("success")
                    else "failed"
                )

                return {
                    "success": bool(result.get("success")),
                    "step_id": step_id,
                    "status": target_step.status,
                    "result": result,
                }

            except Exception as exc:
                logger.exception(
                    "[Planner] Tool execution failed for task %s, step %s",
                    task_id,
                    step_id,
                )

                target_step.status = "failed"

                return {
                    "success": False,
                    "step_id": step_id,
                    "status": "failed",
                    "error": str(exc),
                }

        target_step.status = "completed"

        return {
            "success": True,
            "step_id": step_id,
            "status": "completed",
            "message": "Step executed successfully.",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # GET EXISTING PLAN
    # ─────────────────────────────────────────────────────────────────────────

    def get_plan(
        self,
        task_id: str,
    ) -> Optional[ExecutionPlan]:

        return self._active_plans.get(task_id)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PLANNER INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

planner = GeneralAgentPlanner()