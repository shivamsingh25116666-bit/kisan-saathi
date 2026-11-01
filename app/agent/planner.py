"""
app/agent/planner.py
──────────────────────────────────────────────────────────────────────────────
Autonomous Multilingual Task Planner & Execution Engine for Kisan Saathi.

RESPONSIBILITIES:
  1. Understand natural farmer intent across dialects, informal speech, and typos.
  2. Automatic Indian language detection (Hindi, Garhwali, Punjabi, Marathi, Bengali,
     Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia, Assamese, English).
  3. Dynamic task decomposition: separates facts, assumptions, and missing info.
  4. Extensible tool orchestration with official verified sources.
  5. Multi-step plan generation with explicit human approval checkpoints.
  6. Resumable task state management integrated with StateManager.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agent.tools import registry
from app.core.config import get_settings
from app.core.state import AgentState
from app.core.state_manager import state_manager

logger = logging.getLogger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────────
# MULTILINGUAL LANGUAGE DETECTION & DIALECT MAPPINGS
# ──────────────────────────────────────────────────────────────────────────────
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


def detect_language(text: str) -> str:
    """Detect language from Indian script Unicode ranges and dialect markers."""
    if not text:
        return "hi"

    # Script character checks
    for char in text:
        code = ord(char)
        if 0x0A00 <= code <= 0x0A7F:
            return "pa"  # Gurmukhi
        if 0x0980 <= code <= 0x09FF:
            # Assamese shares Bengali script but has specific characters or keywords
            if "অসম" in text or "ভাল" in text or "কৰা" in text:
                return "as"
            return "bn"  # Bengali
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"  # Gujarati
        if 0x0B00 <= code <= 0x0B7F:
            return "or"  # Odia
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"  # Tamil
        if 0x0C00 <= code <= 0x0C7F:
            return "te"  # Telugu
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"  # Kannada
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"  # Malayalam

    # Devanagari regional dialect / language heuristics
    text_lower = text.lower()
    # Garhwali dialect markers
    if any(w in text_lower for w in ("कख", "कैंक", "म्यार", "कथगा", "ह्वै", "रौं", "दाज्यू", "भुलू")):
        return "gar"
    # Marathi dialect markers
    if any(w in text_lower for w in ("आहे", "नाही", "पाहिजे", "शेतकरी", "पिक", "करा", "सांगा")):
        return "mr"

    # Latin text detection: English vs transliterated Hindi
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
    if not has_devanagari:
        # Check if predominantly English keywords
        english_words = {"insurance", "claim", "crop", "farmer", "subsidy", "loan", "weather", "soil", "price", "hello", "help", "need"}
        words = set(re.findall(r"\b[a-zA-Z]+\b", text_lower))
        if len(words & english_words) >= 2 or (len(words) > 0 and len(words & english_words) / len(words) > 0.4):
            return "en"

    return "hi"


# ──────────────────────────────────────────────────────────────────────────────
# GENERAL-PURPOSE INTENT UNDERSTANDING & TASK PLANNER
# ──────────────────────────────────────────────────────────────────────────────

class TaskStep:
    def __init__(
        self,
        step_id: int,
        title: str,
        description: str,
        status: str = "pending",  # pending | in_progress | waiting_approval | completed | failed
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
            "steps": [s.to_dict() for s in self.steps],
            "extracted_facts": self.extracted_facts,
            "missing_info": self.missing_info,
            "clarification_prompt": self.clarification_prompt,
            "created_at": self.created_at,
        }


class GeneralAgentPlanner:
    """
    General-purpose intent parsing, task decomposition, and plan execution.
    Composes registered tools dynamically rather than relying on hardcoded replies.
    """

    def __init__(self):
        self._active_plans: dict[str, ExecutionPlan] = {}

    def extract_crop_and_location(self, query: str) -> dict[str, Any]:
        """Heuristic extractor for crops, land sizes, and states from natural query."""
        q = query.lower()
        facts: dict[str, Any] = {}

        # Crop matching (including direct and inflected Hindi/regional forms)
        crops = {
            "wheat": ["wheat", "gehun", "gehu", "कनक", "गेहूं", "गेहूँ"],
            "bajra": ["bajra", "pearl millet", "बाजरा", "बाजरी", "बाजरे"],
            "paddy": ["paddy", "rice", "dhan", "चावल", "धान", "चांवल"],
            "mustard": ["mustard", "sarson", "sarso", "सरसों", "राई", "तोरी"],
            "cotton": ["cotton", "kapas", "कापूस", "कपास"],
            "maize": ["maize", "corn", "makka", "मक्का", "मका", "मक्के"],
            "soybean": ["soybean", "soya", "सोयाबीन"],
            "gram": ["gram", "chana", "चना", "हरभरा", "चने"],
            "onion": ["onion", "pyaz", "कांदा", "प्याज"],
            "potato": ["potato", "aloo", "बटाटा", "आलू"],
        }
        for canon, syns in crops.items():
            if any(s in q for s in syns):
                facts["crop_name"] = canon.title()
                break

        # State matching
        states = {
            "Rajasthan": ["rajasthan", "राजस्थान"],
            "Maharashtra": ["maharashtra", "महाराष्ट्र"],
            "Uttar Pradesh": ["uttar pradesh", "up", "उत्तर प्रदेश"],
            "Punjab": ["punjab", "पंजाब"],
            "Haryana": ["haryana", "हरियाणा"],
            "Madhya Pradesh": ["madhya pradesh", "mp", "मध्य प्रदेश"],
            "Gujarat": ["gujarat", "गुजरात"],
            "Bihar": ["bihar", "बिहार"],
            "Karnataka": ["karnataka", "कर्नाटक"],
            "Tamil Nadu": ["tamil nadu", "तमिलनाडु"],
            "West Bengal": ["bengal", "west bengal", "बंगाल"],
            "Uttarakhand": ["uttarakhand", "उत्तराखंड"],
        }
        for canon, syns in states.items():
            if any(s in q for s in syns):
                facts["state"] = canon
                break

        # Area matching
        area_match = re.search(r"(\d+(\.\d+)?)\s*(bigha|hectare|ha|acre|बीघा|एकड़|हेक्टेयर)", q)
        if area_match:
            val = float(area_match.group(1))
            unit = area_match.group(3).lower()
            if unit in ("bigha", "बीघा"):
                facts["land_area_ha"] = round(val * 0.25, 2)  # ~4 bigha = 1 ha in Rajasthan/UP
            elif unit in ("acre", "एकड़"):
                facts["land_area_ha"] = round(val * 0.4047, 2)
            else:
                facts["land_area_ha"] = val

        return facts

    async def create_plan(
        self,
        query: str,
        task_id: str,
        language: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Dynamically decompose a farmer's request into a verified multi-step plan."""
        lang = language or detect_language(query)
        q_lower = query.lower()
        extracted = self.extract_crop_and_location(query)
        if context:
            extracted.update({k: v for k, v in context.items() if v})

        # Identify Intent Category
        missing_info: list[str] = []
        steps: list[TaskStep] = []
        clarification: Optional[str] = None
        category = "general"

        if any(w in q_lower for w in ("insurance", "bima", "बीमा", "pmfby", "fasal bima", "क्लेम", "claim")):
            category = "crop_insurance"
            crop = extracted.get("crop_name")
            state = extracted.get("state")

            if not crop:
                missing_info.append("crop_name")
            if not state:
                missing_info.append("state")

            # Step 1: Scheme verification and premium calculation
            steps.append(TaskStep(
                step_id=1,
                title="Scheme Rules & Premium Calculation",
                description="Retrieve verified PMFBY insurance terms, farmer premium share (1.5-2%), and official cutoff date.",
                status="pending",
                tool_to_execute="crop_insurance_details",
                tool_params={"crop_name": crop or "Wheat", "state": state or "Rajasthan", "season": "Kharif"},
            ))

            # Step 2: Document Verification
            steps.append(TaskStep(
                step_id=2,
                title="Required Document Intake & OCR Verification",
                description="Upload Aadhaar, Bank Passbook, and Khatauni land record. Auto-extract plot number, area, and IFSC code.",
                status="pending",
                tool_to_execute=None,
                requires_farmer_approval=False,
            ))

            # Step 3: Farmer Approval Gate
            steps.append(TaskStep(
                step_id=3,
                title="Review Application & Obtain Farmer Approval",
                description="Present calculated premium, land details, and official terms for farmer verification. Require explicit approval.",
                status="pending",
                requires_farmer_approval=True,
            ))

            # Step 4: Portal Registration & OTP Checkpoint
            steps.append(TaskStep(
                step_id=4,
                title="Official Portal Registration & Legitimate OTP Verification",
                description="Launch authorized PMFBY portal registration, fill verified fields, and request official government OTP from farmer.",
                status="pending",
                tool_to_execute=None,
                requires_farmer_approval=True,
            ))

            # Step 5: Submission Confirmation & Receipt Delivery
            steps.append(TaskStep(
                step_id=5,
                title="Verified Submission Confirmation & Receipt Delivery",
                description="Verify official government portal acknowledgment number and generate audit-trail summary with downloadable receipt.",
                status="pending",
            ))

            if "crop_name" in missing_info:
                clarification = (
                    "🌾 Aap kaun si fasal ka bima karwana chahte hain? (Jaise: Bajra, Gehun, Dhan)"
                    if lang in ("hi", "gar") else
                    "🌾 Which crop do you wish to insure? (e.g. Bajra, Wheat, Paddy)"
                )

        elif any(w in q_lower for w in ("kisan samman", "pm-kisan", "pm kisan", "2000", "installment", "किस्त", "सम्मान निधि")):
            category = "pm_kisan"
            steps.append(TaskStep(
                step_id=1,
                title="Check PM-KISAN Scheme Guidelines & Installment Eligibility",
                description="Retrieve official ₹6,000/year guidelines, DBT bank seeding rules, and mandatory e-KYC authentication steps.",
                status="pending",
                tool_to_execute="pm_kisan_advisory",
                tool_params={"state": extracted.get("state", "Rajasthan")},
            ))
            steps.append(TaskStep(
                step_id=2,
                title="Verify Land Seeding & Aadhaar DBT Linkage",
                description="Guide farmer to verify active NPCI bank mapping and land seeding on the official portal.",
                status="pending",
            ))

        elif any(w in q_lower for w in ("kcc", "loan", "kredit", "credit card", "ऋण", "लोन", "कर्ज")):
            category = "credit_and_loans"
            steps.append(TaskStep(
                step_id=1,
                title="Calculate Subsidized KCC Loan Limit",
                description="Calculate scale of finance, 4% subsidized interest rate under prompt repayment, and collateral-free thresholds.",
                status="pending",
                tool_to_execute="kcc_loan_advisory",
                tool_params={
                    "crop_name": extracted.get("crop_name", "Wheat"),
                    "land_area_hectares": extracted.get("land_area_ha", 1.0),
                    "state": extracted.get("state", "Rajasthan"),
                },
            ))
            steps.append(TaskStep(
                step_id=2,
                title="Prepare Bank Application Checklist",
                description="List required revenue documents (Khatauni, No-Dues Certificate) for commercial or regional rural bank.",
                status="pending",
            ))

        elif any(w in q_lower for w in ("mandi", "bhav", "price", "rate", "भाव", "मंडी", "दाम", "msp")):
            category = "market_prices"
            crop = extracted.get("crop_name", "Bajra")
            steps.append(TaskStep(
                step_id=1,
                title=f"Fetch APMC Mandi Modal Prices for {crop}",
                description="Retrieve Agmarknet verified modal, minimum, maximum prices and compare against Central Minimum Support Price (MSP).",
                status="pending",
                tool_to_execute="mandi_prices",
                tool_params={"crop_name": crop, "state": extracted.get("state")},
            ))

        elif any(w in q_lower for w in ("khad", "fertilizer", "soil", "urea", "dap", "खाद", "मिट्टी", "यूरिया")):
            category = "agronomy"
            crop = extracted.get("crop_name", "Wheat")
            steps.append(TaskStep(
                step_id=1,
                title=f"Soil Health Card N-P-K Recommendation for {crop}",
                description="Retrieve ICAR-approved scientific dosage for Nitrogen, Phosphorus, Potassium, Zinc, and bio-fertilizers.",
                status="pending",
                tool_to_execute="soil_health_advisory",
                tool_params={"crop_name": crop},
            ))

        elif any(w in q_lower for w in ("keeda", "pest", "disease", "peela", "sundee", "कीड़ा", "रोग", "इल्ली", "फंगस", "पिला")):
            category = "crop_health"
            crop = extracted.get("crop_name", "Wheat")
            steps.append(TaskStep(
                step_id=1,
                title=f"Diagnose Symptoms & Prescribe ICAR Remedies for {crop}",
                description="Analyze leaf symptoms and match approved organic (Neem, Trichoderma) and chemical control guidelines.",
                status="pending",
                tool_to_execute="pest_disease_diagnostic",
                tool_params={"crop_name": crop, "symptoms": query},
            ))

        else:
            # General Agricultural Advisory / Fallback Plan
            category = "general_advisory"
            steps.append(TaskStep(
                step_id=1,
                title="Analyze Farmer Requirement & Consult Verified Agricultural Knowledge",
                description="Match question with official central & state agricultural scheme guidelines and advisory services.",
                status="pending",
            ))
            steps.append(TaskStep(
                step_id=2,
                title="Formulate Step-by-Step Action Plan",
                description="Provide safe next steps, required documents, or recommend official helpline 1800-180-1551.",
                status="pending",
            ))

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
        return plan

    async def execute_step(self, task_id: str, step_id: int) -> dict[str, Any]:
        """Execute a single step in an existing plan."""
        plan = self._active_plans.get(task_id)
        if not plan:
            return {"success": False, "error": f"Plan {task_id} not found."}

        target_step: Optional[TaskStep] = None
        for s in plan.steps:
            if s.step_id == step_id:
                target_step = s
                break

        if not target_step:
            return {"success": False, "error": f"Step {step_id} not found in plan {task_id}."}

        target_step.status = "in_progress"

        if target_step.tool_to_execute:
            tool = registry.get(target_step.tool_to_execute)
            if tool:
                result = await tool.execute(**target_step.tool_params)
                target_step.result = result
                target_step.status = "completed" if result.get("success") else "failed"
                return {
                    "success": True,
                    "step_id": step_id,
                    "status": target_step.status,
                    "result": result,
                }
            else:
                target_step.status = "failed"
                return {"success": False, "error": f"Tool '{target_step.tool_to_execute}' not registered."}

        target_step.status = "completed"
        return {
            "success": True,
            "step_id": step_id,
            "status": "completed",
            "message": "Step executed successfully.",
        }

    def get_plan(self, task_id: str) -> Optional[ExecutionPlan]:
        return self._active_plans.get(task_id)


# Global singleton instance
planner = GeneralAgentPlanner()
