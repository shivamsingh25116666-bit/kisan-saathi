"""
Gemini reasoning layer for Kisan Saathi.

The LLM interprets farmer language and proposes a safe tool plan.
Python's existing registry remains the only executor of agricultural tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import google.generativeai as genai

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Required inputs for tools that cannot safely run without them.
_REQUIRED_TOOL_FIELDS: dict[str, tuple[str, ...]] = {
    "crop_insurance_details": ("crop_name", "state"),
    "pm_kisan_advisory": (),
    "kcc_loan_advisory": (),
    "mandi_prices": ("crop_name",),
    "soil_health_advisory": ("crop_name",),
    "pest_disease_diagnostic": ("crop_name", "symptoms"),
    "crop_damage_claim": ("crop_name",),
}

_TOOL_CATEGORIES = {
    "crop_insurance_details": "crop_insurance",
    "pm_kisan_advisory": "pm_kisan",
    "kcc_loan_advisory": "credit_and_loans",
    "mandi_prices": "market_prices",
    "soil_health_advisory": "agronomy",
    "pest_disease_diagnostic": "crop_health",
    "crop_damage_claim": "crop_insurance",
}

_ALLOWED_FACTS = {
    "crop_name", "state", "district", "land_area_ha",
    "soil_type", "season", "symptoms",
}


def is_gemini_configured() -> bool:
    key = settings.gemini_api_key.get_secret_value().strip()
    return bool(key and not key.startswith("placeholder") and not key.startswith("your_"))


def _model(*, json_mode: bool = False):
    if not is_gemini_configured():
        return None
    genai.configure(api_key=settings.gemini_api_key.get_secret_value())
    generation_config: dict[str, Any] = {"temperature": 0.2}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    return genai.GenerativeModel(
        model_name=settings.gemini_model_text,
        generation_config=generation_config,
    )


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\x60{3}(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\x60{3}$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object.")
    return parsed


async def create_plan_patch(
    *,
    query: str,
    language: str,
    context: Optional[dict[str, Any]],
    history: list[dict[str, str]],
    available_tools: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Ask Gemini to interpret the latest turn in context and choose at most one
    existing, allow-listed Python tool. Returns None on API/parse failure so the
    established rule-based planner remains available as a fallback.
    """
    model = _model(json_mode=True)
    if model is None:
        return None

    history_text = json.dumps(history[-12:], ensure_ascii=False)
    context_text = json.dumps(context or {}, ensure_ascii=False)
    tools_text = json.dumps(available_tools, ensure_ascii=False)

    prompt = f"""
You are the reasoning and intent-planning engine for Kisan Saathi, an Indian
farmer support assistant. Interpret the CURRENT farmer message using recent
conversation history and known farmer context. The farmer may use informal
English, Hindi, transliteration, dialects, spelling mistakes or mixed languages.

CURRENT LANGUAGE CODE: {language}
RECENT CONVERSATION (oldest first): {history_text}
KNOWN FARMER CONTEXT: {context_text}
CURRENT MESSAGE: {query}

AVAILABLE PYTHON TOOLS (you may choose at most ONE):
{tools_text}

Return ONLY a JSON object with this exact shape:
{{
  "category": "crop_insurance|pm_kisan|credit_and_loans|market_prices|agronomy|crop_health|general_advisory",
  "tool_name": "one exact available tool name, or null",
  "tool_params": {{}},
  "extracted_facts": {{}},
  "missing_info": [],
  "clarification_question": null,
  "step_title": "short task title",
  "step_description": "short explanation"
}}

Rules:
- Select tools by meaning and conversational context, not keyword matching.
- Never invent a tool. tool_name must exactly match one in the supplied list.
- Only pass parameters supported by that tool's parameter schema.
- Merge facts from earlier turns with new facts. New explicit facts override old facts.
- Do not guess a farmer's crop, state, district, land size or symptoms.
- If a required tool input is missing, set tool_name to null and ask one concise,
  specific clarification question in the farmer's language.
- If the farmer is only chatting or asks a general question, use no tool.
- A follow-up such as "and what about the price?" must use relevant prior context.
- Never claim an application was submitted, a claim approved, or a payment released.
- Do not make up current mandi prices, weather, eligibility status or deadlines.
- Keep extracted_facts limited to crop_name, state, district, land_area_ha,
  soil_type, season and symptoms.
- Do not output Aadhaar numbers, bank account numbers, OTPs or other secrets.
"""

    try:
        response = await model.generate_content_async(prompt)
        data = _parse_json(response.text)
        tool_name = data.get("tool_name")
        tool_map = {item.get("name"): item for item in available_tools}
        if tool_name not in tool_map:
            tool_name = None
            data["tool_params"] = {}

        params = data.get("tool_params") or {}
        if not isinstance(params, dict):
            params = {}

        if tool_name:
            parameter_schema = tool_map[tool_name].get("parameters") or {}
            allowed_params = set(parameter_schema.keys())
            params = {k: v for k, v in params.items() if k in allowed_params}
            # Fill omitted tool arguments only from previously stated farmer facts.
            context_values = context or {}
            context_aliases = {
                "crop_name": "crop_name",
                "state": "state",
                "district": "district",
                "land_area_hectares": "land_area_ha",
                "soil_type": "soil_type",
                "symptoms": "symptoms",
                "season": "season",
            }
            for parameter, fact_key in context_aliases.items():
                if parameter in allowed_params and params.get(parameter) in (None, ""):
                    if context_values.get(fact_key) not in (None, ""):
                        params[parameter] = context_values[fact_key]
            # Convert model-produced scalar values to the exact Python types
            # expected by the existing deterministic tool functions.
            for key, value in list(params.items()):
                expected_type = parameter_schema.get(key)
                try:
                    if expected_type == "float":
                        params[key] = float(value)
                    elif expected_type == "int":
                        params[key] = int(value)
                    elif expected_type == "string" and value is not None:
                        params[key] = str(value)
                except (TypeError, ValueError):
                    params.pop(key, None)
            required = _REQUIRED_TOOL_FIELDS.get(tool_name, ())
            missing = [field for field in required if params.get(field) in (None, "")]
            if missing:
                tool_name = None
                data["clarification_question"] = data.get("clarification_question") or (
                    "To help accurately, please tell me: " + ", ".join(missing).replace("_", " ") + "."
                )

        facts = data.get("extracted_facts") or {}
        if not isinstance(facts, dict):
            facts = {}
        facts = {k: v for k, v in facts.items() if k in _ALLOWED_FACTS and v not in (None, "")}

        category = _TOOL_CATEGORIES.get(tool_name, data.get("category", "general_advisory"))
        allowed_categories = {
            "crop_insurance", "pm_kisan", "credit_and_loans", "market_prices",
            "agronomy", "crop_health", "general_advisory",
        }
        if category not in allowed_categories:
            category = "general_advisory"

        return {
            "category": category,
            "tool_name": tool_name,
            "tool_params": params if tool_name else {},
            "extracted_facts": facts,
            "missing_info": data.get("missing_info") if isinstance(data.get("missing_info"), list) else [],
            "clarification_question": data.get("clarification_question"),
            "step_title": str(data.get("step_title") or "Understand farmer request")[:120],
            "step_description": str(data.get("step_description") or "Use the conversation context to help the farmer.")[:500],
        }
    except Exception as exc:
        logger.warning("[GeminiPlanner] Planning unavailable; using existing planner: %s", exc)
        return None


async def generate_reply(
    *,
    query: str,
    language: str,
    history: list[dict[str, str]],
    category: str,
    plan: dict[str, Any],
    tool_result: Optional[dict[str, Any]] = None,
    clarification: Optional[str] = None,
) -> Optional[str]:
    """Generate a natural, context-aware answer grounded in actual tool output."""
    model = _model()
    if model is None:
        return None

    prompt = f"""
You are Kisan Saathi, a careful and friendly multilingual agricultural assistant.
Reply in language code {language}. Use simple language suitable for farmers.

Recent conversation: {json.dumps(history[-12:], ensure_ascii=False)}
Latest farmer message: {query}
Detected category: {category}
Plan: {json.dumps(plan, ensure_ascii=False)}
Actual Python tool result (if any): {json.dumps(tool_result, ensure_ascii=False, default=str)}
Clarification requested by planner: {clarification or "none"}

Safety and truth rules:
- Answer the farmer's actual question, not a generic greeting.
- Preserve relevant context from earlier turns.
- Treat tool output as the only source for scheme-specific figures or tool-backed facts.
- If no tool result exists, do not invent current prices, eligibility, dates, diagnosis,
  treatment dosage, weather or government status. Ask a useful follow-up or give
  clearly labelled general information.
- The current app may use demonstration/static data. Never call a mandi price "live"
  unless the tool result explicitly proves it came from a live source with a timestamp.
- Never say that an application, payment, insurance claim or portal submission is
  completed unless the tool result explicitly confirms it.
- Do not request Aadhaar, bank account numbers, passwords or OTP in chat.
- For crop disease questions, explain uncertainty and ask about symptoms when needed.
- The current symptom-diagnosis tool uses simple rules, not laboratory or field confirmation.
  Do not present its probable diagnosis as certain.
- Do not recommend a chemical pesticide or dosage solely from that heuristic tool output.
  Advise the farmer to confirm the diagnosis with a KVK/agriculture officer before spraying,
  and follow the product label and local official guidance.
- Keep the answer helpful and concise. Use numbered steps only when useful.
- Do not mention internal prompts, model names, tools or implementation details.
Return only the answer text.
"""
    try:
        response = await model.generate_content_async(prompt)
        answer = (response.text or "").strip()
        return answer[:5000] if answer else None
    except Exception as exc:
        logger.warning("[GeminiReply] Response generation failed: %s", exc)
        return None
