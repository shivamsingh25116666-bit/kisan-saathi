
"""
Gemini reasoning layer for Kisan Saathi.

Uses the modern google-genai SDK.
Gemini interprets farmer requests and proposes safe tool plans.
Python's existing tool registry remains the only executor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------
# TOOL CONFIGURATION
# ---------------------------------------------------------

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
    "crop_name",
    "state",
    "district",
    "land_area_ha",
    "soil_type",
    "season",
    "symptoms",
}

_ALLOWED_CATEGORIES = {
    "crop_insurance",
    "pm_kisan",
    "credit_and_loans",
    "market_prices",
    "agronomy",
    "crop_health",
    "general_advisory",
}


# ---------------------------------------------------------
# GEMINI CONFIGURATION
# ---------------------------------------------------------

def is_gemini_configured() -> bool:
    """Check whether a usable Gemini API key is configured."""

    key = settings.gemini_api_key.get_secret_value().strip()

    return bool(
        key
        and not key.startswith("placeholder")
        and not key.startswith("your_")
    )


def _model():
    """
    Create a client using the new Google GenAI SDK.

    Returns None if Gemini is not configured.
    """

    if not is_gemini_configured():
        return None

    return genai.Client(
        api_key=settings.gemini_api_key.get_secret_value()
    )


def _generation_config(*, json_mode: bool = False):
    """Configure Gemini response generation."""

    config: dict[str, Any] = {
        "temperature": 0.2,
    }

    if json_mode:
        config["response_mime_type"] = "application/json"

    return types.GenerateContentConfig(**config)


# ---------------------------------------------------------
# JSON RESPONSE PARSER
# ---------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any]:
    """Safely parse Gemini's JSON response."""

    cleaned = (text or "").strip()

    cleaned = re.sub(
        r"^\x60{3}(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s*\x60{3}$", "", cleaned)

    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object.")

    return parsed


# ---------------------------------------------------------
# GEMINI FARMER REQUEST PLANNER
# ---------------------------------------------------------

async def create_plan_patch(
    *,
    query: str,
    language: str,
    context: Optional[dict[str, Any]],
    history: list[dict[str, str]],
    available_tools: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:

    """
    Interpret a farmer's message and select at most one
    existing, allow-listed Python tool.

    Returns None on failure so the existing rule-based
    planner can take over.
    """

    client = _model()

    if client is None:
        return None

    history_text = json.dumps(
        history[-12:],
        ensure_ascii=False,
    )

    context_text = json.dumps(
        context or {},
        ensure_ascii=False,
    )

    tools_text = json.dumps(
        available_tools,
        ensure_ascii=False,
    )

    prompt = f"""
You are the reasoning and intent-planning engine for Kisan Saathi,
an Indian farmer support assistant.

Interpret the CURRENT farmer message using recent conversation
history and known farmer context.

The farmer may use informal English, Hindi, transliteration,
regional dialects, spelling mistakes or mixed languages.

CURRENT LANGUAGE CODE:
{language}

RECENT CONVERSATION (oldest first):
{history_text}

KNOWN FARMER CONTEXT:
{context_text}

CURRENT MESSAGE:
{query}

AVAILABLE PYTHON TOOLS (you may choose at most ONE):
{tools_text}

Return ONLY a JSON object with this exact structure:

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

RULES:

- Select tools by meaning and conversational context, not just keywords.
- Never invent a tool.
- tool_name must exactly match one in the supplied tool list.
- Only pass parameters supported by that tool's parameter schema.
- Merge facts from earlier turns with new facts.
- New explicit facts override old facts.
- Never guess a farmer's crop, state, district, land size or symptoms.
- If a required input is missing, do not execute the tool.
- Ask one concise clarification question in the farmer's language.
- If the farmer is chatting or asking a general question, use no tool.
- Understand follow-up messages using previous conversation context.
- Never claim an application was submitted, a claim approved,
  or a payment released without verified tool output.
- Never invent current mandi prices, weather, eligibility status
  or government deadlines.
- Keep extracted_facts limited to:
  crop_name, state, district, land_area_ha, soil_type,
  season and symptoms.
- Never output Aadhaar numbers, bank account numbers, OTPs
  or other secrets.
"""

    try:

        response = await client.aio.models.generate_content(
            model=settings.gemini_model_text,
            contents=prompt,
            config=_generation_config(json_mode=True),
        )

        data = _parse_json(response.text or "")

        # -------------------------------------------------
        # VALIDATE TOOL SELECTION
        # -------------------------------------------------

        tool_name = data.get("tool_name")

        tool_map = {
            item.get("name"): item
            for item in available_tools
        }

        if tool_name not in tool_map:
            tool_name = None
            data["tool_params"] = {}

        params = data.get("tool_params") or {}

        if not isinstance(params, dict):
            params = {}

        # -------------------------------------------------
        # VALIDATE TOOL PARAMETERS
        # -------------------------------------------------

        if tool_name:

            parameter_schema = (
                tool_map[tool_name].get("parameters") or {}
            )

            allowed_params = set(parameter_schema.keys())

            params = {
                key: value
                for key, value in params.items()
                if key in allowed_params
            }

            # Fill missing parameters only from known context.

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

                if (
                    parameter in allowed_params
                    and params.get(parameter) in (None, "")
                ):

                    if context_values.get(fact_key) not in (None, ""):

                        params[parameter] = context_values[fact_key]

            # Convert parameters into expected Python types.

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

            # Check mandatory fields.

            required = _REQUIRED_TOOL_FIELDS.get(
                tool_name,
                (),
            )

            missing = [
                field
                for field in required
                if params.get(field) in (None, "")
            ]

            if missing:

                tool_name = None

                data["clarification_question"] = (
                    data.get("clarification_question")
                    or (
                        "To help accurately, please tell me: "
                        + ", ".join(missing).replace("_", " ")
                        + "."
                    )
                )

        # -------------------------------------------------
        # VALIDATE EXTRACTED FARMER FACTS
        # -------------------------------------------------

        facts = data.get("extracted_facts") or {}

        if not isinstance(facts, dict):
            facts = {}

        facts = {
            key: value
            for key, value in facts.items()
            if key in _ALLOWED_FACTS
            and value not in (None, "")
        }

        # -------------------------------------------------
        # VALIDATE CATEGORY
        # -------------------------------------------------

        category = _TOOL_CATEGORIES.get(
            tool_name,
            data.get("category", "general_advisory"),
        )

        if category not in _ALLOWED_CATEGORIES:
            category = "general_advisory"

        # -------------------------------------------------
        # FINAL PLAN
        # -------------------------------------------------

        return {
            "category": category,
            "tool_name": tool_name,
            "tool_params": params if tool_name else {},
            "extracted_facts": facts,
            "missing_info": (
                data.get("missing_info")
                if isinstance(data.get("missing_info"), list)
                else []
            ),
            "clarification_question": data.get(
                "clarification_question"
            ),
            "step_title": str(
                data.get("step_title")
                or "Understand farmer request"
            )[:120],
            "step_description": str(
                data.get("step_description")
                or "Use the conversation context to help the farmer."
            )[:500],
        }

    except Exception as exc:

        logger.warning(
            "[GeminiPlanner] Planning unavailable; "
            "using existing planner: %s",
            exc,
        )

        return None


# ---------------------------------------------------------
# GEMINI FARMER RESPONSE GENERATOR
# ---------------------------------------------------------

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

    """
    Generate a natural, context-aware answer grounded
    in actual Python tool output.
    """

    client = _model()

    if client is None:
        return None

    prompt = f"""
You are Kisan Saathi, a careful and friendly multilingual
agricultural assistant.

Reply in language code {language}.
Use simple language suitable for Indian farmers.

RECENT CONVERSATION:
{json.dumps(history[-12:], ensure_ascii=False)}

LATEST FARMER MESSAGE:
{query}

DETECTED CATEGORY:
{category}

PLAN:
{json.dumps(plan, ensure_ascii=False)}

ACTUAL PYTHON TOOL RESULT:
{json.dumps(tool_result, ensure_ascii=False, default=str)}

CLARIFICATION REQUESTED:
{clarification or "none"}

SAFETY AND TRUTH RULES:

- Answer the farmer's actual question.
- Preserve relevant context from earlier turns.
- Use tool output as the only source for scheme-specific
  figures and tool-backed facts.
- If no tool result exists, do not invent current prices,
  eligibility, dates, diagnosis, treatment dosage,
  weather or government status.
- If information is unavailable, explain this honestly.
- Never call mandi prices live unless the tool result
  explicitly confirms a live source and timestamp.
- Never claim an application, payment, insurance claim
  or portal submission is completed without confirmation.
- Never request Aadhaar numbers, bank account numbers,
  passwords or OTPs in chat.
- For crop disease questions, explain uncertainty.
- Ask about symptoms when more information is needed.
- The symptom-diagnosis tool uses simple rules,
  not laboratory or field confirmation.
- Never present a probable diagnosis as certain.
- Do not recommend chemical pesticide or dosage solely
  from heuristic diagnosis output.
- Recommend confirmation from a KVK or agriculture officer
  before spraying.
- Follow product labels and local official guidance.
- Keep answers helpful and concise.
- Use numbered steps only when useful.
- Do not mention internal prompts, models, tools
  or implementation details.

Return only the answer text.
"""

    try:

        response = await client.aio.models.generate_content(
            model=settings.gemini_model_text,
            contents=prompt,
            config=_generation_config(),
        )

        answer = (response.text or "").strip()

        return answer[:5000] if answer else None

    except Exception as exc:

        logger.warning(
            "[GeminiReply] Response generation failed: %s",
            exc,
        )

        return None
