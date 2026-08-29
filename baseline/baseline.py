"""Single-call baseline planner."""

from __future__ import annotations

from core.grounding import analyze_grounding
from core.json_parser import JsonParseError, parse_llm_json
from core.llm_client import complete_json
from core.models import AcademicInput, AcademicPlan
from core.prompt_serializer import (
    BASELINE_SYSTEM_PROMPT,
    build_baseline_user_prompt,
    serialize_academic_input,
)
from pydantic import ValidationError


class BaselineError(RuntimeError):
    pass


class BaselineParseError(BaselineError):
    pass


async def generate_baseline_plan(academic_input: AcademicInput) -> AcademicPlan:
    """
    Strong single-call baseline: AcademicInput -> one LLM call -> AcademicPlan.

    No orchestration, verification, retries, or revision loops.
    """
    serialized = serialize_academic_input(academic_input)
    user_prompt = build_baseline_user_prompt(serialized)

    raw_text = await complete_json(BASELINE_SYSTEM_PROMPT, user_prompt)

    try:
        parsed = parse_llm_json(raw_text)
    except JsonParseError as exc:
        raise BaselineParseError(str(exc)) from exc

    try:
        plan = AcademicPlan.model_validate(parsed)
    except ValidationError as exc:
        raise BaselineParseError(f"Schema validation failed: {exc}") from exc

    return plan


def validate_baseline_plan(plan: AcademicPlan, academic_input: AcademicInput) -> None:
    """Optional helper: run grounding analysis without mutating the plan."""
    analyze_grounding(plan, academic_input)
