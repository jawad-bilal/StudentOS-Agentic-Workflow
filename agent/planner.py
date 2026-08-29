"""Planning agent LLM calls (baseline + retry prompts)."""

from __future__ import annotations

from core.findings import AttemptReview
from core.llm_client import complete_json
from core.models import AcademicInput
from core.prompt_serializer import (
    BASELINE_SYSTEM_PROMPT,
    build_agent_retry_user_prompt,
    build_attempt_summary_json,
    build_baseline_user_prompt,
    serialize_academic_input,
)
from core.findings import AttemptReview, PlanReviewFinding


async def call_planning_agent(
    academic_input: AcademicInput,
    *,
    retry_findings: list[PlanReviewFinding] | None = None,
    previous_review: AttemptReview | None = None,
) -> str:
    serialized = serialize_academic_input(academic_input)

    if retry_findings is not None and previous_review is not None:
        findings_payload = [f.model_dump(mode="json") for f in retry_findings]
        summary = build_attempt_summary_json(previous_review)
        user_prompt = build_agent_retry_user_prompt(serialized, findings_payload, summary)
    else:
        user_prompt = build_baseline_user_prompt(serialized)

    return await complete_json(BASELINE_SYSTEM_PROMPT, user_prompt)
