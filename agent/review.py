"""Parse, validate, ground, and verify one planning attempt."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from core.findings import AttemptReview, validation_error_to_findings
from core.grounding import analyze_grounding
from core.json_parser import JsonParseError, parse_llm_json
from core.models import AcademicInput, AcademicPlan
from evaluation.verifier import verify_plan


def review_attempt(
    raw_text: str,
    academic_input: AcademicInput,
    *,
    attempt: Literal["first", "retry"],
    llm_call_number: int,
) -> AttemptReview:
    review = AttemptReview(
        attempt=attempt,
        llm_call_number=llm_call_number,
        raw_text=raw_text,
    )

    try:
        parsed = parse_llm_json(raw_text)
    except JsonParseError as exc:
        review.parse_error = str(exc)
        return review

    try:
        plan = AcademicPlan.model_validate(parsed)
    except ValidationError as exc:
        review.validation_findings = validation_error_to_findings(exc)
        return review

    review.plan = plan
    review.grounding = analyze_grounding(plan, academic_input)
    review.verification = verify_plan(plan, academic_input, review.grounding)
    return review
