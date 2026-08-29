"""Agent workflow orchestrator: plan, verify, optional retry (max 2 LLM calls)."""

from __future__ import annotations

import hashlib
import time

from core.findings import AgentRunMetadata, AttemptReview
from core.llm_client import get_llm_config
from core.models import AcademicInput, AcademicPlan
from agent.failure_policy import select_best_attempt
from agent.planner import call_planning_agent
from agent.review import review_attempt

MAX_LLM_CALLS = 2


def _plan_hash(plan: AcademicPlan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


async def generate_agent_plan(
    academic_input: AcademicInput,
) -> tuple[AcademicPlan | None, AgentRunMetadata, str | None]:
    started = time.perf_counter()
    llm_config = get_llm_config()
    llm_calls = 0
    retry_occurred = False

    raw_first = await call_planning_agent(academic_input)
    llm_calls = 1
    first_review = review_attempt(
        raw_first,
        academic_input,
        attempt="first",
        llm_call_number=1,
    )

    retry_review: AttemptReview | None = None
    if first_review.has_actionable_findings and llm_calls < MAX_LLM_CALLS:
        retry_occurred = True
        findings = first_review.actionable_findings()
        raw_retry = await call_planning_agent(
            academic_input,
            retry_findings=findings,
            previous_review=first_review,
        )
        llm_calls = 2
        retry_review = review_attempt(
            raw_retry,
            academic_input,
            attempt="retry",
            llm_call_number=2,
        )

    if retry_review is not None:
        plan, selected, error = select_best_attempt(first_review, retry_review)
    else:
        plan, selected, error = select_best_attempt(first_review, None)

    selected_review = retry_review if selected == "retry" else first_review
    if plan is None:
        selected_review = retry_review or first_review

    runtime_ms = int((time.perf_counter() - started) * 1000)

    metadata = AgentRunMetadata(
        llm_calls=llm_calls,
        retry_occurred=retry_occurred,
        selected_attempt=selected,
        initial_validation_findings=first_review.validation_findings,
        initial_grounding_findings=first_review.grounding.findings,
        initial_verification_findings=(
            first_review.verification.issues if first_review.verification else []
        ),
        final_validation_ok=selected_review.is_valid_plan if plan is not None else False,
        final_grounding_ok=(
            not selected_review.grounding.has_findings if plan is not None else False
        ),
        final_verification_ok=(
            selected_review.verification.status.value == "pass"
            if plan and selected_review.verification
            else False
        ),
        final_plan_hash=_plan_hash(plan) if plan else "",
        total_runtime_ms=runtime_ms,
        llm_provider=llm_config.provider,
        llm_model=llm_config.model,
    )

    if plan is None:
        return None, metadata, error or "Agent failed to produce a valid AcademicPlan"

    return plan, metadata, None
