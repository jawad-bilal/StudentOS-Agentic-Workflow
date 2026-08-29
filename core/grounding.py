"""Grounding analysis separate from Pydantic schema validation."""

from __future__ import annotations

from core.models import (
    AcademicInput,
    AcademicPlan,
    GroundingAnalysis,
    GroundingFinding,
    GroundingFindingType,
    validate_session_temporal,
)


def analyze_grounding(plan: AcademicPlan, academic_input: AcademicInput) -> GroundingAnalysis:
    """
    Check plan references against source input.

    Both baseline and agent receive identical grounding analysis.
    Findings feed the shared deterministic evaluator; they do not reject
    one system differently from the other.
    """
    findings: list[GroundingFinding] = []
    deadline_ids = academic_input.deadline_ids()
    deadlines_by_id = {d.id: d for d in academic_input.deadlines}

    if plan.student_id != academic_input.student_id:
        findings.append(
            GroundingFinding(
                type=GroundingFindingType.STUDENT_ID_MISMATCH,
                message=(
                    f"Plan student_id {plan.student_id!r} does not match "
                    f"input {academic_input.student_id!r}"
                ),
            )
        )

    if plan.reference_now != academic_input.reference_now:
        findings.append(
            GroundingFinding(
                type=GroundingFindingType.REFERENCE_NOW_MISMATCH,
                message="Plan reference_now does not match input reference_now",
            )
        )

    for session in plan.proposed_sessions:
        if session.task_id is not None and session.task_id not in deadline_ids:
            findings.append(
                GroundingFinding(
                    type=GroundingFindingType.UNKNOWN_TASK_ID,
                    message=f"Session references unknown task_id {session.task_id!r}",
                    session_id=session.session_id,
                    task_id=session.task_id,
                )
            )
        elif session.task_id is not None:
            deadline = deadlines_by_id[session.task_id]
            if deadline.due_at is None:
                findings.append(
                    GroundingFinding(
                        type=GroundingFindingType.SCHEDULED_TASK_WITH_MISSING_DUE,
                        message=(
                            f"Session schedules task {session.task_id!r} which has no due_at"
                        ),
                        session_id=session.session_id,
                        task_id=session.task_id,
                    )
                )

        findings.extend(
            validate_session_temporal(session, academic_input.preferences)
        )

    for item in plan.task_priorities:
        if item.task_id not in deadline_ids:
            findings.append(
                GroundingFinding(
                    type=GroundingFindingType.UNKNOWN_PRIORITY_TASK_ID,
                    message=f"Priority list references unknown task_id {item.task_id!r}",
                    task_id=item.task_id,
                )
            )

    for item in plan.clarification_required:
        if item.task_id not in deadline_ids:
            findings.append(
                GroundingFinding(
                    type=GroundingFindingType.UNKNOWN_TASK_ID,
                    message=(
                        f"clarification_required references unknown task_id {item.task_id!r}"
                    ),
                    task_id=item.task_id,
                )
            )

    return GroundingAnalysis(findings=findings)
