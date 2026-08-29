"""Deterministic Plan Validity Score (0-100)."""

from __future__ import annotations

from core.models import (
    AcademicInput,
    AcademicPlan,
    EvaluationCase,
    GroundTruthConstraints,
    GroundingAnalysis,
    Priority,
    ScoreBreakdown,
    VerificationIssueType,
    VerificationResult,
    session_duration_minutes,
)
from evaluation.constraints import study_minutes_by_date


def _deadline_score(
    plan: AcademicPlan,
    academic_input: AcademicInput,
    ground_truth: GroundTruthConstraints,
    verification: VerificationResult,
) -> float:
    score = 25.0
    deadline_violations = {
        i.task_id
        for i in verification.issues
        if i.type == VerificationIssueType.DEADLINE_VIOLATION
    }

    for deadline_id in ground_truth.must_schedule_before:
        sessions = [s for s in plan.proposed_sessions if s.task_id == deadline_id]
        if not sessions:
            score -= 25.0 / max(len(ground_truth.must_schedule_before), 1)
            continue
        deadline = next(d for d in academic_input.deadlines if d.id == deadline_id)
        if deadline.due_at is None:
            score -= 5.0
            continue
        if deadline_id in deadline_violations:
            score -= 25.0 / max(len(ground_truth.must_schedule_before), 1)

    for deadline_id in ground_truth.min_prep_minutes_for:
        required = ground_truth.min_prep_minutes_for[deadline_id]
        actual = sum(
            session_duration_minutes(s.start, s.end)
            for s in plan.proposed_sessions
            if s.task_id == deadline_id
        )
        if actual < required:
            score -= min(10.0, 25.0 / max(len(ground_truth.min_prep_minutes_for), 1))

    return max(0.0, score)


def _hallucination_score(verification: VerificationResult, ground_truth: GroundTruthConstraints) -> float:
    if not ground_truth.must_not_invent:
        return 20.0
    bad = [
        i
        for i in verification.issues
        if i.type
        in {
            VerificationIssueType.UNKNOWN_TASK_ID,
            VerificationIssueType.HALLUCINATED_TASK,
            VerificationIssueType.HALLUCINATED_DEADLINE,
            VerificationIssueType.INVENTED_DEADLINE_DATE,
        }
    ]
    if not bad:
        return 20.0
    return max(0.0, 20.0 - 5.0 * len(bad))


def _timetable_score(
    plan: AcademicPlan,
    ground_truth: GroundTruthConstraints,
    verification: VerificationResult,
) -> float:
    overlap_issues = [
        i
        for i in verification.issues
        if i.type
        in {
            VerificationIssueType.TIMETABLE_OVERLAP,
            VerificationIssueType.EXISTING_EVENT_OVERLAP,
            VerificationIssueType.SESSION_OVERLAP,
        }
    ]
    if overlap_issues:
        return max(0.0, 20.0 - 4.0 * len(overlap_issues))

    if ground_truth.must_not_overlap:
        blocked_ids = set(ground_truth.must_not_overlap)
        for issue in verification.issues:
            if issue.conflicting_ref and issue.conflicting_ref.id in blocked_ids:
                return 0.0
        return 20.0

    return 20.0 if plan.proposed_sessions else 15.0


def _workload_score(
    plan: AcademicPlan,
    academic_input: AcademicInput,
    ground_truth: GroundTruthConstraints,
    verification: VerificationResult,
) -> float:
    score = 15.0
    overload = [
        i for i in verification.issues if i.type == VerificationIssueType.WORKLOAD_EXCEEDED
    ]
    if overload:
        score -= min(15.0, 5.0 * len(overload))

    max_hours = ground_truth.max_study_hours_per_day or academic_input.preferences.max_study_hours_per_day
    minutes_by_date = study_minutes_by_date(plan.proposed_sessions)
    if minutes_by_date:
        peak_hours = max(m / 60.0 for m in minutes_by_date.values())
        if peak_hours > max_hours:
            score -= 5.0

    if ground_truth.expect_spread_across_days and len(minutes_by_date) <= 1 and len(plan.proposed_sessions) > 2:
        score -= 5.0

    if ground_truth.min_distinct_study_days is not None:
        if len(minutes_by_date) < ground_truth.min_distinct_study_days:
            score -= 5.0

    return max(0.0, score)


def _prioritization_score(plan: AcademicPlan, ground_truth: GroundTruthConstraints) -> float:
    score = 10.0
    if not ground_truth.expect_high_priority_for:
        return score if plan.task_priorities else 8.0

    high_ids = {
        item.task_id
        for item in plan.task_priorities
        if item.priority == Priority.HIGH
    }
    for deadline_id in ground_truth.expect_high_priority_for:
        if deadline_id not in high_ids:
            score -= 10.0 / max(len(ground_truth.expect_high_priority_for), 1)

    return max(0.0, score)


def _missing_info_score(plan: AcademicPlan, ground_truth: GroundTruthConstraints) -> float:
    score = 10.0
    clarified = {c.task_id for c in plan.clarification_required}

    for deadline_id in ground_truth.must_flag_missing:
        if deadline_id not in clarified:
            score -= 5.0

    for deadline_id in ground_truth.expect_clarification_for:
        if deadline_id not in clarified:
            score -= 5.0 / max(len(ground_truth.expect_clarification_for), 1)

    for deadline_id in ground_truth.must_not_schedule:
        scheduled = any(s.task_id == deadline_id for s in plan.proposed_sessions)
        if scheduled:
            score -= 5.0

    if ground_truth.expect_insufficient_capacity:
        text = (plan.explanation + " ".join(plan.assumptions)).lower()
        capacity_words = ("cannot fit", "not enough time", "insufficient", "overload", "trade-off", "tradeoff")
        if not any(word in text for word in capacity_words):
            score -= 5.0

    return max(0.0, score)


def score_plan(
    plan: AcademicPlan,
    academic_input: AcademicInput,
    ground_truth: GroundTruthConstraints,
    verification: VerificationResult,
    _grounding: GroundingAnalysis | None = None,
) -> ScoreBreakdown:
    """Same scoring for baseline and agent; grounding is already reflected in verification."""
    return ScoreBreakdown(
        deadlines_respected=_deadline_score(plan, academic_input, ground_truth, verification),
        no_hallucinations=_hallucination_score(verification, ground_truth),
        no_timetable_conflicts=_timetable_score(plan, ground_truth, verification),
        workload_distribution=_workload_score(plan, academic_input, ground_truth, verification),
        prioritization=_prioritization_score(plan, ground_truth),
        missing_information_handling=_missing_info_score(plan, ground_truth),
    )


def evaluate_case_plan(
    case: EvaluationCase,
    plan: AcademicPlan,
    grounding: GroundingAnalysis,
    verification: VerificationResult,
) -> ScoreBreakdown:
    return score_plan(plan, case.input, case.ground_truth, verification, grounding)
