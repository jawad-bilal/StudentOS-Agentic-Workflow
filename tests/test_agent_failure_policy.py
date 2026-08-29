"""Tests for validity-first failure policy."""

from core.findings import AttemptReview
from core.models import (
    AcademicPlan,
    GroundingAnalysis,
    GroundingFinding,
    GroundingFindingType,
    VerificationIssue,
    VerificationIssueType,
    VerificationResult,
    VerificationStatus,
)
from agent.failure_policy import select_best_attempt


def _valid_plan() -> AcademicPlan:
    return AcademicPlan.model_validate(
        {
            "student_id": "s",
            "reference_now": "2026-09-07T08:00:00+05:00",
            "proposed_sessions": [],
        }
    )


def test_prefers_valid_over_invalid():
    first = AttemptReview(
        attempt="first",
        llm_call_number=1,
        parse_error="bad json",
    )
    retry = AttemptReview(
        attempt="retry",
        llm_call_number=2,
        plan=_valid_plan(),
        grounding=GroundingAnalysis(),
        verification=VerificationResult(status=VerificationStatus.PASS),
    )
    plan, label, err = select_best_attempt(first, retry)
    assert err is None
    assert label == "retry"
    assert plan is not None


def test_prefers_clean_over_dirty():
    plan = _valid_plan()
    first = AttemptReview(
        attempt="first",
        llm_call_number=1,
        plan=plan,
        grounding=GroundingAnalysis(
            findings=[
                GroundingFinding(
                    type=GroundingFindingType.UNKNOWN_TASK_ID,
                    message="bad",
                    task_id="x",
                )
            ]
        ),
        verification=VerificationResult(
            status=VerificationStatus.FAIL,
            issues=[
                VerificationIssue(
                    type=VerificationIssueType.UNKNOWN_TASK_ID,
                    message="bad",
                    task_id="x",
                )
            ],
        ),
    )
    retry = AttemptReview(
        attempt="retry",
        llm_call_number=2,
        plan=plan,
        grounding=GroundingAnalysis(),
        verification=VerificationResult(status=VerificationStatus.PASS),
    )
    _, label, _ = select_best_attempt(first, retry)
    assert label == "retry"


def test_prefers_fewer_issues_when_both_dirty():
    plan = _valid_plan()
    first = AttemptReview(
        attempt="first",
        llm_call_number=1,
        plan=plan,
        grounding=GroundingAnalysis(),
        verification=VerificationResult(
            status=VerificationStatus.FAIL,
            issues=[
                VerificationIssue(type=VerificationIssueType.DEADLINE_VIOLATION, message="a"),
                VerificationIssue(type=VerificationIssueType.DEADLINE_VIOLATION, message="b"),
            ],
        ),
    )
    retry = AttemptReview(
        attempt="retry",
        llm_call_number=2,
        plan=plan,
        grounding=GroundingAnalysis(),
        verification=VerificationResult(
            status=VerificationStatus.FAIL,
            issues=[
                VerificationIssue(type=VerificationIssueType.TIMETABLE_OVERLAP, message="c"),
            ],
        ),
    )
    _, label, _ = select_best_attempt(first, retry)
    assert label == "retry"


def test_neither_valid_returns_none():
    first = AttemptReview(attempt="first", llm_call_number=1, parse_error="x")
    retry = AttemptReview(attempt="retry", llm_call_number=2, parse_error="y")
    plan, label, err = select_best_attempt(first, retry)
    assert plan is None
    assert label is None
    assert err is not None
