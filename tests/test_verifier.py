import json
from pathlib import Path

from core.grounding import analyze_grounding
from core.models import AcademicPlan, VerificationIssueType
from evaluation.run_evaluation import load_case
from evaluation.scorer import evaluate_case_plan
from evaluation.verifier import verify_plan


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "baseline"
    / "case_01.json"
)


def test_case_01_fixture_avoids_timetable_overlap():
    case = load_case("case_01")
    plan = AcademicPlan.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    grounding = analyze_grounding(plan, case.input)
    verification = verify_plan(plan, case.input, grounding)

    overlap_issues = [
        i
        for i in verification.issues
        if i.type
        in {
            VerificationIssueType.TIMETABLE_OVERLAP,
            VerificationIssueType.SESSION_OVERLAP,
            VerificationIssueType.EXISTING_EVENT_OVERLAP,
        }
    ]
    assert not overlap_issues


def test_case_01_fixture_scores_well():
    case = load_case("case_01")
    plan = AcademicPlan.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    grounding = analyze_grounding(plan, case.input)
    verification = verify_plan(plan, case.input, grounding)
    score = evaluate_case_plan(case, plan, grounding, verification)
    assert score.total >= 80.0


def test_case_06_missing_deadline_requires_clarification():
    case = load_case("case_06")
    plan = AcademicPlan.model_validate(
        {
            "student_id": case.input.student_id,
            "reference_now": case.input.reference_now.isoformat(),
            "proposed_sessions": [],
            "clarification_required": [
                {
                    "task_id": "assignment_1",
                    "field": "due_at",
                    "reason": "Due date missing",
                }
            ],
        }
    )
    grounding = analyze_grounding(plan, case.input)
    verification = verify_plan(plan, case.input, grounding)
    score = evaluate_case_plan(case, plan, grounding, verification)
    assert score.missing_information_handling >= 8.0
