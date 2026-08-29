import json
from pathlib import Path

from core.grounding import analyze_grounding
from core.models import AcademicPlan, GroundingFindingType
from evaluation.run_evaluation import load_case


def test_unknown_task_id_finding():
    case = load_case("case_01")
    plan = AcademicPlan.model_validate(
        {
            "student_id": case.input.student_id,
            "reference_now": case.input.reference_now.isoformat(),
            "proposed_sessions": [
                {
                    "session_id": "sess_bad",
                    "task_id": "does_not_exist",
                    "title": "Bad",
                    "date": "2026-09-07",
                    "start": "14:00",
                    "end": "15:00",
                    "priority": "high",
                }
            ],
        }
    )
    grounding = analyze_grounding(plan, case.input)
    assert any(f.type == GroundingFindingType.UNKNOWN_TASK_ID for f in grounding.findings)


def test_estimated_minutes_mismatch():
    case = load_case("case_01")
    plan = AcademicPlan.model_validate(
        {
            "student_id": case.input.student_id,
            "reference_now": case.input.reference_now.isoformat(),
            "proposed_sessions": [
                {
                    "session_id": "sess_1",
                    "task_id": "quiz_1",
                    "title": "Quiz prep",
                    "date": "2026-09-07",
                    "start": "14:00",
                    "end": "16:00",
                    "priority": "high",
                    "estimated_minutes": 60,
                }
            ],
        }
    )
    grounding = analyze_grounding(plan, case.input)
    assert any(
        f.type == GroundingFindingType.ESTIMATED_MINUTES_MISMATCH for f in grounding.findings
    )
