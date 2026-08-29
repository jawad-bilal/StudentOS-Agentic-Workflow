import json
from datetime import datetime
from pathlib import Path

import pytest

from core.models import AcademicInput, AcademicPlan, EvaluationCase


CASES_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "cases"


def test_all_case_files_validate():
    case_paths = sorted(CASES_DIR.glob("case_*.json"))
    assert len(case_paths) == 10
    for path in case_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        case = EvaluationCase.model_validate(data)
        assert case.metadata.case_id == path.stem


def test_study_session_rejects_cross_midnight():
    with pytest.raises(ValueError):
        AcademicPlan.model_validate(
            {
                "student_id": "s",
                "reference_now": "2026-09-07T08:00:00+05:00",
                "proposed_sessions": [
                    {
                        "session_id": "sess_1",
                        "task_id": "assignment_1",
                        "title": "Late study",
                        "date": "2026-09-07",
                        "start": "23:00",
                        "end": "01:00",
                        "priority": "high",
                    }
                ],
            }
        )


def test_clarification_item_structure():
    plan = AcademicPlan.model_validate(
        {
            "student_id": "demo_student_06",
            "reference_now": "2026-09-07T08:00:00+05:00",
            "clarification_required": [
                {
                    "task_id": "assignment_1",
                    "field": "due_at",
                    "reason": "Due date is missing",
                }
            ],
        }
    )
    assert plan.clarification_required[0].field == "due_at"
