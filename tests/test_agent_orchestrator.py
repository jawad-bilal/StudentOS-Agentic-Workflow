"""Tests for agent orchestrator call counts and retry behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agent.orchestrator import generate_agent_plan
from evaluation.run_evaluation import load_case


VALID_PLAN_JSON = {
    "student_id": "demo_student_01",
    "reference_now": "2026-09-07T08:00:00+05:00",
    "proposed_sessions": [
        {
            "session_id": "sess_001",
            "task_id": "quiz_1",
            "title": "Quiz prep",
            "date": "2026-09-07",
            "start": "14:00",
            "end": "16:00",
            "priority": "high",
            "estimated_minutes": 120,
        },
        {
            "session_id": "sess_002",
            "task_id": "assignment_1",
            "title": "Assignment work",
            "date": "2026-09-08",
            "start": "11:00",
            "end": "13:00",
            "priority": "high",
            "estimated_minutes": 120,
        },
    ],
    "task_priorities": [
        {"task_id": "quiz_1", "rank": 1, "priority": "high", "reason": "soon"},
        {"task_id": "assignment_1", "rank": 2, "priority": "high", "reason": "next"},
    ],
    "total_estimated_minutes": 240,
    "explanation": "Plan covers quiz and assignment.",
    "assumptions": [],
    "clarification_required": [],
}


INVALID_ENUM_JSON = {
    **VALID_PLAN_JSON,
    "proposed_sessions": [
        {
            **VALID_PLAN_JSON["proposed_sessions"][0],
            "priority": "medium-high",
        }
    ],
}


OVERLAP_JSON = {
    "student_id": "demo_student_03",
    "reference_now": "2026-09-07T08:00:00+05:00",
    "proposed_sessions": [
        {
            "session_id": "sess_bad",
            "task_id": "assignment_1",
            "title": "Bad overlap",
            "date": "2026-09-08",
            "start": "14:30",
            "end": "16:30",
            "priority": "high",
            "estimated_minutes": 120,
        }
    ],
    "task_priorities": [
        {"task_id": "assignment_1", "rank": 1, "priority": "high", "reason": "due"},
    ],
    "total_estimated_minutes": 120,
    "explanation": "Overlap with Monday class on 2026-09-08... wait weekday",
    "assumptions": [],
    "clarification_required": [],
}


@pytest.mark.asyncio
async def test_first_pass_success_one_llm_call():
    case = load_case("case_01")
    mock = AsyncMock(return_value=json.dumps(VALID_PLAN_JSON))
    with patch("agent.orchestrator.call_planning_agent", mock):
        plan, meta, err = await generate_agent_plan(case.input)
    assert err is None
    assert plan is not None
    assert meta.llm_calls == 1
    assert meta.retry_occurred is False


@pytest.mark.asyncio
async def test_schema_failure_triggers_retry():
    case = load_case("case_01")
    invalid = {
        **VALID_PLAN_JSON,
        "proposed_sessions": [
            {**VALID_PLAN_JSON["proposed_sessions"][0], "priority": "medium-high"}
        ],
    }
    mock = AsyncMock(side_effect=[json.dumps(invalid), json.dumps(VALID_PLAN_JSON)])
    with patch("agent.orchestrator.call_planning_agent", mock):
        plan, meta, err = await generate_agent_plan(case.input)
    assert mock.call_count == 2
    assert meta.llm_calls == 2
    assert meta.retry_occurred is True
    assert plan is not None
    assert err is None


@pytest.mark.asyncio
async def test_verification_failure_triggers_retry():
    case = load_case("case_03")
    # 2026-09-08 is Monday; class_os_mon is 14:00-16:00
    overlap = {
        "student_id": case.input.student_id,
        "reference_now": case.input.reference_now.isoformat(),
        "proposed_sessions": [
            {
                "session_id": "sess_bad",
                "task_id": "assignment_1",
                "title": "Overlap",
                "date": "2026-09-07",
                "start": "14:30",
                "end": "16:30",
                "priority": "high",
                "estimated_minutes": 120,
            }
        ],
        "task_priorities": [
            {"task_id": "assignment_1", "rank": 1, "priority": "high", "reason": "x"},
        ],
        "total_estimated_minutes": 120,
        "explanation": "bad",
        "assumptions": [],
        "clarification_required": [],
    }
    fixed = {
        **overlap,
        "proposed_sessions": [
            {
                "session_id": "sess_ok",
                "task_id": "assignment_1",
                "title": "Good",
                "date": "2026-09-07",
                "start": "10:00",
                "end": "12:00",
                "priority": "high",
                "estimated_minutes": 120,
            }
        ],
    }
    mock = AsyncMock(side_effect=[json.dumps(overlap), json.dumps(fixed)])
    with patch("agent.orchestrator.call_planning_agent", mock):
        plan, meta, err = await generate_agent_plan(case.input)
    assert mock.call_count == 2
    assert plan is not None
    assert meta.final_verification_ok is True


@pytest.mark.asyncio
async def test_retry_failure_no_third_call():
    case = load_case("case_01")
    invalid = {
        **VALID_PLAN_JSON,
        "proposed_sessions": [
            {**VALID_PLAN_JSON["proposed_sessions"][0], "priority": "medium-high"}
        ],
    }
    mock = AsyncMock(side_effect=[json.dumps(invalid), json.dumps(invalid)])
    with patch("agent.orchestrator.call_planning_agent", mock):
        plan, meta, err = await generate_agent_plan(case.input)
    assert mock.call_count == 2
    assert meta.llm_calls == 2
    assert plan is None
    assert err is not None


@pytest.mark.asyncio
async def test_no_post_hoc_repair():
    case = load_case("case_01")
    raw = json.dumps(VALID_PLAN_JSON)
    mock = AsyncMock(return_value=raw)
    with patch("agent.orchestrator.call_planning_agent", mock):
        plan, _, _ = await generate_agent_plan(case.input)
    assert plan is not None
    from core.models import AcademicPlan

    expected = AcademicPlan.model_validate(VALID_PLAN_JSON)
    assert plan.model_dump(mode="json") == expected.model_dump(mode="json")
