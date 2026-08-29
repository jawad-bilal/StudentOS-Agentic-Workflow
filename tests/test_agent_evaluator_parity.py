"""Evaluator parity and deterministic review tests."""

import json

from core.grounding import analyze_grounding
from evaluation.run_evaluation import evaluate_loaded_plan, load_case
from evaluation.verifier import verify_plan
from agent.review import review_attempt


FIXTURE_PLAN = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "baseline"
    / "case_01.json"
)


def test_same_evaluator_baseline_and_agent_labels():
    case = load_case("case_01")
    plan = json.loads(FIXTURE_PLAN.read_text(encoding="utf-8"))
    from core.models import AcademicPlan

    academic_plan = AcademicPlan.model_validate(plan)
    baseline_result = evaluate_loaded_plan(case, academic_plan, "baseline", 100)
    agent_result = evaluate_loaded_plan(case, academic_plan, "agent", 100)
    assert baseline_result.total_score == agent_result.total_score
    assert baseline_result.score == agent_result.score


def test_review_deterministic():
    case = load_case("case_01")
    raw = FIXTURE_PLAN.read_text(encoding="utf-8")
    a = review_attempt(raw, case.input, attempt="first", llm_call_number=1)
    b = review_attempt(raw, case.input, attempt="first", llm_call_number=1)
    assert a.model_dump() == b.model_dump()


def test_retry_prompt_excludes_ground_truth():
    from core.prompt_serializer import build_agent_retry_user_prompt, serialize_academic_input

    case = load_case("case_03")
    prompt = build_agent_retry_user_prompt(
        serialize_academic_input(case.input),
        [{"source": "validation", "type": "invalid_enum", "message": "x", "required_change": "y"}],
        '{"attempt":"first","plan_produced":false}',
    )
    assert "must_schedule_before" not in prompt
    assert "ground_truth" not in prompt
    assert "case_03" not in prompt
