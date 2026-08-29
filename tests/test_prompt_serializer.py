from core.prompt_serializer import build_prompt_payload, serialize_academic_input
from evaluation.run_evaluation import load_case


def test_serialized_payload_has_evaluation_context():
    case = load_case("case_03")
    payload = build_prompt_payload(case.input)
    assert "evaluation_context" in payload
    assert payload["evaluation_context"]["reference_now"] == case.input.reference_now.isoformat()
    assert payload["evaluation_context"]["timezone"] == "Asia/Karachi"
    assert "instruction" in payload["evaluation_context"]


def test_serialized_payload_is_stable_json():
    case = load_case("case_03")
    text = serialize_academic_input(case.input)
    assert "assignment_1" in text
    assert "class_os_mon" in text
