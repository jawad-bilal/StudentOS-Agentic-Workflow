import pytest

from baseline.baseline import BaselineParseError, generate_baseline_plan
from core.llm_client import LlmNotConfiguredError
from evaluation.run_evaluation import load_case


@pytest.mark.asyncio
async def test_baseline_requires_api_key_without_fixtures(monkeypatch):
    case = load_case("case_01")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(LlmNotConfiguredError):
        await generate_baseline_plan(case.input)


@pytest.mark.asyncio
async def test_baseline_parse_error_on_invalid_json(monkeypatch):
    case = load_case("case_01")

    async def fake_complete(*_args, **_kwargs):
        return "not-json"

    monkeypatch.setattr("baseline.baseline.complete_json", fake_complete)
    with pytest.raises(BaselineParseError):
        await generate_baseline_plan(case.input)
