import pytest

from evaluation.run_evaluation import main_async


@pytest.mark.asyncio
async def test_fixture_run_case_01():
    code = await main_async(["--system", "baseline", "--case", "case_01", "--fixtures"])
    assert code == 0


@pytest.mark.asyncio
async def test_missing_key_without_fixtures_returns_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    code = await main_async(["--system", "baseline", "--case", "case_01"])
    assert code == 1
