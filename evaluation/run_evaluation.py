"""Load evaluation cases and run deterministic evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from baseline.baseline import BaselineParseError, generate_baseline_plan
from core.grounding import analyze_grounding
from core.llm_client import LlmNotConfiguredError, get_llm_config
from core.models import AcademicInput, AcademicPlan, EvaluationCase, EvaluationResult
from core.prompt_serializer import build_prompt_payload, serialize_academic_input
from core.findings import AgentRunMetadata
from evaluation.scorer import evaluate_case_plan
from evaluation.verifier import verify_plan

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = Path(__file__).resolve().parent / "cases"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

load_dotenv(ROOT / ".env")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_case(case_id: str) -> EvaluationCase:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationCase.model_validate(data)


def list_case_ids() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("case_*.json"))


def apply_reference_now(case: EvaluationCase, reference_now: datetime | None) -> EvaluationCase:
    if reference_now is None:
        return case
    updated_input = case.input.model_copy(update={"reference_now": reference_now})
    return case.model_copy(update={"input": updated_input})


def load_fixture_plan(system: str, case_id: str) -> AcademicPlan:
    path = FIXTURES_DIR / system / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return AcademicPlan.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_manual_plan(path: Path) -> AcademicPlan:
    return AcademicPlan.model_validate(json.loads(path.read_text(encoding="utf-8")))


async def produce_plan(
    system: str,
    case: EvaluationCase,
    *,
    use_fixtures: bool,
    plan_fixture: Path | None,
) -> tuple[AcademicPlan | None, str | None, str, AgentRunMetadata | None]:
    if plan_fixture is not None:
        return load_manual_plan(plan_fixture), None, "manual", None

    if use_fixtures:
        try:
            return load_fixture_plan(system, case.metadata.case_id), None, "fixture", None
        except FileNotFoundError as exc:
            return None, str(exc), system, None

    if system == "baseline":
        try:
            plan = await generate_baseline_plan(case.input)
            return plan, None, "baseline", None
        except LlmNotConfiguredError as exc:
            return None, str(exc), "baseline", None
        except BaselineParseError as exc:
            return None, str(exc), "baseline", None
        except Exception as exc:  # noqa: BLE001 - surface LLM failures in CLI
            return None, str(exc), "baseline", None

    if system == "agent":
        try:
            from agent.orchestrator import generate_agent_plan

            plan, agent_meta, error = await generate_agent_plan(case.input)
            if plan is None:
                return None, error, "agent", agent_meta
            return plan, None, "agent", agent_meta
        except LlmNotConfiguredError as exc:
            return None, str(exc), "agent", None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc), "agent", None

    return None, f"Unsupported live system: {system}", system, None


def evaluate_loaded_plan(
    case: EvaluationCase,
    plan: AcademicPlan,
    system: str,
    runtime_ms: int,
    agent_meta: AgentRunMetadata | None = None,
) -> EvaluationResult:
    grounding = analyze_grounding(plan, case.input)
    verification = verify_plan(plan, case.input, grounding)
    score = evaluate_case_plan(case, plan, grounding, verification)

    serialized = serialize_academic_input(case.input)
    llm_config = get_llm_config()

    metadata: dict = {
        "reference_now": case.input.reference_now.isoformat(),
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
        "prompt_hash": _sha256_text(serialized),
        "input_hash": _sha256_text(json.dumps(build_prompt_payload(case.input), sort_keys=True)),
        "plan_hash": _sha256_text(plan.model_dump_json()),
        "grounding_findings": len(grounding.findings),
        "verification_issues": len(verification.issues),
    }
    if agent_meta is not None:
        metadata["agent"] = agent_meta.model_dump(mode="json")

    return EvaluationResult(
        case_id=case.metadata.case_id,
        system=system,  # type: ignore[arg-type]
        plan=plan,
        grounding=grounding,
        verification=verification,
        score=score,
        total_score=score.total,
        runtime_ms=runtime_ms,
        metadata=metadata,
    )


async def run_single(
    case_id: str,
    system: str,
    *,
    reference_now: datetime | None,
    use_fixtures: bool,
    plan_fixture: Path | None,
) -> EvaluationResult:
    started = time.perf_counter()
    case = apply_reference_now(load_case(case_id), reference_now)
    plan, error, resolved_system, agent_meta = await produce_plan(
        system,
        case,
        use_fixtures=use_fixtures,
        plan_fixture=plan_fixture,
    )
    runtime_ms = int((time.perf_counter() - started) * 1000)

    if plan is None:
        meta = {"reference_now": case.input.reference_now.isoformat()}
        if agent_meta is not None:
            meta["agent"] = agent_meta.model_dump(mode="json")
        return EvaluationResult(
            case_id=case_id,
            system=resolved_system,  # type: ignore[arg-type]
            runtime_ms=runtime_ms,
            error=error,
            metadata=meta,
        )

    agent_runtime = agent_meta.total_runtime_ms if agent_meta else runtime_ms
    result = evaluate_loaded_plan(case, plan, resolved_system, agent_runtime, agent_meta)
    return result


def _parse_reference_now(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic academic plan evaluation")
    parser.add_argument("--system", choices=["baseline", "agent", "none"], default="baseline")
    parser.add_argument("--case", help="Case id, e.g. case_01")
    parser.add_argument("--all", action="store_true", help="Run all cases")
    parser.add_argument("--fixtures", action="store_true", help="Load cached plan fixtures")
    parser.add_argument(
        "--reference-now",
        help="Override reference_now for all cases (ISO 8601)",
    )
    parser.add_argument(
        "--plan-fixture",
        type=Path,
        help="Evaluate a specific AcademicPlan JSON file",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    return parser


async def main_async(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if os.getenv("EVAL_USE_FIXTURES", "").lower() in {"1", "true", "yes"}:
        args.fixtures = True

    reference_now = _parse_reference_now(args.reference_now)

    if args.plan_fixture is not None:
        if not args.case:
            print("--case is required with --plan-fixture", file=sys.stderr)
            return 2
        result = await run_single(
            args.case,
            "none",
            reference_now=reference_now,
            use_fixtures=False,
            plan_fixture=args.plan_fixture,
        )
        results = [result]
    elif args.all:
        case_ids = list_case_ids()
        if not case_ids:
            print("No cases found", file=sys.stderr)
            return 2
        system = args.system if args.system != "none" else "baseline"
        results = []
        for case_id in case_ids:
            results.append(
                await run_single(
                    case_id,
                    system,
                    reference_now=reference_now,
                    use_fixtures=args.fixtures,
                    plan_fixture=None,
                )
            )
    elif args.case:
        system = args.system if args.system != "none" else "baseline"
        results = [
            await run_single(
                args.case,
                system,
                reference_now=reference_now,
                use_fixtures=args.fixtures,
                plan_fixture=None,
            )
        ]
    else:
        parser.print_help()
        return 2

    exit_code = 0
    for result in results:
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            status = "ERROR" if result.error else f"score={result.total_score:.1f}"
            print(f"{result.case_id} [{result.system}] {status} ({result.runtime_ms}ms)")
            if result.error:
                print(f"  error: {result.error}")
            elif result.score:
                print(
                    "  breakdown: "
                    f"deadlines={result.score.deadlines_respected:.1f}, "
                    f"hallucinations={result.score.no_hallucinations:.1f}, "
                    f"conflicts={result.score.no_timetable_conflicts:.1f}, "
                    f"workload={result.score.workload_distribution:.1f}, "
                    f"prioritization={result.score.prioritization:.1f}, "
                    f"missing={result.score.missing_information_handling:.1f}"
                )
        if result.error:
            exit_code = 1

    return exit_code


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
