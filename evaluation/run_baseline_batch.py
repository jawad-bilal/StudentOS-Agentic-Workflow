"""Run live baseline across all cases and save reproducible artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV = ROOT.parent / "backend" / ".env"
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "baseline"


def load_api_key() -> None:
    if os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"):
        return
    if BACKEND_ENV.exists():
        values = dotenv_values(BACKEND_ENV)
        key = (values.get("GEMINI_API_KEY") or "").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key
            return
    raise SystemExit(
        "GEMINI_API_KEY not found. Set it in agentic-workflow/.env or backend/.env"
    )


async def main() -> int:
    load_api_key()

    from core.llm_client import get_llm_config
    from evaluation.run_evaluation import list_case_ids, run_single

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = get_llm_config()
    manifest = {
        "run_id": run_id,
        "system": "baseline",
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cases": [],
    }

    results = []
    for case_id in list_case_ids():
        print(f"Running {case_id}...", flush=True)
        result = await run_single(
            case_id,
            "baseline",
            reference_now=None,
            use_fixtures=False,
            plan_fixture=None,
        )
        results.append(result)

        case_dir = out_dir / case_id
        case_dir.mkdir(exist_ok=True)
        (case_dir / "result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        if result.plan is not None:
            (case_dir / "plan.json").write_text(
                result.plan.model_dump_json(indent=2), encoding="utf-8"
            )
        manifest["cases"].append(
            {
                "case_id": case_id,
                "total_score": result.total_score,
                "error": result.error,
                "runtime_ms": result.runtime_ms,
                "metadata": result.metadata,
            }
        )

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summary_path = out_dir / "summary.json"
    scored = [r for r in results if r.error is None and r.score is not None]
    scores = [r.total_score for r in scored]
    summary = {
        "run_id": run_id,
        "case_count": len(results),
        "parse_failures": sum(1 for r in results if r.error is not None),
        "mean_score": statistics.mean(scores) if scores else 0,
        "median_score": statistics.median(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "average_runtime_ms": statistics.mean([r.runtime_ms for r in results])
        if results
        else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Artifacts saved to: {out_dir}")
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
