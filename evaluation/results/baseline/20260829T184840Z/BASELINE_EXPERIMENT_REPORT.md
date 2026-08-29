# Baseline Experiment Report

**Run ID:** `20260829T184840Z`  
**System:** baseline (single LLM call)  
**Model:** `gemini-2.5-flash` via Gemini API  
**Temperature:** 0  
**Started:** 2026-08-29T18:48:40Z  
**Finished:** 2026-08-29T18:51:34Z  

**Artifacts:** `evaluation/results/baseline/20260829T184840Z/`  
- `manifest.json` - run metadata and per-case hashes  
- `summary.json` - aggregate metrics  
- `case_XX/plan.json` - raw `AcademicPlan` output (when parse succeeded)  
- `case_XX/result.json` - full evaluation result  

**Reproduction:**

```bash
cd agentic-workflow
# GEMINI_API_KEY required (or present in ../backend/.env)
python -m evaluation.run_baseline_batch
# or
python -m evaluation.run_evaluation --system baseline --all
```

---

## 1. Per-case results

| case_id | total | deadlines | hallucinations | conflicts | workload | prioritization | missing | runtime_ms | failure |
|---------|------:|----------:|---------------:|----------:|---------:|---------------:|--------:|-----------:|---------|
| case_01 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 16684 | - |
| case_02 | 0.0 | - | - | - | - | - | - | 17532 | **Schema validation:** invalid priority enum `medium-high` (3 fields) |
| case_03 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 13219 | - |
| case_04 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 14823 | - |
| case_05 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 18572 | - |
| case_06 | 93.0 | 25.0 | 20.0 | 15.0 | 15.0 | 8.0 | 10.0 | 6667 | Scorer-only (see analysis) |
| case_07 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 12471 | - |
| case_08 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 20244 | - |
| case_09 | 95.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 5.0 | 28583 | Scorer-only (see analysis) |
| case_10 | 100.0 | 25.0 | 20.0 | 20.0 | 15.0 | 10.0 | 10.0 | 24888 | - |

---

## 2. Aggregate metrics

### Scores (parsed plans only, n=9)

| Metric | Value |
|--------|------:|
| Mean | 98.67 |
| Median | 100.0 |
| Min | 93.0 |
| Max | 100.0 |

### Scores (all 10 cases, parse failure = 0)

| Metric | Value |
|--------|------:|
| Mean | 88.8 |
| Median | 100.0 |
| Min | 0.0 |
| Max | 100.0 |

### Operational

| Metric | Value |
|--------|------:|
| Parse/validation failures | 1 |
| Total grounding violations | 0 |
| Total timetable conflicts | 0 |
| Total existing-event conflicts | 0 |
| Total generated-session conflicts | 0 |
| Total deadline violations | 0 |
| Average runtime | 17368 ms (~17.4 s/case) |
| Total runtime | ~174 s |

### Perfect scores

7 of 9 parsed cases scored **100.0** (case_01, case_03, case_04, case_05, case_07, case_08, case_10).

---

## 3. Failure analysis (cases below 100)

### case_02 - total 0.0 (parse failure)

**Deterministic rule:** Pydantic schema validation (`AcademicPlan.priority` must be `low|medium|high`).

**Behavior:** LLM returned `medium-high` for two sessions and one task priority entry. JSON parsed successfully; schema validation rejected the plan before grounding or evaluation.

**No plan saved.** Baseline contract: no retry, score recorded as 0.

---

### case_06 - total 93.0

**Ground truth intent:** Missing `due_at`; must not schedule; must request clarification.

**Generated behavior (correct):**

- `proposed_sessions`: []  
- `clarification_required`: `[{ task_id: assignment_1, field: due_at, reason: ... }]`  
- Explanation correctly states missing due date prevents scheduling  

**Deterministic verification:** 0 issues, 0 grounding findings.

**Score deductions (scorer rules, not constraint violations):**

| Category | Lost | Rule |
|----------|-----:|------|
| conflicts | 5.0 | `_timetable_score`: empty `proposed_sessions` returns 15.0 instead of 20.0 |
| prioritization | 2.0 | `_prioritization_score`: empty `task_priorities` returns 8.0 instead of 10.0 |

The baseline handled missing information correctly but left prioritization empty and produced no sessions (as required).

---

### case_09 - total 95.0

**Ground truth intent:** Over-constrained 3-day horizon; must schedule quiz; avoid overlaps; acknowledge insufficient capacity.

**Generated behavior:**

- 7 sessions, quiz_os covered before deadline  
- Avoided `class_os_mon` (14:00-16:00) and `existing_event_1` (17:00-21:00)  
- Explanation states workload **exceeds available study capacity** and assignments **cannot be fully completed**  
- 0 verification issues, 0 grounding findings  

**Score deduction:**

| Category | Lost | Rule |
|----------|-----:|------|
| missing-information | 5.0 | `expect_insufficient_capacity`: keyword scan for `cannot fit`, `not enough time`, `insufficient`, `overload`, `trade-off`, `tradeoff` |

Baseline explanation used **exceeds the available study capacity** but did not match the frozen keyword list exactly (no substring hit).

---

## 4. Fixture contract confirmation

Files under `evaluation/fixtures/baseline/` contain **only** cached `AcademicPlan` JSON (e.g. `case_01.json`).

They do **not** contain scores or evaluation bypass logic. When run with `--fixtures`, plans load from fixture files and pass through the **same** pipeline:

`analyze_grounding` -> `verify_plan` -> `score_plan`

Verified: fixture `case_01` scores 100.0 via deterministic evaluator (test suite + manual run).

---

## 5. Frozen experiment contract

For baseline-vs-agent comparison, the following remain frozen:

- 10 evaluation case JSON files  
- Baseline prompt text  
- Scoring weights and rules  
- Shared `AcademicInput` / `AcademicPlan` schemas  
- Deterministic evaluator  

---

## 6. Observations

1. **Constraint satisfaction is strong** when output parses: zero timetable, event, session, or deadline violations across 9 parsed plans.  
2. **Primary baseline weakness is output reliability:** 1/10 schema validation failure (invalid enum).  
3. **Secondary gaps are scorer-shaped**, not planning-constraint failures (case_06 empty-list penalties, case_09 keyword sensitivity).  
4. **No evidence** that baseline failed due to poor context organization on multi-source inputs (cases 3-5, 8-10 succeeded with complex inputs).

---

## 7. Minimum agent architecture (justified by failures)

Based on measured failures, not a pre-decided multi-agent design:

### Include

| Component | Justification |
|-----------|---------------|
| **Deterministic verification (pre-submit)** | Zero constraint violations in successful parses suggests verification is most valuable as a **gate before evaluation** and to feed retries. Catches schema-adjacent issues when extended with format checks. |
| **Revision loop (max 1-2 retries)** | **Justified primarily for case_02-type failures:** feed Pydantic validation errors back to the planner (e.g. invalid enum values). One retry would likely recover `medium-high` -> `medium`. Not justified for constraint repair yet (0 constraint violations observed). |
| **Single planning stage** | Baseline planning quality is already high on parsed outputs; keep one planner LLM stage with same model and prompt quality bar. |

### Do not include yet

| Component | Reason |
|-----------|--------|
| **LLM Context Builder** | No failures indicating scattered-input misunderstanding. Parsed cases handled timetable, existing events, multi-deadline, and missing-info correctly. |
| **Multi-agent orchestration** | No measured benefit; adds complexity without addressing observed failure modes. |
| **Extra revision loops for constraints** | No timetable/deadline/workload violations to correct in successful parses. |

### Recommended minimal agent pipeline

```text
AcademicInput
    -> Planning Agent (1 LLM call, same model/prompt bar as baseline)
    -> Pydantic validate AcademicPlan
    -> Deterministic grounding + constraint verification
    -> IF schema/verification fail AND retries remain:
           feed structured findings to Planning Agent (1 retry max)
       ELSE:
           return plan for shared evaluator
```

**Expected agent gains:** recover case_02 (+11.2 mean points if counted as 0 today); optional small gains on case_06/09 via explicit prioritization list or capacity keywords if planner is instructed via verification feedback (without changing frozen baseline prompt).

**Not expected from Context Builder alone:** baseline already scored 100 on complex scheduling cases when parse succeeded.
