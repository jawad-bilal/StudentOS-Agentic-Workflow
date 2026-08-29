"""Serialize AcademicInput for LLM prompts."""

from __future__ import annotations

import json
from typing import Any

from core.models import AcademicInput


def build_prompt_payload(academic_input: AcademicInput) -> dict[str, Any]:
    horizon_end = academic_input.horizon_end()
    payload: dict[str, Any] = {
        "evaluation_context": {
            "reference_now": academic_input.reference_now.isoformat(),
            "planning_horizon_start": academic_input.reference_now.date().isoformat(),
            "planning_horizon_end": horizon_end.date().isoformat(),
            "planning_horizon_days": academic_input.planning_horizon_days,
            "timezone": academic_input.preferences.timezone,
            "instruction": (
                "Treat reference_now as the current date and time. "
                "Do not use any other notion of today."
            ),
        },
        "student_id": academic_input.student_id,
        "courses": [c.model_dump(mode="json") for c in academic_input.courses],
        "deadlines": [d.model_dump(mode="json") for d in academic_input.deadlines],
        "timetable": [t.model_dump(mode="json") for t in academic_input.timetable],
        "existing_events": [
            e.model_dump(mode="json") for e in academic_input.existing_events
        ],
        "preferences": academic_input.preferences.model_dump(mode="json"),
    }
    if academic_input.notes_summary is not None:
        payload["notes_summary"] = academic_input.notes_summary
    return payload


def serialize_academic_input(academic_input: AcademicInput) -> str:
    payload = build_prompt_payload(academic_input)
    return json.dumps(payload, indent=2, sort_keys=True)


BASELINE_SYSTEM_PROMPT = """You are an academic planning assistant for university students.

You receive a complete snapshot of a student's academic situation at a fixed reference time.
Your job is to produce one realistic study plan as structured JSON.

Hard rules:
1. Treat evaluation_context.reference_now as the current date and time. Never use any other "now".
2. Schedule study sessions only within the planning horizon.
3. Do not invent courses, deadlines, classes, or calendar events that are not in the input.
4. If a deadline has due_at=null, do not assign it a due date or schedule concrete sessions for it.
   Instead add a clarification_required item with task_id, field, and reason.
5. Avoid overlapping study sessions with:
   - recurring timetable blocks on the matching weekday
   - existing_events where blocks_study is true
   - other proposed study sessions
6. Respect preferences.max_study_hours_per_day for each calendar date.
7. Prefer scheduling required work before each deadline's due_at.
8. Use only deadline ids from the input as task_id values.
9. When estimated_minutes is present on a session, it must equal the actual session duration in minutes.
10. Sessions cannot cross midnight in V1: end must be after start on the same date.
11. Return JSON only. No markdown fences. No prose outside the JSON object.
12. Use clarification_required only when information is actually missing or needs user input.
    If total required work cannot fit within constraints, explain that in explanation and/or assumptions.
    Do not treat capacity infeasibility as missing information."""


def build_baseline_user_prompt(serialized_input: str) -> str:
    return f"""Create a study plan for the student described below.

Return a JSON object matching this schema exactly:

{{
  "student_id": "<same as input>",
  "reference_now": "<same ISO datetime as evaluation_context.reference_now>",
  "proposed_sessions": [
    {{
      "session_id": "<unique string, e.g. sess_001>",
      "task_id": "<deadline id or null>",
      "title": "<short session title>",
      "date": "YYYY-MM-DD",
      "start": "HH:MM",
      "end": "HH:MM",
      "priority": "low|medium|high",
      "reason": "<why this session helps>",
      "estimated_minutes": <integer equal to session duration, or null>
    }}
  ],
  "task_priorities": [
    {{
      "task_id": "<deadline id>",
      "rank": <1 = most urgent>,
      "priority": "low|medium|high",
      "reason": "<short reason>"
    }}
  ],
  "total_estimated_minutes": <sum of session minutes, integer>,
  "explanation": "<2-5 sentences summarizing the plan for the student>",
  "assumptions": ["<optional assumption strings>"],
  "clarification_required": [
    {{
      "task_id": "<deadline id>",
      "field": "<field name, e.g. due_at>",
      "reason": "<why clarification is needed>"
    }}
  ]
}}

Planning requirements:
- Cover urgent and upcoming deadlines before they are due.
- Spread workload across days when possible; avoid cramming beyond max_study_hours_per_day.
- Use preferred_study_start and preferred_study_end as soft bounds unless timetable or existing events require otherwise.
- Session length should be between min_session_minutes and max_session_minutes when feasible.
- For exams and large assignments, allow multiple sessions before the due date when estimated_hours suggests it.
- If total required work cannot fit within constraints, say so in explanation and/or assumptions.
  Do not add clarification_required entries for capacity infeasibility alone.

Student data:
{serialized_input}"""


def build_attempt_summary_json(review: AttemptReview) -> str:
    import json

    payload = {
        "attempt": review.attempt,
        "parse_error": review.parse_error,
        "plan_produced": review.plan is not None,
        "session_count": len(review.plan.proposed_sessions) if review.plan else 0,
        "note": (
            "Previous response could not be validated; produce a fresh valid AcademicPlan."
            if not review.plan
            else "Previous plan failed validation or verification; generate a new plan that fixes all findings."
        ),
    }
    return json.dumps(payload, indent=2)


def build_agent_retry_user_prompt(
    serialized_input: str,
    revision_findings: list[dict],
    attempt_summary_json: str,
) -> str:
    import json

    findings_json = json.dumps(revision_findings, indent=2)
    return f"""Your previous study plan attempt failed validation or verification.

Revise the plan by producing a completely new JSON AcademicPlan from scratch.
Do not patch the previous JSON. Do not return a diff.
Fix every issue listed in revision_findings.

Return a JSON object matching this schema exactly:

{{
  "student_id": "<same as input>",
  "reference_now": "<same ISO datetime as evaluation_context.reference_now>",
  "proposed_sessions": [
    {{
      "session_id": "<unique string, e.g. sess_001>",
      "task_id": "<deadline id or null>",
      "title": "<short session title>",
      "date": "YYYY-MM-DD",
      "start": "HH:MM",
      "end": "HH:MM",
      "priority": "low|medium|high",
      "reason": "<why this session helps>",
      "estimated_minutes": <integer equal to session duration, or null>
    }}
  ],
  "task_priorities": [
    {{
      "task_id": "<deadline id>",
      "rank": <1 = most urgent>,
      "priority": "low|medium|high",
      "reason": "<short reason>"
    }}
  ],
  "total_estimated_minutes": <sum of session minutes, integer>,
  "explanation": "<2-5 sentences summarizing the plan for the student>",
  "assumptions": ["<optional assumption strings>"],
  "clarification_required": [
    {{
      "task_id": "<deadline id>",
      "field": "<field name, e.g. due_at>",
      "reason": "<why clarification is needed>"
    }}
  ]
}}

Revision rules:
- Fix every item in revision_findings.
- priority must be exactly one of: low, medium, high.
- Use only deadline ids from the input as task_id values.
- Do not invent courses, deadlines, classes, or calendar events.
- Avoid timetable overlaps, existing-event overlaps, and session overlaps.
- Respect max_study_hours_per_day.
- When estimated_minutes is present, it must equal the session duration in minutes.
- Sessions cannot cross midnight in V1.
- Use clarification_required only for genuinely missing information, not for capacity infeasibility.
- If total required work cannot fit within constraints, explain that in explanation and/or assumptions using clear wording such as: insufficient capacity, cannot fit, not enough time, overload, or trade-off.

Previous attempt summary:
{attempt_summary_json}

revision_findings:
{findings_json}

Student data:
{serialized_input}"""
