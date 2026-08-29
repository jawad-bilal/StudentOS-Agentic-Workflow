"""Deterministic plan verification (shared by baseline and agent)."""

from __future__ import annotations

from core.models import (
    AcademicInput,
    AcademicPlan,
    GroundingAnalysis,
    GroundingFindingType,
    SourceRef,
    VerificationIssue,
    VerificationIssueType,
    VerificationResult,
    VerificationStatus,
    session_duration_minutes,
)
from evaluation.constraints import (
    intervals_overlap,
    session_interval,
    study_minutes_by_date,
    timetable_occurrences,
)


def verify_plan(
    plan: AcademicPlan,
    academic_input: AcademicInput,
    grounding: GroundingAnalysis | None = None,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    warnings: list[str] = []
    checks_run: list[str] = [
        "grounding_findings",
        "timetable_overlap",
        "existing_event_overlap",
        "session_overlap",
        "deadline_violation",
        "workload_exceeded",
        "invented_deadline_date",
    ]

    grounding = grounding or GroundingAnalysis()
    deadline_map = {d.id: d for d in academic_input.deadlines}
    tz = academic_input.timezone_info()
    max_hours = academic_input.preferences.max_study_hours_per_day

    for finding in grounding.findings:
        if finding.type == GroundingFindingType.UNKNOWN_TASK_ID:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.UNKNOWN_TASK_ID,
                    message=finding.message,
                    session_id=finding.session_id,
                    task_id=finding.task_id,
                )
            )
        elif finding.type == GroundingFindingType.UNKNOWN_PRIORITY_TASK_ID:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.UNKNOWN_TASK_ID,
                    message=finding.message,
                    task_id=finding.task_id,
                )
            )
        elif finding.type == GroundingFindingType.SCHEDULED_TASK_WITH_MISSING_DUE:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.INVENTED_DEADLINE_DATE,
                    message=finding.message,
                    session_id=finding.session_id,
                    task_id=finding.task_id,
                )
            )
        elif finding.type == GroundingFindingType.SESSION_TEMPORAL_INVALID:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.TEMPORAL_INVALID,
                    message=finding.message,
                    session_id=finding.session_id,
                    task_id=finding.task_id,
                )
            )
        elif finding.type == GroundingFindingType.ESTIMATED_MINUTES_MISMATCH:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.ESTIMATED_MINUTES_MISMATCH,
                    message=finding.message,
                    session_id=finding.session_id,
                    task_id=finding.task_id,
                )
            )

    for session in plan.proposed_sessions:
        s_start, s_end = session_interval(session, tz)

        for block in academic_input.timetable:
            for _, t_start, t_end in timetable_occurrences(block, academic_input):
                if session.date == t_start.date().isoformat() and intervals_overlap(
                    s_start, s_end, t_start, t_end
                ):
                    issues.append(
                        VerificationIssue(
                            type=VerificationIssueType.TIMETABLE_OVERLAP,
                            message=(
                                f"Session {session.session_id} overlaps timetable block {block.id}"
                            ),
                            session_id=session.session_id,
                            task_id=session.task_id,
                            conflicting_ref=SourceRef(kind="timetable", id=block.id),
                            date=session.date,
                        )
                    )

        for event in academic_input.existing_events:
            if not event.blocks_study:
                continue
            if intervals_overlap(s_start, s_end, event.start, event.end):
                issues.append(
                    VerificationIssue(
                        type=VerificationIssueType.EXISTING_EVENT_OVERLAP,
                        message=(
                            f"Session {session.session_id} overlaps existing event {event.id}"
                        ),
                        session_id=session.session_id,
                        task_id=session.task_id,
                        conflicting_ref=SourceRef(kind="existing_event", id=event.id),
                        date=session.date,
                    )
                )

    sessions = plan.proposed_sessions
    for i, left in enumerate(sessions):
        l_start, l_end = session_interval(left, tz)
        for right in sessions[i + 1 :]:
            if left.date != right.date:
                continue
            r_start, r_end = session_interval(right, tz)
            if intervals_overlap(l_start, l_end, r_start, r_end):
                issues.append(
                    VerificationIssue(
                        type=VerificationIssueType.SESSION_OVERLAP,
                        message=(
                            f"Sessions {left.session_id} and {right.session_id} overlap"
                        ),
                        session_id=left.session_id,
                        date=left.date,
                    )
                )

    for session in plan.proposed_sessions:
        if not session.task_id:
            continue
        deadline = deadline_map.get(session.task_id)
        if not deadline or deadline.due_at is None:
            continue
        _, s_end = session_interval(session, tz)
        if s_end > deadline.due_at:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.DEADLINE_VIOLATION,
                    message=(
                        f"Session {session.session_id} ends after deadline {session.task_id} due_at"
                    ),
                    session_id=session.session_id,
                    task_id=session.task_id,
                    date=session.date,
                )
            )

    for deadline_id, deadline in deadline_map.items():
        if deadline.due_at is None:
            continue
        task_sessions = [s for s in plan.proposed_sessions if s.task_id == deadline_id]
        if not task_sessions:
            continue
        latest_end = max(session_interval(s, tz)[1] for s in task_sessions)
        if latest_end > deadline.due_at:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.DEADLINE_VIOLATION,
                    message=f"Work for {deadline_id} is not fully scheduled before due_at",
                    task_id=deadline_id,
                )
            )

    minutes_by_date = study_minutes_by_date(plan.proposed_sessions)
    for day, minutes in minutes_by_date.items():
        hours = minutes / 60.0
        if hours > max_hours + 1e-9:
            issues.append(
                VerificationIssue(
                    type=VerificationIssueType.WORKLOAD_EXCEEDED,
                    message=f"Daily workload on {day} is {hours:.2f}h (max {max_hours}h)",
                    date=day,
                )
            )

    clarification_ids = {c.task_id for c in plan.clarification_required}
    for deadline in academic_input.deadlines:
        if deadline.due_at is None and deadline.id not in clarification_ids:
            scheduled = any(s.task_id == deadline.id for s in plan.proposed_sessions)
            if scheduled:
                issues.append(
                    VerificationIssue(
                        type=VerificationIssueType.INVENTED_DEADLINE_DATE,
                        message=(
                            f"Task {deadline.id} has no due_at but was scheduled without "
                            "clarification_required"
                        ),
                        task_id=deadline.id,
                    )
                )

    status = VerificationStatus.PASS if not issues else VerificationStatus.FAIL
    required_changes = [issue.message for issue in issues[:5]]

    return VerificationResult(
        status=status,
        issues=issues,
        warnings=warnings,
        required_changes=required_changes,
        checks_run=checks_run,
    )
