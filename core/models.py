"""Normalized Pydantic models for the agentic academic workflow."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator


class Weekday(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class DeadlineType(StrEnum):
    ASSIGNMENT = "assignment"
    QUIZ = "quiz"
    LAB = "lab"
    EXAM = "exam"
    OTHER = "other"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW_REQUIRED = "review_required"


class VerificationIssueType(StrEnum):
    HALLUCINATED_TASK = "hallucinated_task"
    HALLUCINATED_DEADLINE = "hallucinated_deadline"
    UNKNOWN_TASK_ID = "unknown_task_id"
    TIMETABLE_OVERLAP = "timetable_overlap"
    EXISTING_EVENT_OVERLAP = "existing_event_overlap"
    SESSION_OVERLAP = "session_overlap"
    DEADLINE_VIOLATION = "deadline_violation"
    WORKLOAD_EXCEEDED = "workload_exceeded"
    INVENTED_DEADLINE_DATE = "invented_deadline_date"
    INSUFFICIENT_PREP_TIME = "insufficient_prep_time"
    TEMPORAL_INVALID = "temporal_invalid"
    ESTIMATED_MINUTES_MISMATCH = "estimated_minutes_mismatch"


class GroundingFindingType(StrEnum):
    UNKNOWN_TASK_ID = "unknown_task_id"
    UNKNOWN_PRIORITY_TASK_ID = "unknown_priority_task_id"
    STUDENT_ID_MISMATCH = "student_id_mismatch"
    REFERENCE_NOW_MISMATCH = "reference_now_mismatch"
    SESSION_TEMPORAL_INVALID = "session_temporal_invalid"
    ESTIMATED_MINUTES_MISMATCH = "estimated_minutes_mismatch"
    SCHEDULED_TASK_WITH_MISSING_DUE = "scheduled_task_with_missing_due"


TimeStr = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
DateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


def parse_time_str(value: TimeStr) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def session_duration_minutes(start: TimeStr, end: TimeStr) -> int:
    start_t = parse_time_str(start)
    end_t = parse_time_str(end)
    start_min = start_t.hour * 60 + start_t.minute
    end_min = end_t.hour * 60 + end_t.minute
    return end_min - start_min


class Course(BaseModel):
    id: str = Field(min_length=1)
    code: str | None = None
    name: str = Field(min_length=1)


class Deadline(BaseModel):
    id: str = Field(min_length=1)
    course_id: str
    title: str
    type: DeadlineType = DeadlineType.ASSIGNMENT
    due_at: datetime | None = None
    estimated_hours: float | None = Field(default=None, ge=0.25, le=40)
    description: str = ""


class TimetableBlock(BaseModel):
    id: str = Field(min_length=1)
    course_id: str | None = None
    weekday: Weekday
    start: TimeStr
    end: TimeStr
    title: str
    venue: str = ""

    @model_validator(mode="after")
    def end_after_start(self) -> TimetableBlock:
        if self.end <= self.start:
            raise ValueError("Timetable block end must be after start")
        return self


class ExistingEvent(BaseModel):
    id: str = Field(min_length=1)
    title: str
    start: datetime
    end: datetime
    blocks_study: bool = True

    @model_validator(mode="after")
    def end_after_start(self) -> ExistingEvent:
        if self.end <= self.start:
            raise ValueError("Existing event end must be after start")
        return self


class StudyPreferences(BaseModel):
    timezone: str = "Asia/Karachi"
    max_study_hours_per_day: float = Field(default=4.0, ge=0.5, le=16)
    preferred_study_start: TimeStr = "10:00"
    preferred_study_end: TimeStr = "22:00"
    min_session_minutes: int = Field(default=30, ge=15, le=240)
    max_session_minutes: int = Field(default=120, ge=30, le=480)


class AcademicInput(BaseModel):
    student_id: str
    reference_now: datetime
    planning_horizon_days: int = Field(default=14, ge=1, le=60)
    courses: list[Course] = Field(min_length=1)
    deadlines: list[Deadline] = Field(default_factory=list)
    timetable: list[TimetableBlock] = Field(default_factory=list)
    existing_events: list[ExistingEvent] = Field(default_factory=list)
    preferences: StudyPreferences = Field(default_factory=StudyPreferences)
    notes_summary: str | None = None

    @model_validator(mode="after")
    def validate_references(self) -> AcademicInput:
        course_ids = {c.id for c in self.courses}
        for deadline in self.deadlines:
            if deadline.course_id not in course_ids:
                raise ValueError(
                    f"Deadline {deadline.id} references unknown course {deadline.course_id}"
                )
        for block in self.timetable:
            if block.course_id and block.course_id not in course_ids:
                raise ValueError(
                    f"Timetable block {block.id} references unknown course {block.course_id}"
                )
        return self

    def timezone_info(self) -> ZoneInfo:
        return ZoneInfo(self.preferences.timezone)

    def horizon_end(self) -> datetime:
        return self.reference_now + timedelta(days=self.planning_horizon_days)

    def deadline_ids(self) -> set[str]:
        return {d.id for d in self.deadlines}


class ClarificationItem(BaseModel):
    task_id: str
    field: str
    reason: str


class StudySession(BaseModel):
    session_id: str = Field(min_length=1)
    task_id: str | None = None
    title: str
    date: DateStr
    start: TimeStr
    end: TimeStr
    priority: Priority
    reason: str = ""
    estimated_minutes: int | None = Field(default=None, ge=15)

    @model_validator(mode="after")
    def end_after_start_same_day(self) -> StudySession:
        if self.end <= self.start:
            raise ValueError("Session end must be after start on the same calendar date (V1: no cross-midnight sessions)")
        return self


class TaskPriority(BaseModel):
    task_id: str
    rank: int = Field(ge=1)
    priority: Priority
    reason: str = ""


class AcademicPlan(BaseModel):
    student_id: str
    reference_now: datetime
    proposed_sessions: list[StudySession] = Field(default_factory=list)
    task_priorities: list[TaskPriority] = Field(default_factory=list)
    total_estimated_minutes: int = Field(default=0, ge=0)
    explanation: str = ""
    assumptions: list[str] = Field(default_factory=list)
    clarification_required: list[ClarificationItem] = Field(default_factory=list)


class SourceRef(BaseModel):
    kind: Literal["deadline", "timetable", "existing_event", "course"]
    id: str


class GroundingFinding(BaseModel):
    type: GroundingFindingType
    message: str
    session_id: str | None = None
    task_id: str | None = None


class GroundingAnalysis(BaseModel):
    findings: list[GroundingFinding] = Field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0


class VerificationIssue(BaseModel):
    type: VerificationIssueType
    message: str
    session_id: str | None = None
    task_id: str | None = None
    conflicting_ref: SourceRef | None = None
    date: DateStr | None = None


class VerificationResult(BaseModel):
    status: VerificationStatus
    issues: list[VerificationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    deadlines_respected: float = Field(ge=0, le=25)
    no_hallucinations: float = Field(ge=0, le=20)
    no_timetable_conflicts: float = Field(ge=0, le=20)
    workload_distribution: float = Field(ge=0, le=15)
    prioritization: float = Field(ge=0, le=10)
    missing_information_handling: float = Field(ge=0, le=10)

    @property
    def total(self) -> float:
        return (
            self.deadlines_respected
            + self.no_hallucinations
            + self.no_timetable_conflicts
            + self.workload_distribution
            + self.prioritization
            + self.missing_information_handling
        )


class EvaluationResult(BaseModel):
    case_id: str
    system: Literal["baseline", "agent", "fixture", "manual"]
    plan: AcademicPlan | None = None
    grounding: GroundingAnalysis = Field(default_factory=GroundingAnalysis)
    verification: VerificationResult | None = None
    score: ScoreBreakdown | None = None
    total_score: float = Field(default=0.0, ge=0, le=100)
    runtime_ms: int = Field(default=0, ge=0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundTruthConstraints(BaseModel):
    must_schedule_before: list[str] = Field(default_factory=list)
    must_not_schedule: list[str] = Field(default_factory=list)
    must_not_overlap: list[str] = Field(default_factory=list)
    must_not_invent: bool = True
    must_flag_missing: list[str] = Field(default_factory=list)
    max_study_hours_per_day: float | None = None
    min_sessions_for: dict[str, int] = Field(default_factory=dict)
    min_prep_minutes_for: dict[str, int] = Field(default_factory=dict)
    expect_clarification_for: list[str] = Field(default_factory=list)
    expect_high_priority_for: list[str] = Field(default_factory=list)
    expect_spread_across_days: bool = False
    min_distinct_study_days: int | None = None
    expect_insufficient_capacity: bool = False


class EvaluationCaseMetadata(BaseModel):
    case_id: str = Field(pattern=r"^case_\d{2}$")
    title: str
    description: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    metadata: EvaluationCaseMetadata
    input: AcademicInput
    ground_truth: GroundTruthConstraints


def combine_local(session_date: DateStr, clock: TimeStr, tz: ZoneInfo) -> datetime:
    d = date.fromisoformat(session_date)
    t = parse_time_str(clock)
    return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=tz)


def validate_session_temporal(session: StudySession, preferences: StudyPreferences) -> list[GroundingFinding]:
    """Temporal rules beyond basic Pydantic schema validation."""
    findings: list[GroundingFinding] = []

    if session.end <= session.start:
        findings.append(
            GroundingFinding(
                type=GroundingFindingType.SESSION_TEMPORAL_INVALID,
                message="Session end must be after start",
                session_id=session.session_id,
                task_id=session.task_id,
            )
        )
        return findings

    duration = session_duration_minutes(session.start, session.end)
    if duration <= 0:
        findings.append(
            GroundingFinding(
                type=GroundingFindingType.SESSION_TEMPORAL_INVALID,
                message="Session duration must be positive (V1: sessions cannot cross midnight)",
                session_id=session.session_id,
                task_id=session.task_id,
            )
        )

    if session.estimated_minutes is not None and session.estimated_minutes != duration:
        findings.append(
            GroundingFinding(
                type=GroundingFindingType.ESTIMATED_MINUTES_MISMATCH,
                message=(
                    f"estimated_minutes ({session.estimated_minutes}) must equal "
                    f"session duration ({duration} minutes) when present"
                ),
                session_id=session.session_id,
                task_id=session.task_id,
            )
        )

    if duration < preferences.min_session_minutes or duration > preferences.max_session_minutes:
        findings.append(
            GroundingFinding(
                type=GroundingFindingType.SESSION_TEMPORAL_INVALID,
                message=(
                    f"Session duration {duration}m outside allowed range "
                    f"{preferences.min_session_minutes}-{preferences.max_session_minutes}m"
                ),
                session_id=session.session_id,
                task_id=session.task_id,
            )
        )

    return findings
