"""Structured findings for agent review and retry (no scorer data)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from core.models import (
    AcademicPlan,
    GroundingAnalysis,
    GroundingFinding,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)


class FindingSource(StrEnum):
    VALIDATION = "validation"
    GROUNDING = "grounding"
    VERIFICATION = "verification"


class PlanReviewFinding(BaseModel):
    source: FindingSource
    type: str
    message: str
    required_change: str
    field: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    conflicting_ref_id: str | None = None


class AttemptReview(BaseModel):
    attempt: Literal["first", "retry"]
    llm_call_number: int
    raw_text: str | None = None
    plan: AcademicPlan | None = None
    parse_error: str | None = None
    validation_findings: list[PlanReviewFinding] = Field(default_factory=list)
    grounding: GroundingAnalysis = Field(default_factory=GroundingAnalysis)
    verification: VerificationResult | None = None

    @property
    def is_valid_plan(self) -> bool:
        return self.plan is not None and not self.parse_error and not self.validation_findings

    @property
    def is_clean(self) -> bool:
        return (
            self.is_valid_plan
            and not self.grounding.has_findings
            and self.verification is not None
            and self.verification.status == VerificationStatus.PASS
        )

    @property
    def has_actionable_findings(self) -> bool:
        if self.parse_error:
            return True
        if self.validation_findings:
            return True
        if self.grounding.has_findings:
            return True
        if self.verification and self.verification.status != VerificationStatus.PASS:
            return True
        return False

    @property
    def grounding_verification_finding_count(self) -> int:
        count = len(self.grounding.findings)
        if self.verification:
            count += len(self.verification.issues)
        return count

    def actionable_findings(self) -> list[PlanReviewFinding]:
        findings: list[PlanReviewFinding] = list(self.validation_findings)
        for gf in self.grounding.findings:
            findings.append(
                PlanReviewFinding(
                    source=FindingSource.GROUNDING,
                    type=gf.type.value,
                    message=gf.message,
                    required_change=_grounding_required_change(gf),
                    session_id=gf.session_id,
                    task_id=gf.task_id,
                )
            )
        if self.verification:
            for issue in self.verification.issues:
                ref_id = issue.conflicting_ref.id if issue.conflicting_ref else None
                findings.append(
                    PlanReviewFinding(
                        source=FindingSource.VERIFICATION,
                        type=issue.type.value,
                        message=issue.message,
                        required_change=_verification_required_change(issue),
                        session_id=issue.session_id,
                        task_id=issue.task_id,
                        conflicting_ref_id=ref_id,
                    )
                )
        if self.parse_error:
            findings.insert(
                0,
                PlanReviewFinding(
                    source=FindingSource.VALIDATION,
                    type="json_parse_error",
                    message=self.parse_error,
                    required_change="Return valid JSON matching the AcademicPlan schema.",
                ),
            )
        return findings


class AgentRunMetadata(BaseModel):
    llm_calls: int
    retry_occurred: bool
    selected_attempt: Literal["first", "retry"] | None = None
    initial_validation_findings: list[PlanReviewFinding] = Field(default_factory=list)
    initial_grounding_findings: list[GroundingFinding] = Field(default_factory=list)
    initial_verification_findings: list[VerificationIssue] = Field(default_factory=list)
    final_validation_ok: bool = False
    final_grounding_ok: bool = False
    final_verification_ok: bool = False
    final_plan_hash: str = ""
    total_runtime_ms: int = 0
    llm_provider: str = ""
    llm_model: str = ""


def _grounding_required_change(finding: GroundingFinding) -> str:
    if finding.type.value == "unknown_task_id":
        return f"Remove or replace unknown task_id {finding.task_id!r} with a valid deadline id from input."
    if finding.type.value == "unknown_priority_task_id":
        return f"Remove or replace unknown priority task_id {finding.task_id!r}."
    if finding.type.value == "scheduled_task_with_missing_due":
        return f"Do not schedule {finding.task_id!r}; add clarification_required instead."
    if finding.type.value == "estimated_minutes_mismatch":
        return f"Set estimated_minutes on session {finding.session_id} to match start/end duration."
    if finding.type.value == "session_temporal_invalid":
        return f"Fix temporal fields on session {finding.session_id}: end must be after start on same date."
    if finding.type.value == "student_id_mismatch":
        return "Set student_id to match the input student_id exactly."
    if finding.type.value == "reference_now_mismatch":
        return "Set reference_now to match evaluation_context.reference_now exactly."
    return finding.message


def _verification_required_change(issue: VerificationIssue) -> str:
    t = issue.type.value
    if t == "timetable_overlap":
        ref = issue.conflicting_ref.id if issue.conflicting_ref else "timetable block"
        return f"Reschedule session {issue.session_id} to avoid overlapping {ref}."
    if t == "existing_event_overlap":
        ref = issue.conflicting_ref.id if issue.conflicting_ref else "existing event"
        return f"Reschedule session {issue.session_id} to avoid overlapping {ref}."
    if t == "session_overlap":
        return f"Reschedule session {issue.session_id} so it does not overlap another session."
    if t == "deadline_violation":
        return f"Schedule work for {issue.task_id} to finish before its due_at."
    if t == "workload_exceeded":
        return "Reduce daily scheduled minutes to respect max_study_hours_per_day."
    if t == "invented_deadline_date":
        return f"Do not schedule {issue.task_id}; use clarification_required if due_at is missing."
    if t == "unknown_task_id":
        return f"Remove references to unknown task_id {issue.task_id!r}."
    if t == "temporal_invalid":
        return f"Fix temporal relationship on session {issue.session_id}."
    if t == "estimated_minutes_mismatch":
        return f"Align estimated_minutes with session duration on {issue.session_id}."
    return issue.message


def validation_error_to_findings(exc: Exception) -> list[PlanReviewFinding]:
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return [
            PlanReviewFinding(
                source=FindingSource.VALIDATION,
                type="schema_validation_error",
                message=str(exc),
                required_change="Fix schema validation errors and return a valid AcademicPlan.",
            )
        ]

    findings: list[PlanReviewFinding] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        err_type = err.get("type", "")
        finding_type = "invalid_enum" if err_type == "enum" else "schema_validation_error"
        required = (
            f"Set {loc} to exactly one of: low, medium, high."
            if finding_type == "invalid_enum" and "priority" in loc
            else f"Fix field {loc}: {msg}"
        )
        findings.append(
            PlanReviewFinding(
                source=FindingSource.VALIDATION,
                type=finding_type,
                field=loc or None,
                message=msg,
                required_change=required,
            )
        )
    return findings
