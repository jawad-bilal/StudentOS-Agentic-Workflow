"""Select the best valid AcademicPlan after first and optional retry attempts."""

from __future__ import annotations

from typing import Literal

from core.findings import AttemptReview
from core.models import AcademicPlan


def _attempt_rank(review: AttemptReview) -> tuple[int, int, int, int]:
    """
    Lower is better (validity-first hierarchy).

    Tier 0: valid AcademicPlan exists
    Tier 1: verification PASS + no grounding findings (clean)
    Tier 2: count of grounding + verification issues
    """
    if not review.is_valid_plan:
        return (1, 1, 9999, review.llm_call_number)

    if review.is_clean:
        return (0, 0, 0, review.llm_call_number)

    issue_count = review.grounding_verification_finding_count
    verification_fail = 0 if review.verification and review.verification.status.value == "pass" else 1
    return (0, verification_fail, issue_count, review.llm_call_number)


def select_best_attempt(
    first: AttemptReview,
    retry: AttemptReview | None,
) -> tuple[AcademicPlan | None, Literal["first", "retry"] | None, str | None]:
    """
    Validity-first failure policy:

    1. Prefer attempts with a valid AcademicPlan.
    2. Among valid attempts:
       a. Prefer verification PASS + no grounding findings.
       b. Otherwise prefer fewer grounding/verification findings.
    3. Tie -> prefer retry.
    4. If neither validates -> None (evaluator records 0).
    5. Never repair either plan.
    """
    candidates: list[tuple[Literal["first", "retry"], AttemptReview]] = [("first", first)]
    if retry is not None:
        candidates.append(("retry", retry))

    valid = [(label, review) for label, review in candidates if review.is_valid_plan]
    if not valid:
        errors = []
        if first.parse_error:
            errors.append(f"first: {first.parse_error}")
        if first.validation_findings:
            errors.append(f"first: {len(first.validation_findings)} validation finding(s)")
        if retry:
            if retry.parse_error:
                errors.append(f"retry: {retry.parse_error}")
            if retry.validation_findings:
                errors.append(f"retry: {len(retry.validation_findings)} validation finding(s)")
        if not errors:
            errors.append("No valid AcademicPlan produced")
        return None, None, "; ".join(errors)

    best_label, best_review = min(
        valid,
        key=lambda item: (_attempt_rank(item[1]), 0 if item[0] == "retry" else 1),
    )
    assert best_review.plan is not None
    return best_review.plan, best_label, None
