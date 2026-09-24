"""Recovery of the final reviewer on one exact, immutable candidate.

The reviewer is never allowed to change the candidate, so every recovery
here reuses the same candidate SHA, tree and deterministic evidence.  Only
transport failures are retried; an unparsable review is a format failure the
review correction loop handles, and credentials are a waiting condition.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..evidence import EvidenceBundle
from ..llm.chat import LLMError, LLMHTTPError, LLMProtocolError
from ..models import ReviewVerdict
from ..recovery_policy import RecoveryBudgets
from ..review import ReviewParseError, ReviewResult, structured_review_reason
from .pipeline_v2 import PipelineFailure
from .recovery import RecoveryCoordinator
from .shared import _bounded_v2_report

_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})


def bounded_detail(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500]


def classify_reviewer_error(exc: LLMError) -> str:
    """Map a reviewer client error onto a stable failure code."""

    status_match = re.search(r"\bHTTP\s+(\d{3})\b", str(exc))
    status_code = int(status_match.group(1)) if status_match else None
    if status_code in {401, 403}:
        return f"LLM_{status_code}"
    if isinstance(exc, LLMProtocolError):
        return "REVIEW_FORMAT_INVALID"
    if (
        not isinstance(exc, LLMHTTPError)
        or status_code in _RETRYABLE_HTTP
        or any(marker in str(exc).casefold() for marker in (
            "timed out", "before receiving an http response",
        ))
    ):
        return "REVIEWER_TRANSPORT_FAILURE"
    return "REVIEW_FORMAT_INVALID"


class ReviewRecovery:
    """Transport and evidence recovery of one run's final reviewer."""

    def __init__(self, recovery: RecoveryCoordinator, *, budgets: RecoveryBudgets) -> None:
        self._recovery = recovery
        self._budgets = budgets

    def with_transport_retries(
        self,
        *,
        cycle: int,
        candidate_tree: str | None,
        run_review: Callable[[], ReviewResult],
        archive_attempt: Callable[[], Any],
    ) -> ReviewResult:
        """Retry transient reviewer transport on this candidate only."""

        key = self._recovery.budget_key("review-transport", f"{cycle:03d}")
        while True:
            try:
                return run_review()
            except ReviewParseError as exc:
                raise PipelineFailure("REVIEW_FORMAT_INVALID", bounded_detail(exc)) from exc
            except LLMError as exc:
                code = classify_reviewer_error(exc)
                if code.startswith("LLM_"):
                    raise PipelineFailure(code, "reviewer authorization is required") from exc
                if code != "REVIEWER_TRANSPORT_FAILURE":
                    raise PipelineFailure(code, bounded_detail(exc)) from exc
                admission = self._recovery.admit(
                    key, reason=code, budget=self._budgets.max_review_transport_retries,
                    phase="final_review", cycle=cycle,
                    tree_before=candidate_tree, tree_after=candidate_tree,
                )
                if not admission.admitted:
                    raise PipelineFailure(code, bounded_detail(exc)) from exc
                archive_attempt()

    def after_reviewer_fail(
        self,
        *,
        cycle: int,
        first_review: ReviewResult,
        rebuild_evidence: Callable[[], EvidenceBundle],
        rerun: Callable[[EvidenceBundle, bool], ReviewResult],
        archive_attempt: Callable[[], Any],
    ) -> ReviewResult:
        """Rebuild local authority, then allow one reviewer evidence retry."""

        reason_class = structured_review_reason(first_review.findings)
        durable_evidence = rebuild_evidence()
        key = self._recovery.budget_key("review-fail-recovery", f"{cycle:03d}")
        admission = self._recovery.admit(
            key, reason="REVIEW_EVIDENCE_RETRY", budget=1, phase="final_review",
            cycle=cycle, tree_before=durable_evidence.staged_tree_sha,
            tree_after=durable_evidence.staged_tree_sha,
        )
        if not admission.admitted:
            raise PipelineFailure(
                "REVIEW_EVIDENCE_UNRESOLVED",
                {
                    "reason_class": reason_class,
                    "findings": _bounded_v2_report(first_review.findings),
                },
            )
        archive_attempt()
        review = rerun(durable_evidence, reason_class == "EVIDENCE_UNAVAILABLE")
        if review.verdict is ReviewVerdict.FAIL:
            raise PipelineFailure(
                "REVIEW_EVIDENCE_UNRESOLVED",
                {
                    "first_reason_class": reason_class,
                    "retry_reason_class": structured_review_reason(review.findings),
                    "findings": _bounded_v2_report(review.findings),
                },
            )
        return review


__all__ = ["ReviewRecovery", "bounded_detail", "classify_reviewer_error"]
