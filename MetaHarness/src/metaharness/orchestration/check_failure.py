"""Classify deterministic gate failures using the global recovery policy."""

from __future__ import annotations

from typing import Any

from ..evidence import EvidenceBundle
from ..recovery_policy import FailureClass, classify_failure


def hard_failure_items(failures: Any) -> list[str]:
    return [
        item for item in failures
        if isinstance(item, str)
        and classify_failure(item.split(":", 1)[0]).failure_class is FailureClass.FATAL
    ]


def hard_integrity_failures(bundle: EvidenceBundle) -> list[str]:
    return hard_failure_items(bundle.failures)
