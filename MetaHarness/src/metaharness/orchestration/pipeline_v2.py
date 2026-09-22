"""Generic pipeline-v2 coordinator and durable artifact layout helpers.

The coordinator deliberately knows only the pipeline boundary and injected
operations.  It does not own an :class:`Orchestrator` instance, provider
names, or a fixed number of review cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..models import RunCycle
from ..result import RunResult
from ..state import RunStateStore


def cycle_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    number = cycle.number if isinstance(cycle, RunCycle) else cycle
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("cycle number must be a positive integer")
    return Path(run_dir) / "cycles" / f"{number:03d}"


def gate_dir(run_dir: Path, cycle: RunCycle | int, stage: str) -> Path:
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("gate stage must be a non-empty string")
    return cycle_dir(run_dir, cycle) / "checks" / stage.casefold().replace("_", "-")


def implementation_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "implementation"


def semantic_revision_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "semantic-revision"


def correction_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "correction"


def candidate_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "candidate"


def review_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "review"


def check_repair_attempt_dir(
    run_dir: Path, cycle: RunCycle | int, stage: str, attempt: int,
) -> Path:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("check-repair attempt must be a positive integer")
    return (
        cycle_dir(run_dir, cycle) / "check-repair" / stage.casefold().replace("_", "-")
        / "attempts" / f"{attempt:03d}"
    )


@dataclass(frozen=True)
class PipelineV2Dependencies:
    """Operations injected by the façade; none is provider-specific."""

    execute: Callable[..., RunResult]


@dataclass
class PipelineV2Coordinator:
    dependencies: PipelineV2Dependencies

    def run(self, **kwargs: Any) -> RunResult:
        return self.dependencies.execute(**kwargs)


__all__ = [
    "PipelineV2Coordinator", "PipelineV2Dependencies", "candidate_dir",
    "check_repair_attempt_dir", "correction_dir", "cycle_dir", "gate_dir",
    "implementation_dir", "review_dir", "semantic_revision_dir",
]
