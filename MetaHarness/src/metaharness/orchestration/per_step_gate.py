"""The optional fast gate a step pays before its own commit.

``[gate] per_step`` names a subset of the trusted check catalogue.  After a
step mutates its tree, and *before* that step is committed, only those checks
run: their candidate result is compared against the run's baseline of the base
commit, and only a *new* regression refuses the step.  The refusal hands the
worker the bounded evidence of the regression -- the check id, its new failing
test ids, a short excerpt of its output and the changed paths -- never the whole
log, and never a decision: the harness is the only authority on the checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..baseline import judge_results, preflight_skips
from ..result import atomic_write_text
from ..state import RunStateStore
from ..validation import bounded_tail, config_with_check_authority, run_checks

if TYPE_CHECKING:  # pragma: no cover - the service is the composition root
    from .gates import GateService


_FEEDBACK_EXCERPT_BYTES = 2000


@dataclass(frozen=True)
class PerStepGateOutcome:
    """What the optional fast gate answered for one mutated step tree."""

    enabled: bool
    passed: bool
    feedback: str = ""
    warnings: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()


def _excerpt_text(result: Any) -> str:
    """The bounded log text of one check result, whatever shape it has."""

    stdout = str(getattr(result, "stdout_tail", "") or "")
    stderr = str(getattr(result, "stderr_tail", "") or "")
    return f"{stdout}\n{stderr}".strip()


def _regression_feedback(
    *,
    regressions: Any,
    results: Any,
    changed_paths: Any,
) -> str:
    """The bounded evidence one refused step hands back to its worker."""

    by_id = {result.name: result for result in results}
    lines = ["The fast per-step gate refused this step tree."]
    # Put durable path facts before the bounded log excerpt. The retry
    # addendum has its own size limit and should not clip this actionable data.
    if changed_paths:
        lines.append("CHANGED PATHS:")
        lines.extend(f"- {path}" for path in tuple(changed_paths)[:20])
    for judgement in regressions:
        lines.append(f"CHECK: {judgement.check_id}")
        if judgement.new_failure_ids:
            lines.append("NEW FAILING TEST IDS:")
            lines.extend(f"- {item}" for item in judgement.new_failure_ids[:10])
        result = by_id.get(judgement.check_id)
        if result is not None:
            excerpt = bounded_tail(_excerpt_text(result), _FEEDBACK_EXCERPT_BYTES).strip()
            if excerpt:
                lines.append("FAILURE EXCERPT:")
                lines.append(excerpt)
    return "\n".join(lines)


def record_step_warnings(store: RunStateStore, warnings: Any) -> None:
    """Keep the non-blocking verdicts of the fast gate durable."""

    state = store.load()
    existing = [
        item for item in (state.get("check_warnings") or []) if isinstance(item, str)
    ]
    merged = list(dict.fromkeys((*existing, *warnings)))
    if merged != existing:
        store.update_metadata(check_warnings=merged)


def run_per_step_gate(
    service: "GateService",
    *,
    run_dir: Path,
    worktree: Path,
    base_sha: str,
    step_dir: Path,
    check_ids: Any,
    changed_paths: Any,
) -> PerStepGateOutcome:
    """Run the configured fast checks and judge only their *new* failures."""

    if not check_ids:
        return PerStepGateOutcome(enabled=False, passed=True)
    runtime = service.runtime
    check_config, _ids = config_with_check_authority(
        runtime.config, run_dir, requested_check_ids=list(check_ids),
        expected_sha256=runtime.approved_check_authority_sha256(run_dir),
    )
    selected = check_config.select_checks(list(check_ids))
    skipped = preflight_skips(run_dir, selected)
    baseline = service.gate_baseline(
        run_dir=run_dir, worktree=worktree, base_sha=base_sha,
        check_config=check_config, checks=selected, skipped=skipped,
    )
    directory = step_dir / "per-step-gate"
    directory.mkdir(parents=True, exist_ok=True)
    evaluated = tuple(check for check in selected if check.id not in skipped)
    results = run_checks(
        worktree, check_config, required_check_ids=[check.id for check in evaluated],
        logs_dir=directory / "checks", secrets=runtime.secrets,
    )
    judgements = judge_results(baseline, results, evaluated, skipped=skipped)
    regressions = tuple(item for item in judgements if item.blocking)
    warnings = tuple(item.warning for item in judgements if item.warning)
    payload = {
        "schema_version": 1,
        "step_dir": step_dir.name,
        "check_ids": list(check_ids),
        "baseline_check_config_sha": baseline.check_config_sha,
        "regressions": [item.check_id for item in regressions],
        "checks": [
            {
                "id": item.check_id,
                "verdict": item.verdict.value,
                "new_failure_ids": list(item.new_failure_ids),
                "warning": item.warning,
            }
            for item in judgements
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(directory / f"attempt-{_attempt_index(step_dir):02d}.json", text)
    atomic_write_text(directory / "per-step-gate.json", text)
    if not regressions:
        return PerStepGateOutcome(
            enabled=True, passed=True, warnings=warnings, regressions=(),
        )
    feedback = _regression_feedback(
        regressions=regressions, results=results, changed_paths=changed_paths,
    )
    return PerStepGateOutcome(
        enabled=True, passed=False, feedback=feedback, warnings=warnings,
        regressions=tuple(item.check_id for item in regressions),
    )


def _attempt_index(step_dir: Path) -> int:
    """The 1-based index of the worker attempt this gate just judged."""

    root = step_dir / "attempts"
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    return index


__all__ = ["PerStepGateOutcome", "record_step_warnings", "run_per_step_gate"]
