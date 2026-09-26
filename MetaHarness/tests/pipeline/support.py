"""Shared helpers for the split pipeline-v2 test modules.

Every helper here is a *port*: scripted chat/executor doubles, the git remote
transport, the durable checkpoint boundaries and the recovery budgets.  The
tests patch no orchestration internal: a remote outage is a real unreachable
remote, a crash is raised at a real durable boundary and an authority failure
is produced by tampering with a real artifact.
"""

from __future__ import annotations

from unittest import mock

from metaharness.gitops import build_run_branch

from tests.pipeline_support import (
    PipelineHarness,
    ScriptedChat,
    ScriptedWorkers,
    check_repair_result,
    correction_plan,
    git,
    initial_plan,
    ladder_ledger,
    ladder_strategies,
    plan,
    repaired_step_contract,
    review,
    write,
)

SPEC = "Make feature.txt good.\n"
STEP = ("S01", "feature.txt", "Write the feature")

__all__ = [
    "PipelineHarness", "ScriptedChat", "ScriptedWorkers", "SPEC", "STEP",
    "check_repair_result", "correction_plan", "git", "initial_plan",
    "ladder_ledger", "ladder_strategies", "plan", "review", "write",
    "repaired_step_contract", "run_branch", "break_remote", "restore_remote",
    "divergent_run_branch", "move_run_branch", "reject_pushes",
    "crash_at_checkpoint", "crash_on_review", "crash_on_revision",
]


# --- the git transport port -------------------------------------------------

def run_branch(run_id: str = "run", slug: str = "add-the-feature") -> str:
    """The run branch this harness's scripted plan publishes to."""

    return build_run_branch(slug, run_id)


def break_remote(harness: PipelineHarness) -> None:
    """Point ``origin`` at a repository that does not exist."""

    git(harness.repo, "remote", "set-url", "origin", str(harness.root / "unreachable.git"))


def restore_remote(harness: PipelineHarness) -> None:
    git(harness.repo, "remote", "set-url", "origin", str(harness.remote))


def _divergent_commit(harness: PipelineHarness) -> str:
    """One committed tip that is not an ancestor of any candidate."""

    git(harness.repo, "checkout", "-q", "-b", "harness-divergence")
    (harness.repo / "divergent.txt").write_text("divergent\n", encoding="utf-8")
    git(harness.repo, "add", "divergent.txt")
    git(harness.repo, "commit", "-qm", "divergent tip")
    sha = git(harness.repo, "rev-parse", "HEAD")
    git(harness.repo, "checkout", "-q", "main")
    git(harness.repo, "branch", "-q", "-D", "harness-divergence")
    return sha


def divergent_run_branch(harness: PipelineHarness, branch: str) -> str:
    """Publish an unrelated tip to *branch*, without touching the base ref."""

    sha = _divergent_commit(harness)
    git(harness.repo, "push", "-q", "origin", f"{sha}:refs/heads/{branch}")
    return sha


def move_run_branch(harness: PipelineHarness, branch: str) -> str:
    """Force *branch* onto an unrelated tip while a run is in flight."""

    sha = _divergent_commit(harness)
    git(harness.repo, "push", "-q", "--force", "origin", f"{sha}:refs/heads/{branch}")
    return sha


def reject_pushes(harness: PipelineHarness) -> None:
    """Make the bare remote refuse every push while staying readable."""

    hook = harness.remote / "hooks" / "pre-receive"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)


# --- crash-injection ports --------------------------------------------------

def crash_at_checkpoint(
    orchestrator, phase: str, *, occurrence: int = 1, attempt: int | None = None,
):
    """Interrupt the run where it durably enters *phase*.

    The checkpoint writer is the run's durable boundary: raising inside it is
    exactly an abruptly killed process, and the next ``resume`` must observe
    the state the boundary described.
    """

    from metaharness.resume import ResumePhase

    runtime = orchestrator._runtime
    real = type(runtime).write_checkpoint
    seen: list[int] = []

    def write(run_dir, next_phase, **fields):
        if next_phase is ResumePhase(phase) and (
            attempt is None or fields.get("check_repair_attempt") == attempt
        ):
            seen.append(1)
            if len(seen) == occurrence:
                raise RuntimeError(f"crash at {phase}")
        return real(run_dir, next_phase, **fields)

    return mock.patch.object(type(runtime), "write_checkpoint", staticmethod(write))


def crash_on_review(orchestrator, number: int = 1):
    """Interrupt the run at the *number*-th candidate review."""

    reviews = orchestrator._runtime.reviews
    real = type(reviews).review_candidate
    calls: list[int] = []

    def review_or_crash(owner, *args, **kwargs):
        calls.append(1)
        if len(calls) == number:
            raise RuntimeError("crash at final review")
        return real(owner, *args, **kwargs)

    return mock.patch.object(type(reviews), "review_candidate", review_or_crash)


def crash_on_revision(orchestrator):
    """Interrupt the run before the semantic reviser runs."""

    revisions = orchestrator._runtime.semantic_revision
    return mock.patch.object(
        type(revisions), "run_revision_with_recovery",
        side_effect=RuntimeError("crash before semantic worker"),
    )
