"""Internal components of the MetaHarness orchestration state machine.

``metaharness.orchestrator`` is the façade and the only public entry point:
the modules in this package are its internal sub-domains and must never
import it back.  The dependency order is one-way::

    shared <- check_failure <- durable_readers
    durable_readers <- resume_integrity
    shared <- candidate, audit
    pipeline_v2 <- recovery <- worker_recovery, check_recovery
    pipeline_v2 <- step_authority <- worker_attempt <- step_execution
    step_acceptance <- step_execution
    run_bootstrap <- run_composition <- runtime
    run_observability, run_failure <- runtime

The run authorities sit next to the kernel that composes them: ``run_bootstrap``
owns the planning, approval, worktree and setup of a new run,
``run_composition`` the immutable context, the ``PipelineV2Operations`` wiring
and the cycle authority, ``run_failure`` the durable projection of a failure
that left its recovery loop and ``run_observability`` the trace, the session
metadata and the diagnostics of a run.

The step services split the one step transaction by transaction:
``step_execution`` runs one approved step as a bounded ladder of attempts,
``worker_attempt`` runs the single worker request and normalizes its candidate
result, and ``step_acceptance`` owns the durable commit boundary and its resume.

``recovery`` applies :func:`metaharness.recovery_policy.classify_failure`:
it owns durable recovery budgets, attempt records, the ``recovery.*`` trace
and the projection of a failure onto ``FAILED`` or a ``WAITING_*`` state.
The ``*_recovery`` services run the phase-specific actions (worker rollback
and executor fallback, check infrastructure retries); the Git transaction
every attempt shares lives in :mod:`metaharness.attempt_transaction`.

Post-implementation authority is single: the deterministic gate alternates
with the AUDIT service until the candidate is ready, then the candidate is
pushed and published.  The resume domain is split the same way:
``durable_readers`` owns every fail-closed reader of a durable artifact and
``resume_integrity`` the single gate in front of every resumed checkpoint,
which rebuilds a ``ResumedRun`` from those artifacts.
"""
