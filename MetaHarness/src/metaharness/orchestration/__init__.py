"""Internal components of the MetaHarness orchestration state machine.

``metaharness.orchestrator`` is the façade and the only public entry point:
the modules in this package are its internal sub-domains and must never
import it back.  The dependency order is one-way::

    shared <- gates, publication, audit
    shared <- run_resume
    pipeline_v2 <- recovery
    pipeline_v2 <- worker_attempt <- step_execution
    run_bootstrap <- run_composition <- runtime
    run_failure <- runtime

The run authorities sit next to the kernel that composes them: ``run_bootstrap``
owns the planning, approval, worktree and setup of a new run,
``run_composition`` the immutable context, the ``PipelineV2Operations`` wiring
and the cycle authority, ``run_failure`` the durable projection of a failure
that left its recovery loop.  Runtime observability is kept beside those
authorities in ``runtime``.

The step services split the one step transaction by transaction:
``step_execution`` runs one approved step as a bounded ladder of attempts and
owns its acceptance boundary, while ``worker_attempt`` runs the single worker
request and normalizes its candidate result.

``recovery`` applies :func:`metaharness.recovery_policy.classify_failure`:
it owns durable recovery budgets, attempt records, the ``recovery.*`` trace
and the projection of a failure onto ``FAILED`` or a ``WAITING_*`` state.
The recovery services run the phase-specific actions (worker rollback and
executor fallback, check infrastructure retries); the Git transaction
every attempt shares lives in :mod:`metaharness.attempt_transaction`.

Post-implementation authority is single: the deterministic gate alternates
with the AUDIT service until the candidate is ready, then the candidate is
pushed and published.  ``run_resume`` verifies the run branch and restores
its last accepted commit before rebuilding the effective plan and selection.
"""
