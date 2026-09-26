"""Internal components of the MetaHarness orchestration state machine.

``metaharness.orchestrator`` is the façade and the only public entry point:
the modules in this package are its internal sub-domains and must never
import it back.  The dependency order is one-way::

    shared <- check_failure <- check_scope <- gate_recovery, gate_acceptance
    shared <- revision <- durable_readers, cycle_loader, gates
    durable_readers, cycle_loader <- resume_integrity
    shared <- candidate, correction_scope
    pipeline_v2 <- recovery <- worker_recovery, check_recovery, review_recovery
    pipeline_v2 <- step_authority <- worker_attempt <- step_execution
    contract_recovery <- step_execution, step_replan
    step_acceptance <- step_execution
    run_bootstrap <- run_composition <- runtime
    run_observability, run_failure <- runtime

The run authorities sit next to the kernel that composes them: ``run_bootstrap``
owns the planning, approval, worktree and setup of a new run,
``run_composition`` the immutable context, the ``PipelineV2Operations`` wiring
and the cycle authority, ``run_failure`` the durable projection of a failure
that left its recovery loop and ``run_observability`` the trace, the session
metadata and the diagnostics of a run.

The step services split the one old step transaction by transaction:
``step_execution`` runs one approved step as a bounded ladder of attempts,
``worker_attempt`` runs the single worker request and normalizes its candidate
result, ``step_acceptance`` owns the durable commit boundary and its resume,
``contract_recovery`` owns the durable semantic contract repair slot and the
effective repaired authority, and ``step_replan`` owns the red-gate rung that
rewrites one step's contract and re-executes its suffix.

``recovery`` applies :func:`metaharness.recovery_policy.classify_failure`:
it owns durable recovery budgets, attempt records, the ``recovery.*`` trace
and the projection of a failure onto ``FAILED`` or a ``WAITING_*`` state.
The ``*_recovery`` services run the phase-specific actions (worker rollback
and executor fallback, check infrastructure retries, reviewer transport and
evidence recovery); the Git transaction every attempt shares lives in
:mod:`metaharness.attempt_transaction`.

The review domain is split by authority: ``candidate_review`` owns the
reviewer decision, ``review_correction`` the review-driven correction routes
and the candidate evidence they read, ``check_replan_service`` the red-gate
cycle replan -- which never consults the reviewer -- ``correction_scope`` the
one mutable-scope policy every correction applies, and ``semantic_revision``
the semantic revision pass and its reviser/repair worker transaction.

The resume domain is split the same way: ``durable_readers`` owns every
fail-closed reader of a durable artifact, ``cycle_loader`` the cycle records,
the review and check-replan bindings, the correction plans and the mutable
scopes they authorize, and ``resume_integrity`` the single gate in front of
every resumed checkpoint, which rebuilds a ``ResumedRun`` from those artifacts.
"""
