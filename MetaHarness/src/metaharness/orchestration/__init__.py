"""Internal components of the MetaHarness orchestration state machine.

``metaharness.orchestrator`` is the façade and the only public entry point:
the modules in this package are its internal sub-domains and must never
import it back.  The dependency order is one-way::

    shared <- revision <- check_repair <- resume_validation
    shared <- candidate, scope_repair
    pipeline_v2 <- recovery <- worker_recovery, check_recovery, review_recovery

``recovery`` applies :func:`metaharness.recovery_policy.classify_failure`:
it owns durable recovery budgets, attempt records, the ``recovery.*`` trace
and the projection of a failure onto ``FAILED`` or a ``WAITING_*`` state.
The ``*_recovery`` services run the phase-specific actions (worker rollback
and executor fallback, check infrastructure retries, reviewer transport and
evidence recovery); the Git transaction every attempt shares lives in
:mod:`metaharness.attempt_transaction`.
"""
