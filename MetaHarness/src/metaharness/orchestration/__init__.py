"""Internal components of the MetaHarness orchestration state machine.

``metaharness.orchestrator`` is the façade and the only public entry point:
the modules in this package are its internal sub-domains and must never
import it back.  The dependency order is one-way::

    shared <- revision <- check_repair <- resume_validation
    shared <- candidate, scope_repair
"""
