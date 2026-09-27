# Current pipeline: diagnostic gate and active audit

A new pipeline v2 run executes its approved implementation steps, records the
POST_IMPLEMENTATION gate evidence, invokes one writable `AUDITOR` code agent,
then reruns the same trusted checks from the harness. A red diagnostic gate is
input to audit. A green diagnostic gate still gets a SPEC audit when the batch
has a diff. A fatal integrity failure and a blocking harness preflight remain
subject to the global recovery policy before audit.

The auditor receives the full SPEC, executed plan and normalizations, settled
step records, cumulative diff, changed paths, out of scope paths, baseline and
candidate verdicts, new failure IDs, bounded failure excerpts, warnings and
retry history. It may edit source, tests and build files. The run's `ScopePolicy`
hard-deny list remains the only path prohibition. Newly touched refactor paths
are recorded in the audit report; the prompt asks the agent to keep them within
15. Audit does not require the agent to run checks from its own sandbox.

The final message ends in `META AUDIT v1`. `DONE` means the agent has completed
its available work. `NEEDS_WORK` records its remaining technical work; it never
asks a human or claims the harness infrastructure is unavailable.
`SPEC_DECISION` is reserved for a product decision absent from the SPEC and
routes to `WAIT_HUMAN`.

A changed audit tree passes path and staged-content safety checks, is committed
on the run branch, then receives a fresh deterministic gate run from the
harness. At most two audit passes are allowed for this batch. Green evidence is
bound to the candidate and publication. Red evidence after both passes records
`AUDIT_REMAINING` with the agent's remaining work and the gate failure IDs.
C8 will consume that state for the next iteration.

New run options and execution selections contain one `audit_profile` authority.
Older run schemas and correction-cycle checkpoints are incompatible with this
pipeline.
