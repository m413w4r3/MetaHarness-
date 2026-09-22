META PLAN v2

STATUS: READY
TITLE: AW-001 trusted validation

OBJECTIVE
Implement AW-001 with complete deterministic validation.

CONSTRAINTS
Use only the trusted catalogue checks.

REQUIRED_CHECKS
- lint
- typecheck
- test
- test-integration
- alembic-heads

EXECUTION_MODE: SINGLE
STEP_COUNT: 1

BEGIN STEP S01
TITLE: Implement AW-001
EXECUTION_CLASS: REASONING
DEPENDS_ON: NONE

OBJECTIVE
Implement the requested change.

READ_SET
- backend/pyproject.toml :: project metadata

WRITE_SET
- backend/pyproject.toml

CREATE_SET
NONE

DELETE_SET
NONE

INSTRUCTIONS
1. Implement the scoped change.

VERIFY
- Run the selected trusted checks.

FORBIDDEN
- Do not change paths outside the declared sets.

END STEP S01

ACCEPTANCE
All trusted required checks pass.

TESTS
The selected trusted checks are the final evidence.

RISKS
NONE

BLOCKERS
NONE

END META PLAN
