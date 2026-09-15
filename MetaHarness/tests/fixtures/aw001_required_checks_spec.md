# AW-001 regression fixture

Implement the AutoWork change described by this SPEC.

Acceptance requires lint, typecheck, the normal test suite, PostgreSQL-backed
integration tests, and exact Alembic migration-head evidence. The normal test
command may skip PostgreSQL tests, so integration validation is a separate
mandatory gate.
