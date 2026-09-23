# MetaHarness MCP tools

The extension publishes a deliberately small MCP surface. `start`, `stop`, and
`doctor` remain extension backend methods but are not registered as MCP tools.

## Read-only

`metaharness.status`, `metaharness.get_config`, `metaharness.model_profiles`,
`metaharness.list_runs`, `metaharness.get_run`, `metaharness.progress`, and
`metaharness.get_artifact` inspect configuration or run data without changing
MetaHarness state.

## Mutations

`metaharness.create_run`, `metaharness.approve_run`, `metaharness.approve_scope`,
`metaharness.resume_run`, and `metaharness.recover_plan` can change state. Use
them only when the user explicitly requests the action. Never approve or reject
a plan automatically, recover a plan without user supplied content and clear
intent, or launch multiple runs to compensate for an error. Approval calls must
name both `runId` and `decision`; no tool infers the latest run. Run creation
requires `spec`, resume requires `runId`, and plan recovery requires a non-empty
`input.plan` plus `runId`.
