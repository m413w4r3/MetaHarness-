export type RunCategory = 'active' | 'awaiting-action' | 'completed' | 'failed' | 'other';

const ACTIVE_STATUSES = new Set([
  'created', 'planning', 'worktree_ready', 'preparing', 'implementing',
  'validating', 'pre_revision_validating', 'revising', 'revalidating',
  // `approved` is transient: the orchestrator commits or publishes next.
  'reviewing', 'approved', 'publishing',
]);
const AWAITING_STATUSES = new Set([
  'blocked', 'awaiting_plan_approval', 'waiting_scope_approval', 'plan_rejected',
]);
const COMPLETED_STATUSES = new Set(['published', 'committed']);
const FAILED_STATUSES = new Set(['failed', 'interrupted']);

export function classifyRunStatus(status: string): RunCategory {
  const normalized = status.toLowerCase();
  if (ACTIVE_STATUSES.has(normalized)) return 'active';
  if (AWAITING_STATUSES.has(normalized)) return 'awaiting-action';
  if (COMPLETED_STATUSES.has(normalized)) return 'completed';
  if (FAILED_STATUSES.has(normalized)) return 'failed';
  return 'other';
}
