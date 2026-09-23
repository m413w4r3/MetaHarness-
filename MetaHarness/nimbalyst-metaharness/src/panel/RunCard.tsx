import type { RunSummary } from '../types';
import { classifyRunStatus } from './runStatus';
import { StatusBadge } from './StatusBadge';

function updatedLabel(value: string | null | undefined): string | undefined {
  if (!value) return undefined;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(date);
}

function failureLabel(failure: unknown): string | undefined {
  if (typeof failure === 'string') return failure;
  if (typeof failure === 'object' && failure !== null && 'reason' in failure) {
    const reason = (failure as { reason?: unknown }).reason;
    if (typeof reason === 'string') return reason;
  }
  return undefined;
}

export function RunCard({ run, onClick }: { run: RunSummary; onClick: () => void }) {
  const runId = typeof run.run_id === 'string' ? run.run_id : 'Unknown run';
  const status = typeof run.status === 'string' ? run.status : 'unknown';
  const updated = updatedLabel(run.updated_at);
  const failure = failureLabel(run.failure);

  return (
    <button className="metaharness-run-card" type="button" onClick={onClick}>
      <span className="metaharness-run-card__topline">
        <strong className="metaharness-run-card__id">{runId}</strong>
        <StatusBadge status={status} />
        {typeof run.commit_sha === 'string' && <code>{run.commit_sha}</code>}
      </span>
      {run.plan_title && <span className="metaharness-run-card__title">{run.plan_title}</span>}
      <span className="metaharness-run-card__meta">
        {updated && <span>Updated {updated}</span>}
        {failure && <span className="metaharness-run-card__failure">{failure}</span>}
      </span>
      <span className="sr-only">{classifyRunStatus(status)}</span>
    </button>
  );
}
