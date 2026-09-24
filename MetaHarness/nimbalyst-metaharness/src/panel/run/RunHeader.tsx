import { StatusBadge } from '../StatusBadge';

type Data = Record<string, unknown>;

export function shortSha(value?: string | null): string {
  return value ? value.slice(0, 7) : '';
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function failureText(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object') {
    const failure = value as Data;
    const reason = text(failure.reason);
    const detail = text(failure.detail);
    return reason && detail ? `${reason} · ${detail}` : reason ?? detail;
  }
  return undefined;
}

function Sha({ label, value }: { label: string; value: unknown }) {
  const full = text(value);
  if (!full) return null;
  return <span className="metaharness-run-header__sha">{label} <code title={full} onClick={(event) => {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(event.currentTarget);
    selection?.removeAllRanges();
    selection?.addRange(range);
  }}>{shortSha(full)}</code></span>;
}

export function RunHeader({ runId, data, loading, onBack, onRefresh }: {
  runId: string;
  data: Data;
  loading: boolean;
  onBack: () => void;
  onRefresh: () => void;
}) {
  const status = text(data.status) ?? 'unknown';
  const title = text(data.plan_title);
  const updated = text(data.updated_at);
  const failure = failureText(data.failure);
  const candidate = data.candidate && typeof data.candidate === 'object' ? data.candidate as Data : {};
  return <>
    <div className="metaharness-run-header__actions">
      <button className="metaharness-link-button" type="button" onClick={onBack}>← Back</button>
      <button className="metaharness-secondary-button" type="button" onClick={onRefresh} disabled={loading}>Refresh</button>
    </div>
    <header className="metaharness-run-header">
      <div><p className="metaharness-run-header__id">{runId}</p><h1 id="metaharness-run-detail-title">{title ?? 'Run detail'}</h1></div>
      <StatusBadge status={status} />
      <dl className="metaharness-run-header__facts">
        {updated && <div><dt>Updated</dt><dd>{updated}</dd></div>}
        <Sha label="Candidate" value={candidate.commit_sha ?? candidate.sha ?? data.candidate_sha} />
        <Sha label="Commit" value={data.commit_sha} />
        {failure && <div className="metaharness-run-header__failure"><dt>Failure</dt><dd>{failure}</dd></div>}
      </dl>
    </header>
  </>;
}
