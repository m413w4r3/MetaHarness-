import { shortSha } from './RunHeader';
import { StatusBadge } from '../StatusBadge';

type Data = Record<string, unknown>;

function text(value: unknown): string | undefined {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : undefined;
}

function count(value: unknown): number | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.length;
}

function record(value: unknown): Data {
  return typeof value === 'object' && value !== null ? value as Data : {};
}

export function StepCard({ step }: { step: Data }) {
  const id = text(step.id) ?? 'Step';
  const title = text(step.title) ?? text(step.description) ?? 'Untitled step';
  const status = text(step.status) ?? 'waiting';
  const result = record(step.result);
  const checks = step.checks ?? step.check_state ?? result.checks;
  const repair = step.repair_state ?? step.repair ?? result.repair;
  const commit = text(step.commit_sha) ?? text(result.commit_sha);
  const files = count(step.changed_files) ?? count(result.changed_files);
  return <article className="metaharness-step-card">
    <h3><code>{id}</code> {title}</h3>
    {text(step.description) && step.description !== title && <p>{text(step.description)}</p>}
    <dl className="metaharness-step-card__facts">
      <div><dt>Status</dt><dd><StatusBadge status={status} /></dd></div>
      {(text(step.execution_class) || text(step.execution_mode)) && <div><dt>Execution</dt><dd>{text(step.execution_class) ?? text(step.execution_mode)}</dd></div>}
      {(text(step.profile_id) || text(step.profile)) && <div><dt>Profile</dt><dd>{text(step.profile_id) ?? text(step.profile)}</dd></div>}
      {commit && <div><dt>Commit</dt><dd><code title={commit}>{shortSha(commit)}</code></dd></div>}
      {files !== undefined && <div><dt>Changed files</dt><dd>{files}</dd></div>}
      {checks !== undefined && <div><dt>Checks</dt><dd><pre>{typeof checks === 'string' ? checks : JSON.stringify(checks, null, 2)}</pre></dd></div>}
      {repair !== undefined && <div><dt>Repair</dt><dd><pre>{typeof repair === 'string' ? repair : JSON.stringify(repair, null, 2)}</pre></dd></div>}
      {text(step.failure_reason) && <div><dt>Failure</dt><dd>{text(step.failure_reason)}</dd></div>}
    </dl>
  </article>;
}
