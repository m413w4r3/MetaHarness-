type Data = Record<string, unknown>;

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function display(value: unknown): string {
  if (value === undefined || value === null || value === '') return '—';
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled';
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  return '—';
}

type Row = [label: string, value: unknown];

function Group({ title, rows }: { title: string; rows: Row[] }) {
  return (
    <div className="metaharness-effective-config__group" role="group" aria-label={title}>
      <h3>{title}</h3>
      {rows.map(([label, value]) => (
        <div className="metaharness-effective-config__row" key={label}>
          <span>{label}</span>
          <span>{display(value)}</span>
        </div>
      ))}
    </div>
  );
}

const ROLE_DEFAULTS: Row[] = [
  ['Planner default', 'planner_profile'],
  ['Reviewer default', 'final_reviewer_profile'],
  ['Reviser default', 'semantic_reviser_profile'],
  ['Repair default', 'check_repair_profile'],
];

/** Read-only rendering of `metaharness.get_config`; every field is optional. */
export function EffectiveConfigView({ config }: { config: Data }) {
  const repository = object(config.repository);
  const planning = object(config.planning);
  const revision = object(config.revision);
  const approval = object(config.approval);
  const publish = object(config.publish);
  const routing = object(config.routing);
  const defaults = object(config.defaults);
  const ui = object(config.ui);
  const checks = Array.isArray(config.checks) ? config.checks.map(object) : [];
  const roleRows = ROLE_DEFAULTS
    .map(([label, key]) => [label, routing[key as string] ?? defaults[key as string]] as Row)
    .filter(([, value]) => value !== undefined && value !== null && value !== '');
  const fingerprint = typeof config.config_fingerprint === 'string' ? config.config_fingerprint.slice(0, 12) : undefined;

  return (
    <div className="metaharness-effective-config">
      <Group title="Repository" rows={[
        ['Repository', repository.repo],
        ['Base ref', repository.base_ref],
        ['Remote', repository.remote],
      ]} />
      <Group title="Planning" rows={[
        ['Planning protocol', planning.protocol],
        ['Decomposition', planning.decomposition],
        ['Execution policy', planning.execution_mode_policy],
        ['Single-step max mutable paths', planning.single_step_max_mutable_paths],
        ['Staged-step max mutable paths', planning.staged_step_max_mutable_paths],
        ['Max steps per plan', planning.max_steps_per_plan],
      ]} />
      <Group title="Routing" rows={[
        ['Mechanical', routing.mechanical_profile],
        ['Reasoning', routing.reasoning_profile],
        ['Agentic', routing.agentic_profile],
        ...roleRows,
      ]} />
      <Group title="Revision & approval" rows={[
        ['Plan approval', approval.require_plan_approval],
        ['Semantic revision', revision.enabled],
        ['Max check repair attempts', revision.max_check_repair_attempts],
        ['Max review repair cycles', revision.max_review_repair_cycles],
      ]} />
      <Group title="Publish" rows={[
        ['Publish', publish.enabled],
        ['Publish mode', publish.mode],
        ['Publish remote', publish.remote],
      ]} />
      <Group title="Server" rows={[
        ['Max active runs', ui.max_active_runs],
        ...(fingerprint ? [['Config fingerprint', fingerprint] as Row] : []),
      ]} />
      <div className="metaharness-effective-config__group" role="group" aria-label="Checks">
        <h3>Checks</h3>
        {checks.length === 0
          ? <p className="metaharness-muted">No checks reported.</p>
          : <ul className="metaharness-effective-config__checks">
            {checks.map((check, index) => (
              <li key={`${String(check.id)}-${index}`}>
                <span aria-hidden="true">✓</span> <code>{display(check.id)}</code>
                {typeof check.description === 'string' && check.description && <span className="metaharness-muted"> — {check.description}</span>}
              </li>
            ))}
          </ul>}
        {config.checks_truncated === true && <p className="metaharness-muted">Check list truncated by MetaHarness.</p>}
      </div>
    </div>
  );
}

type DoctorCheck = { id: string; status: string; message: string };

function doctorSymbol(status: string): string {
  if (status === 'pass' || status === 'ok') return '✓';
  if (status === 'warn' || status === 'warning') return '!';
  return '✗';
}

/** Structured `metaharness doctor --json` report; falls back to a bounded raw view. */
export function DoctorReport({ report }: { report: Data }) {
  const checks: DoctorCheck[] = Array.isArray(report.checks)
    ? report.checks.map(object).map((check, index) => ({
      id: typeof check.id === 'string' ? check.id : `check-${index + 1}`,
      status: typeof check.status === 'string' ? check.status.toLowerCase() : 'unknown',
      message: typeof check.message === 'string' ? check.message : '',
    }))
    : [];
  const failed = checks.filter((check) => doctorSymbol(check.status) === '✗').length;
  const ok = typeof report.ok === 'boolean' ? report.ok : failed === 0;
  return (
    <details className="metaharness-doctor" open>
      <summary>Doctor — {ok ? 'all checks passed' : `${failed} failing check${failed === 1 ? '' : 's'}`}</summary>
      {checks.length
        ? <ul className="metaharness-doctor__checks">
          {checks.map((check, index) => (
            <li key={`${check.id}-${index}`} className={`metaharness-doctor__check metaharness-doctor__check--${check.status}`}>
              <span aria-hidden="true">{doctorSymbol(check.status)}</span>
              <span className="metaharness-doctor__status">{check.status}</span>
              <code>{check.id}</code>
              {check.message && <span>{check.message}</span>}
            </li>
          ))}
        </ul>
        : <pre className="metaharness-doctor__raw">{JSON.stringify(report, null, 2).slice(0, 4000)}</pre>}
    </details>
  );
}
