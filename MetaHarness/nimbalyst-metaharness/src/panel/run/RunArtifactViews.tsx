type Data = Record<string, unknown>;

function object(value: unknown): Data { return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {}; }
function text(value: unknown): string | undefined { return typeof value === 'string' ? value : value == null ? undefined : JSON.stringify(value, null, 2); }
function entries(value: unknown): Data[] { return Array.isArray(value) ? value.map(object) : []; }
function first(...values: unknown[]): unknown { return values.find((value) => value !== undefined && value !== null && value !== ''); }
function Empty({ children = 'No data available yet.' }: { children?: string }) { return <p className="metaharness-muted">{children}</p>; }
function Raw({ title, value }: { title: string; value: unknown }) { const content = text(value); return content ? <details className="metaharness-plan-artifact"><summary>{title}</summary><pre>{content}</pre></details> : null; }

export function ChecksView({ data }: { data: Data }) {
  const checks = object(data.checks);
  const rows = entries(Array.isArray(data.checks) ? data.checks : checks.results ?? checks.checks ?? checks.items ?? data.check_results);
  const cycleChecks = entries(data.cycle_artifacts).flatMap((cycle) => entries(object(object(cycle).checks).check_repair_attempts));
  const repairs = [...entries(checks.repairs ?? data.check_repairs ?? object(data.revision).check_repairs), ...cycleChecks];
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-checks-title"><h2 id="metaharness-checks-title">Checks</h2>
    {!rows.length && !repairs.length ? <Empty /> : <>
      {rows.map((check, index) => <CheckRow key={`${String(check.id ?? check.name ?? index)}-${index}`} check={check} />)}
      {repairs.map((repair, index) => <div key={index} className="metaharness-artifact-group"><h3>Repair attempt {String(repair.attempt ?? index + 1)}</h3>{entries(repair.checks ?? repair.results ?? object(repair.record).checks).map((check, item) => <CheckRow key={item} check={check} />)}<Raw title="Repair result" value={repair.record ?? repair.failure ?? repair.message ?? repair.output} /></div>)}
      <Raw title="Check output" value={checks.output ?? checks.message} />
    </>}
  </section>;
}

function CheckRow({ check }: { check: Data }) {
  const passed = check.passed ?? check.pass ?? check.success ?? (typeof check.exit_code === 'number' ? check.exit_code === 0 : undefined);
  const status = typeof passed === 'boolean' ? passed : ['passed', 'success'].includes(String(check.status ?? '').toLowerCase());
  const duration = first(check.duration_seconds, check.duration_s, check.duration);
  const seconds = typeof duration === 'number' ? `${duration.toFixed(1)}s` : typeof duration === 'string' ? duration : undefined;
  return <article className={`metaharness-check-row ${passed === false || ['failed', 'failure'].includes(String(check.status).toLowerCase()) ? 'is-failed' : ''}`}>
    <strong aria-label={status ? 'passed' : 'failed'}>{status ? '✓' : '✗'}</strong><b>{String(check.id ?? check.name ?? 'check')}</b>
    {seconds && <span>{seconds}</span>}{check.attempt !== undefined && <span>Attempt {String(check.attempt)}</span>}
    {text(check.message ?? check.output ?? [check.stdout_tail, check.stderr_tail].filter(Boolean).join('\n')) && <pre>{(text(check.message ?? check.output ?? [check.stdout_tail, check.stderr_tail].filter(Boolean).join('\n')) ?? '').slice(0, 4000)}</pre>}
  </article>;
}

export function ReviewView({ data }: { data: Data }) {
  const review = object(data.review), candidate = object(data.candidate);
  const reviewRaw = first(data.reviewer_raw, review.raw);
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-review-title"><h2 id="metaharness-review-title">Review</h2>
    {!Object.keys(review).length && reviewRaw == null ? <Empty /> : <dl className="metaharness-detail-grid">
      <Fact label="Verdict" value={first(review.verdict, review.decision)} /><Fact label="Category" value={first(review.category, review.outcome)} />
      <Fact label="Reviewer profile" value={first(review.reviewer_profile, review.profile_id, review.profile)} />
      <Fact label="Cycle" value={first(review.cycle, data.cycle)} /><Fact label="Candidate SHA" value={first(review.candidate_sha, review.commit_sha, candidate.commit_sha, candidate.sha, data.commit_sha)} />
    </dl>}
    <Raw title="Reviewer raw" value={reviewRaw} />
    <Raw title="Semantic revision" value={data.revision} />
  </section>;
}

function Fact({ label, value }: { label: string; value: unknown }) { return value == null ? null : <div><dt>{label}</dt><dd>{text(value)}</dd></div>; }

export function DiffView({ data, workspacePath, openFile }: { data: Data; workspacePath?: string; openFile?: (path: string) => void }) {
  const candidate = object(data.candidate);
  const files = Array.isArray(candidate.changed_files) ? candidate.changed_files.filter((value): value is string => typeof value === 'string') : [];
  const count = first(candidate.file_count, candidate.changed_file_count, files.length || undefined);
  const diff = first(candidate.diff_tail, candidate.diff_patch, data.diff_patch);
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-diff-title"><h2 id="metaharness-diff-title">Diff</h2>
    {count !== undefined && <p>{String(count)} changed file{count === 1 ? '' : 's'}</p>}
    {files.map((path) => <div className="metaharness-file-row" key={path}><code>{path}</code>{workspacePath && openFile && <button type="button" className="metaharness-secondary-button" onClick={() => { const normalized = workspaceFile(workspacePath, path); if (normalized) openFile(normalized); }}>Open file</button>}</div>)}
    {candidate.diff_truncated === true && <p className="metaharness-warning">Diff is truncated.</p>}
    {text(diff) ? <pre className="metaharness-diff">{text(diff)}</pre> : <Empty>Diff is not available.</Empty>}
  </section>;
}

function workspaceFile(root: string, input: string): string | undefined {
  const normalizedRoot = root.replace(/[\\/]+$/, '');
  if (!normalizedRoot || !input || input.startsWith('/') || input.startsWith('\\') || /^[A-Za-z]:/.test(input)) return undefined;
  const parts = input.replace(/\\/g, '/').split('/');
  const normalizedParts: string[] = [];
  for (const part of parts) {
    if (!part || part === '.') continue;
    if (part === '..') { if (!normalizedParts.length) return undefined; normalizedParts.pop(); }
    else normalizedParts.push(part);
  }
  if (!normalizedParts.length) return undefined;
  const result = `${normalizedRoot}/${normalizedParts.join('/')}`;
  const prefix = normalizedRoot.endsWith('/') ? normalizedRoot : `${normalizedRoot}/`;
  return result.startsWith(prefix) ? result : undefined;
}

export function UsageView({ data }: { data: Data }) {
  const usage = object(data.usage);
  const phases = entries(usage.phases ?? usage.by_phase ?? data.phase_usage);
  const implementer = object(usage.implementer);
  const known = ([
    { phase: 'planner', ...object(usage.planner) },
    { phase: 'implementer', ...object(implementer.total ?? usage.implementer) },
    ...entries(implementer.steps).map((step) => ({ phase: `step ${String(step.id ?? step.step_id ?? '')}`, ...object(step.usage ?? step) })),
    { phase: 'repair', ...object(usage.check_repair) },
    { phase: 'reviewer', ...object(usage.final_reviewer) },
    { phase: 'semantic revision', ...object(usage.semantic_reviser) },
    { phase: 'correction planner', ...object(usage.correction_planner) },
  ] as Data[]).filter((phase) => ['input_tokens', 'output_tokens', 'total_tokens', 'tokens'].some((key) => phase[key] != null));
  const phaseRows: Data[] = phases.length ? phases : known;
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-usage-title"><h2 id="metaharness-usage-title">Usage</h2>
    {!Object.keys(usage).length && !phases.length ? <Empty /> : <div className="metaharness-usage-list">
      {phaseRows.map((phase, index) => {
        const label = String(phase.phase ?? phase.name ?? phase.role ?? `Phase ${index + 1}`);
        return <dl className="metaharness-detail-grid" key={`${label}-${index}`}><dt>{label}</dt><dd>{token('Input', phase.input_tokens)}{token('Output', phase.output_tokens)}{token('Total', phase.total_tokens ?? phase.tokens)}</dd></dl>;
      })}
    </div>}
  </section>;
}
function token(label: string, value: unknown) { return value == null ? '' : `${label}: ${String(value)} `; }

export function DiagnosticsView({ data, artifact }: { data: Data; artifact?: Data }) {
  const diagnostics = object(data.diagnostics), setup = object(data.workspace_setup);
  const report = first(diagnostics.content, diagnostics.report, artifact?.content);
  const warnings = entries(diagnostics.warnings ?? data.warnings);
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-diagnostics-title"><h2 id="metaharness-diagnostics-title">Diagnostics</h2>
    {report == null && !warnings.length && !Object.keys(setup).length && data.failure == null && data.resumability == null ? <Empty /> : <>
      {report != null && <Raw title="MetaHarness report" value={report} />}
      {warnings.map((warning, index) => <p className="metaharness-warning" key={index}>{text(warning)}</p>)}
      <Raw title="Setup and results" value={first(setup.results, data.results)} /><Raw title="Failure details" value={data.failure} />
      <Raw title="Resumability" value={first(data.resumability, object(data.state).resumability)} />
    </>}
  </section>;
}

export function ResultsView({ data }: { data: Data }) {
  const results = first(data.results, object(data.workspace_setup).results, data.failure);
  return <section className="metaharness-detail-section" aria-labelledby="metaharness-results-title"><h2 id="metaharness-results-title">Results</h2>{results == null ? <Empty /> : <Raw title="Run results" value={results} />}</section>;
}
