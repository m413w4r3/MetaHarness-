import { useCallback, useEffect, useRef, useState } from 'react';
import type { RunSummary as RunSummaryData } from '../../types';
import { RunApproval } from './RunApproval';
import { RunHeader } from './RunHeader';
import { RunSummary } from './RunSummary';
import { PlanView } from './PlanView';
import { StepsView } from './StepsView';
import { ProgressView } from './ProgressView';
import { ChecksView, DiagnosticsView, DiffView, ResultsView, ReviewView, UsageView } from './RunArtifactViews';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type Data = Record<string, unknown>;

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function entries(value: unknown): Data[] { return Array.isArray(value) ? value.map(object) : []; }

function unwrap(value: unknown): unknown {
  const result = object(value);
  if (result.ok === false) {
    const error = object(result.error);
    throw new Error(typeof error.message === 'string' ? error.message : 'MetaHarness backend call failed.');
  }
  return value;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unexpected MetaHarness error.';
}

export function RunDetail({ runId, run, callBackendTool, onBack, pollIntervalMs = 1000, workspacePath, openFile }: {
  runId: string;
  run?: RunSummaryData;
  callBackendTool?: BackendCall;
  onBack: () => void;
  pollIntervalMs?: number;
  workspacePath?: string;
  openFile?: (path: string) => void;
}) {
  const [detail, setDetail] = useState<Data>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [actionBusy, setActionBusy] = useState(false);
  const [activeTab, setActiveTab] = useState('Overview');
  const [artifact, setArtifact] = useState<Data>({});
  const actionInFlight = useRef(false);
  const pollInFlight = useRef(false);
  const refreshSequence = useRef(0);
  useEffect(() => { setArtifact({}); }, [runId, detail?.cycle]);
  const refresh = useCallback(async (preserveError = false, force = false) => {
    if (!callBackendTool) return;
    if (pollInFlight.current && !force) return;
    pollInFlight.current = true;
    const sequence = ++refreshSequence.current;
    setLoading(true);
    if (!preserveError) setError('');
    try {
      const result = object(unwrap(await callBackendTool('metaharness.get_run', { runId })));
      if (sequence === refreshSequence.current) setDetail(object(result.run ?? result));
    } catch (caught) {
      if (sequence === refreshSequence.current) setError(errorMessage(caught));
    } finally {
      if (sequence === refreshSequence.current) {
        pollInFlight.current = false;
        setLoading(false);
      }
    }
  }, [callBackendTool, runId]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!callBackendTool) return undefined;
    const timer = window.setInterval(() => { void refresh(true); }, pollIntervalMs);
    return () => window.clearInterval(timer);
  }, [callBackendTool, pollIntervalMs, refresh]);
  const decide = useCallback(async (kind: 'plan' | 'scope', decision: 'APPROVE' | 'REJECT', input?: Data) => {
    if (!callBackendTool || actionInFlight.current) return;
    actionInFlight.current = true;
    setActionBusy(true);
    setError('');
    try {
      const tool = kind === 'plan' ? 'metaharness.approve_run' : 'metaharness.approve_scope';
      const result = await callBackendTool(tool, { runId, input: input ?? { decision } });
      unwrap(result);
      await refresh(true, true);
    } catch (caught) {
      setError(errorMessage(caught));
      await refresh(true, true);
    } finally {
      actionInFlight.current = false;
      setActionBusy(false);
    }
  }, [callBackendTool, refresh, runId]);
  const displayed = { ...object(run), ...detail };
  useEffect(() => {
    if (!detail || !callBackendTool) return;
    const cycle = typeof detail.cycle === 'number' && detail.cycle > 0 ? detail.cycle : 1;
    const candidate = object(detail.candidate);
    const diagnostics = object(detail.diagnostics);
    const state = object(detail.state);
    const cycleData = entries(detail.cycle_artifacts).find((item) => item.number === cycle);
    const gateStage = object(object(cycleData).checks).stage;
    const gate = typeof gateStage === 'string' && /^[a-z-]+$/.test(gateStage)
      ? gateStage
      : state.status === 'checking' || state.status === 'review' ? 'post-implementation' : 'post-review';
    const requests: Array<[string, string, boolean]> = [];
    const cycleRoot = `cycles/${String(cycle).padStart(3, '0')}`;
    if (activeTab === 'Checks' && detail.checks == null && artifact['checks.json'] === undefined) requests.push(['checks.json', `${cycleRoot}/checks/post-implementation/checks.json`, true]);
    if (activeTab === 'Review') {
      if (detail.review == null && artifact['review.json'] === undefined) requests.push(['review.json', `${cycleRoot}/review/review.json`, true]);
      if (detail.reviewer_raw == null && artifact['reviewer.raw.md'] === undefined) requests.push(['reviewer.raw.md', `${cycleRoot}/review/reviewer.raw.md`, true]);
      if (detail.revision == null && artifact['revision.json'] === undefined) requests.push(['revision.json', `${cycleRoot}/semantic-revision/report.json`, true]);
    }
    if (activeTab === 'Diff') {
      if (candidate.diff_tail == null && artifact['diff.patch'] === undefined) requests.push(['diff.patch', `${cycleRoot}/checks/${gate}/diff.patch`, true]);
      if (candidate.changed_files == null && artifact['changed-files.txt'] === undefined) requests.push(['changed-files.txt', `${cycleRoot}/checks/${gate}/changed-files.txt`, true]);
    }
    if (activeTab === 'Diagnostics' && diagnostics.content == null && artifact['diagnostics.json'] === undefined) requests.push(['diagnostics.json', 'diagnostics.json', true]);
    if (!requests.length) return;
    let cancelled = false;
    void Promise.all(requests.map(async ([key, name]) => {
      try {
        const response = object(unwrap(await callBackendTool('metaharness.get_artifact', { runId, name })));
        if (!cancelled) setArtifact((current) => ({ ...current, [key]: response as Data }));
      } catch { if (!cancelled) setArtifact((current) => ({ ...current, [key]: { exists: false } })); }
    }));
    return () => { cancelled = true; };
  }, [activeTab, artifact, callBackendTool, detail, runId]);
  const tabs = ['Overview', 'Plan', 'Steps', 'Checks', 'Review', 'Diff', 'Usage', 'Logs', 'Diagnostics', 'Results'];
  return <section className="metaharness-dashboard metaharness-run-detail-page" aria-labelledby="metaharness-run-detail-title">
    <RunHeader runId={runId} data={displayed} loading={loading} onBack={onBack} onRefresh={() => void refresh()} />
    {error && <p className="metaharness-error" role="alert">{error}</p>}
    {!detail && loading && <p className="metaharness-muted" role="status">Loading run…</p>}
    {detail && <>
      <RunApproval runId={runId} data={displayed} callBackendTool={callBackendTool} disabled={actionBusy} onDecision={decide} />
      <nav className="metaharness-run-tabs" aria-label="Run detail views">{tabs.map((tab) => <button type="button" role="tab" aria-selected={activeTab === tab} key={tab} onClick={() => setActiveTab(tab)}>{tab}</button>)}</nav>
      <div role="tabpanel" aria-label={activeTab}>
        {activeTab === 'Overview' && <RunSummary data={displayed} />}
        {activeTab === 'Plan' && <PlanView data={displayed} />}
        {activeTab === 'Steps' && <StepsView data={displayed} />}
        {activeTab === 'Checks' && <ChecksView data={{ ...displayed, checks: displayed.checks ?? parseArtifact(artifact['checks.json']) }} />}
        {activeTab === 'Review' && <ReviewView data={{ ...displayed, review: displayed.review ?? parseArtifact(artifact['review.json']), reviewer_raw: displayed.reviewer_raw ?? object(artifact['reviewer.raw.md']).content, revision: displayed.revision ?? parseArtifact(artifact['revision.json']) }} />}
        {activeTab === 'Diff' && <DiffView data={{ ...displayed, candidate: { ...object(displayed.candidate), changed_files: object(displayed.candidate).changed_files ?? artifactFileList(artifact['changed-files.txt']), diff_tail: object(artifact['diff.patch']).content ?? object(displayed.candidate).diff_tail, diff_truncated: object(artifact['diff.patch']).truncated ?? diffIsTruncated(object(displayed.candidate).diff_tail) } }} workspacePath={propsWorkspace(workspacePath)} openFile={openFile} />}
        {activeTab === 'Usage' && <UsageView data={displayed} />}
        {activeTab === 'Logs' && <ProgressView runId={runId} status={typeof displayed.status === 'string' ? displayed.status : undefined} callBackendTool={callBackendTool} intervalMs={pollIntervalMs} />}
        {activeTab === 'Diagnostics' && <DiagnosticsView data={displayed} artifact={object(artifact['diagnostics.json'])} />}
        {activeTab === 'Results' && <ResultsView data={displayed} />}
      </div>
    </>}
  </section>;
}

function propsWorkspace(path?: string): string | undefined { return typeof path === 'string' ? path : undefined; }
function parseArtifact(value: unknown): unknown {
  const content = object(value).content;
  if (typeof content !== 'string') return undefined;
  try { return JSON.parse(content); } catch { return content; }
}
function diffIsTruncated(value: unknown): boolean { return typeof value === 'string' && value.length >= 65536; }
function artifactFileList(value: unknown): string[] | undefined {
  const content = object(value).content;
  return typeof content === 'string' ? content.split(/\r?\n/).map((line) => line.trim()).filter(Boolean) : undefined;
}
