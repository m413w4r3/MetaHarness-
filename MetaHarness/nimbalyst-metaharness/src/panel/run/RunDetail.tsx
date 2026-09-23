import { useCallback, useEffect, useRef, useState } from 'react';
import type { RunSummary as RunSummaryData } from '../../types';
import { RunApproval } from './RunApproval';
import { RunHeader } from './RunHeader';
import { RunSummary } from './RunSummary';
import { PlanView } from './PlanView';
import { StepsView } from './StepsView';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type Data = Record<string, unknown>;

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

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

export function RunDetail({ runId, run, callBackendTool, onBack, pollIntervalMs = 1000 }: {
  runId: string;
  run?: RunSummaryData;
  callBackendTool?: BackendCall;
  onBack: () => void;
  pollIntervalMs?: number;
}) {
  const [detail, setDetail] = useState<Data>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [actionBusy, setActionBusy] = useState(false);
  const actionInFlight = useRef(false);
  const pollInFlight = useRef(false);
  const refreshSequence = useRef(0);
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
  return <section className="metaharness-dashboard metaharness-run-detail-page" aria-labelledby="metaharness-run-detail-title">
    <RunHeader runId={runId} data={displayed} loading={loading} onBack={onBack} onRefresh={() => void refresh()} />
    {error && <p className="metaharness-error" role="alert">{error}</p>}
    {!detail && loading && <p className="metaharness-muted" role="status">Loading run…</p>}
    {detail && <>
      <RunApproval runId={runId} data={displayed} callBackendTool={callBackendTool} disabled={actionBusy} onDecision={decide} />
      <RunSummary data={displayed} />
      <PlanView data={displayed} />
      <StepsView data={displayed} />
    </>}
  </section>;
}
