import { useCallback, useEffect, useState } from 'react';
import type { RunSummary as RunSummaryData } from '../../types';
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

export function RunDetail({ runId, run, callBackendTool, onBack }: {
  runId: string;
  run?: RunSummaryData;
  callBackendTool?: BackendCall;
  onBack: () => void;
}) {
  const [detail, setDetail] = useState<Data>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    if (!callBackendTool) return;
    setLoading(true);
    setError('');
    try {
      const result = object(unwrap(await callBackendTool('metaharness.get_run', { runId })));
      setDetail(object(result.run ?? result));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
    }
  }, [callBackendTool, runId]);

  useEffect(() => { void refresh(); }, [refresh]);
  const displayed = { ...object(run), ...detail };
  return <section className="metaharness-dashboard metaharness-run-detail-page" aria-labelledby="metaharness-run-detail-title">
    <RunHeader runId={runId} data={displayed} loading={loading} onBack={onBack} onRefresh={() => void refresh()} />
    {error && <p className="metaharness-error" role="alert">{error}</p>}
    {!detail && loading && <p className="metaharness-muted" role="status">Loading run…</p>}
    {detail && <>
      <RunSummary data={displayed} />
      <PlanView data={displayed} />
      <StepsView data={displayed} />
    </>}
  </section>;
}
