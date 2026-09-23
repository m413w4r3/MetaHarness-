import { useCallback, useEffect, useRef, useState } from 'react';
import type { MetaHarnessSettingsData } from '../settings/MetaHarnessSettings';
import type { RunSummary } from '../types';
import { RunCard } from './RunCard';
import { NewRunForm } from './NewRunForm';
import { StatusBadge } from './StatusBadge';
import { classifyRunStatus, type RunCategory } from './runStatus';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type PanelView = { kind: 'dashboard' } | { kind: 'run'; runId: string } | { kind: 'new-run' };
type ConnectionStatus = { configured?: boolean; connected?: boolean; recommendedConfigPath?: string };

function object(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? value as Record<string, unknown> : {};
}

function unwrap(value: unknown): unknown {
  const result = object(value);
  if (result.ok === false) {
    const error = object(result.error);
    throw new Error(typeof error.message === 'string' ? error.message : 'MetaHarness backend call failed.');
  }
  return value;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Unexpected MetaHarness error.';
}

function failureText(value: unknown): string {
  if (typeof value === 'string') return value;
  const reason = object(value).reason;
  return typeof reason === 'string' ? reason : 'Failure reported';
}

const GROUPS: Array<{ category: RunCategory; title: string }> = [
  { category: 'active', title: 'Active' },
  { category: 'awaiting-action', title: 'Awaiting action' },
  { category: 'completed', title: 'Completed' },
  { category: 'failed', title: 'Failed' },
  { category: 'other', title: 'Other' },
];

export function RunsDashboard({
  callBackendTool,
  settings,
  view,
  onViewChange,
  onOpenSettings,
}: {
  callBackendTool?: BackendCall;
  settings: MetaHarnessSettingsData;
  view: PanelView;
  onViewChange: (view: PanelView) => void;
  onOpenSettings: () => void;
}) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [connection, setConnection] = useState<ConnectionStatus>();
  const [effectiveSettings, setEffectiveSettings] = useState(settings);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (!callBackendTool || inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError('');
    try {
      let selectedSettings = effectiveSettings;
      let status = object(unwrap(await callBackendTool('metaharness.status', { settings: selectedSettings }))) as ConnectionStatus;
      if (!selectedSettings.configPath.trim() && typeof status.recommendedConfigPath === 'string' && status.recommendedConfigPath) {
        selectedSettings = { ...selectedSettings, configPath: status.recommendedConfigPath };
        setEffectiveSettings(selectedSettings);
        status = object(unwrap(await callBackendTool('metaharness.status', { settings: selectedSettings }))) as ConnectionStatus;
      }

      if (!status.configured) {
        setConnection(status);
        setRuns([]);
        return;
      }
      if (!status.connected && selectedSettings.autoStart) {
        unwrap(await callBackendTool('metaharness.start', { settings: selectedSettings }));
        status = object(unwrap(await callBackendTool('metaharness.status', { settings: selectedSettings }))) as ConnectionStatus;
      }
      setConnection(status);
      if (!status.connected) {
        setRuns([]);
        return;
      }
      const result = unwrap(await callBackendTool('metaharness.list_runs'));
      if (!Array.isArray(result)) throw new Error('MetaHarness returned an invalid runs list.');
      setRuns(result as RunSummary[]);
    } catch (caught) {
      setError(message(caught));
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }, [callBackendTool, effectiveSettings]);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    if (!callBackendTool) return undefined;
    const timer = window.setInterval(() => { void refresh(); }, settings.pollIntervalMs);
    return () => window.clearInterval(timer);
  }, [callBackendTool, refresh, settings.pollIntervalMs]);

  const selectedRun = view.kind === 'run' ? runs.find((run) => run.run_id === view.runId) : undefined;
  if (view.kind === 'new-run') {
    return <NewRunForm callBackendTool={callBackendTool} settings={effectiveSettings} onBack={() => onViewChange({ kind: 'dashboard' })} onCreated={(runId) => onViewChange({ kind: 'run', runId })} />;
  }
  if (view.kind === 'run') {
    return <RunDetail runId={view.runId} run={selectedRun} callBackendTool={callBackendTool} onBack={() => onViewChange({ kind: 'dashboard' })} />;
  }

  if (!callBackendTool) {
    return <ConfigurationScreen onOpenSettings={onOpenSettings} />;
  }
  if (!loading && connection?.configured === false) {
    return <ConfigurationScreen onOpenSettings={onOpenSettings} />;
  }

  return (
    <section className="metaharness-dashboard" aria-labelledby="metaharness-dashboard-title">
      <header className="metaharness-dashboard__header">
        <h1 id="metaharness-dashboard-title">MetaHarness</h1>
        <span className={`metaharness-connection ${connection?.connected ? 'is-connected' : ''}`}>
          <span className="metaharness-status-dot" aria-hidden="true" />
          {loading && !connection ? 'Connecting…' : connection?.connected ? 'Connected' : 'Disconnected'}
        </span>
      </header>
      <div className="metaharness-dashboard__actions">
        <button className="metaharness-button" type="button" onClick={() => onViewChange({ kind: 'new-run' })}>＋ New Run</button>
        <button className="metaharness-secondary-button" type="button" onClick={() => void refresh()} disabled={loading}>Refresh</button>
      </div>
      {error && <p className="metaharness-error" role="alert">{error}</p>}
      {loading && runs.length === 0 && <p className="metaharness-muted" role="status">Loading runs…</p>}
      {!loading && !connection?.connected && !error && <p className="metaharness-muted">MetaHarness is not connected. Check your configuration or try again.</p>}
      {!loading && connection?.connected && runs.length === 0 && <p className="metaharness-muted">No runs in this workspace yet.</p>}
      {GROUPS.map(({ category, title }) => {
        const groupRuns = runs.filter((run) => classifyRunStatus(typeof run.status === 'string' ? run.status : 'unknown') === category);
        if (!groupRuns.length) return null;
        return <section className="metaharness-run-group" key={category} aria-label={title}>
          <h2>{title}</h2>
          <div className="metaharness-run-group__cards">
            {groupRuns.map((run, index) => (
              <RunCard key={run.run_id ?? `${category}-${index}`} run={run} onClick={() => {
                if (typeof run.run_id === 'string') onViewChange({ kind: 'run', runId: run.run_id });
              }} />
            ))}
          </div>
        </section>;
      })}
    </section>
  );
}

function ConfigurationScreen({ onOpenSettings }: { onOpenSettings: () => void }) {
  return <section className="metaharness-dashboard metaharness-configuration" aria-labelledby="metaharness-config-title">
    <h1 id="metaharness-config-title">Configure MetaHarness</h1>
    <p>Set a MetaHarness configuration file to see runs for this workspace.</p>
    <button className="metaharness-button" type="button" onClick={onOpenSettings}>Open Settings</button>
  </section>;
}

function RunDetail({ runId, run, callBackendTool, onBack }: { runId: string; run?: RunSummary; callBackendTool?: BackendCall; onBack: () => void }) {
  const [detail, setDetail] = useState<Record<string, unknown>>();
  const [detailError, setDetailError] = useState('');
  useEffect(() => {
    let active = true;
    if (!callBackendTool) return () => { active = false; };
    void callBackendTool('metaharness.get_run', { runId }).then((value) => {
      const result = object(unwrap(value));
      const loaded = object(result.run ?? result);
      if (active) setDetail(loaded);
    }).catch((error) => { if (active) setDetailError(message(error)); });
    return () => { active = false; };
  }, [callBackendTool, runId]);
  const displayed = detail ? { ...run, ...detail } : run;
  const status = typeof displayed?.status === 'string' ? displayed.status : 'unknown';
  const hasFailure = displayed?.failure !== undefined && displayed.failure !== null && displayed.failure !== '';
  return <section className="metaharness-dashboard" aria-labelledby="metaharness-run-detail-title">
    <button className="metaharness-link-button" type="button" onClick={onBack}>← All runs</button>
    <h1 id="metaharness-run-detail-title">{runId}</h1>
    {displayed ? <dl className="metaharness-run-detail">
      <div><dt>Status</dt><dd><StatusBadge status={status} /></dd></div>
      {displayed.plan_title && <div><dt>Plan</dt><dd>{String(displayed.plan_title)}</dd></div>}
      {displayed.updated_at && <div><dt>Updated</dt><dd>{String(displayed.updated_at)}</dd></div>}
      {displayed.commit_sha && <div><dt>Commit</dt><dd><code>{String(displayed.commit_sha)}</code></dd></div>}
      {hasFailure && <div><dt>Failure</dt><dd>{failureText(displayed.failure)}</dd></div>}
    </dl> : detailError ? <p className="metaharness-error" role="alert">{detailError}</p> : <p className="metaharness-muted" role="status">Loading run…</p>}
  </section>;
}
