import { useCallback, useEffect, useRef, useState } from 'react';
import type { ExtensionStorage } from '@nimbalyst/extension-sdk';
import { MetaHarnessConfigForm } from '../../config/MetaHarnessConfigForm';
import { validateSettings, type MetaHarnessSettingsData } from '../../config/settings';
import { isBackendFailure } from '../../contract';
import { DoctorReport, EffectiveConfigView } from './EffectiveConfigView';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type Data = Record<string, unknown>;
type ServerStatus = { connected: boolean; serverOwned: boolean; api_version?: number };
type Busy = '' | 'test' | 'start' | 'restart' | 'doctor' | 'config';

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function unwrap(value: unknown): unknown {
  if (isBackendFailure(value)) {
    throw new Error(typeof value.error.message === 'string' ? value.error.message : 'MetaHarness backend call failed.');
  }
  return value;
}

function message(error: unknown): string {
  return error instanceof Error && error.message ? error.message : 'Unexpected MetaHarness error.';
}

/** Only an absolute path strictly inside the workspace is handed to host.openFile. */
function insideWorkspace(path: string, workspacePath?: string): boolean {
  if (!workspacePath || !path.startsWith('/') || path.split('/').includes('..')) return false;
  const root = workspacePath.replace(/\/+$/, '');
  return path.startsWith(`${root}/`);
}

export function ConfigurationView({
  settings,
  storage,
  theme,
  workspacePath,
  callBackendTool,
  openFile,
  onBack,
  onSettingsSaved,
}: {
  settings: MetaHarnessSettingsData;
  storage: ExtensionStorage;
  theme: string;
  workspacePath?: string;
  callBackendTool: BackendCall;
  openFile?: (path: string) => void;
  onBack: () => void;
  onSettingsSaved: (settings: MetaHarnessSettingsData) => void;
}) {
  const [status, setStatus] = useState<ServerStatus>();
  const [statusError, setStatusError] = useState('');
  const [config, setConfig] = useState<Data>();
  const [configError, setConfigError] = useState('');
  const [doctor, setDoctor] = useState<Data>();
  const [doctorError, setDoctorError] = useState('');
  const [busy, setBusy] = useState<Busy>('');
  const [dirty, setDirty] = useState(false);
  const [copied, setCopied] = useState(false);
  const generation = useRef(0);
  // Hosts may pass a new function each render; reloads follow settings, not identity.
  const backendRef = useRef(callBackendTool);
  backendRef.current = callBackendTool;

  const call = useCallback(async (name: string, args: Record<string, unknown> = {}) => (
    unwrap(await backendRef.current(name, args))
  ), []);

  const loadStatus = useCallback(async (): Promise<ServerStatus> => {
    const value = object(await call('metaharness.status', { settings }));
    // Ownership comes from the backend only; the UI never infers it.
    const next = {
      connected: value.connected === true,
      serverOwned: value.serverOwned === true,
      ...(typeof value.api_version === 'number' ? { api_version: value.api_version } : {}),
    };
    return next;
  }, [call, settings]);

  const loadConfig = useCallback(async () => object(await call('metaharness.get_config', { settings })), [call, settings]);

  /** Refetch status, then the effective config when a server answers. */
  const reload = useCallback(async (id = ++generation.current) => {
    setStatusError('');
    setConfigError('');
    let next: ServerStatus;
    try {
      next = await loadStatus();
    } catch (error) {
      if (id !== generation.current) return;
      setStatus(undefined);
      setConfig(undefined);
      setStatusError(message(error));
      return;
    }
    if (id !== generation.current) return;
    setStatus(next);
    if (!next.connected) {
      setConfig(undefined);
      return;
    }
    try {
      const loaded = await loadConfig();
      if (id === generation.current) setConfig(loaded);
    } catch (error) {
      if (id === generation.current) { setConfig(undefined); setConfigError(message(error)); }
    }
  }, [loadConfig, loadStatus]);

  useEffect(() => { void reload(); }, [reload]);

  async function run(kind: Busy, action: () => Promise<void>) {
    if (busy) return;
    setBusy(kind);
    try { await action(); } finally { setBusy(''); }
  }

  const testConnection = () => run('test', () => reload());

  const refreshConfig = () => run('config', async () => {
    setConfigError('');
    try { setConfig(await loadConfig()); } catch (error) { setConfigError(message(error)); }
  });

  const startServer = () => run('start', async () => {
    const id = ++generation.current;
    try {
      await call('metaharness.start', { settings });
    } catch (error) {
      setStatusError(message(error));
      return;
    }
    await reload(id);
  });

  const restartServer = () => run('restart', async () => {
    // Never stop a server this extension does not own.
    if (!status?.connected || !status.serverOwned) return;
    const id = ++generation.current;
    try {
      await call('metaharness.stop');
      await call('metaharness.start', { settings });
    } catch (error) {
      setStatusError(message(error));
      await reload(id);
      return;
    }
    await reload(id);
  });

  const runDoctor = () => run('doctor', async () => {
    setDoctorError('');
    try { setDoctor(object(await call('metaharness.doctor', { settings }))); } catch (error) {
      setDoctor(undefined);
      setDoctorError(message(error));
    }
  });

  async function copyPath() {
    try {
      await navigator.clipboard.writeText(settings.configPath);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  const handleDirty = useCallback((value: boolean) => setDirty(value), []);
  const connected = status?.connected === true;
  const owned = connected && status?.serverOwned === true;
  const external = connected && status?.serverOwned === false;
  const canStart = !connected && !validateSettings(settings);
  const statusLabel = status ? (connected ? 'Connected' : 'Disconnected') : statusError ? 'Unavailable' : 'Checking…';
  const ownership = !connected ? '—' : owned ? 'Nimbalyst' : 'External';

  return (
    <section className="metaharness-dashboard metaharness-config-view" aria-labelledby="metaharness-config-view-title">
      <button className="metaharness-link-button" type="button" onClick={onBack}>← Back to runs</button>
      <h1 id="metaharness-config-view-title">MetaHarness Configuration</h1>

      <section className="metaharness-config-section" aria-labelledby="metaharness-config-connection">
        <h2 id="metaharness-config-connection">Connection</h2>
        <p className="metaharness-config-section__label">Nimbalyst extension settings</p>
        <MetaHarnessConfigForm
          storage={storage}
          theme={theme}
          workspacePath={workspacePath}
          callBackendTool={callBackendTool}
          mode="embedded"
          showTitle={false}
          onSaved={onSettingsSaved}
          onDirtyChange={handleDirty}
        />
        <div className="metaharness-config-file">
          <span className="metaharness-muted">TOML file</span>
          <code>{settings.configPath || '—'}</code>
          <div className="metaharness-config-file__actions">
            {openFile && insideWorkspace(settings.configPath, workspacePath) && (
              <button className="metaharness-secondary-button" type="button" onClick={() => openFile(settings.configPath)}>OPEN CONFIG FILE</button>
            )}
            {settings.configPath && <button className="metaharness-secondary-button" type="button" onClick={() => void copyPath()}>{copied ? 'COPIED' : 'COPY PATH'}</button>}
          </div>
        </div>
      </section>

      <section className="metaharness-config-section" aria-labelledby="metaharness-config-server">
        <h2 id="metaharness-config-server">Server</h2>
        <div className="metaharness-server-status">
          <div className="metaharness-effective-config__row">
            <span>Status</span>
            <span className={`metaharness-connection ${connected ? 'is-connected' : ''}`}>
              <span className="metaharness-status-dot" aria-hidden="true" />{statusLabel}
            </span>
          </div>
          <div className="metaharness-effective-config__row"><span>Ownership</span><span>{ownership}</span></div>
          <div className="metaharness-effective-config__row"><span>API version</span><span>{status?.api_version ?? '—'}</span></div>
          <div className="metaharness-effective-config__row"><span>Config loaded</span><span>{config ? '✓' : '—'}</span></div>
        </div>
        {dirty && <p className="metaharness-settings__dirty" role="note">Status reflects the saved settings. Settings changed — connection not retested.</p>}
        {statusError && <p className="metaharness-error" role="alert">{statusError}</p>}
        {external && <div className="metaharness-config-note" role="note">
          <strong>External MetaHarness process</strong>
          <p>This MetaHarness server was not started by Nimbalyst. Restart it manually to reload changes from the TOML configuration.</p>
        </div>}
        <div className="metaharness-settings__actions">
          <button className="metaharness-secondary-button" type="button" onClick={() => void testConnection()} disabled={Boolean(busy)}>
            {busy === 'test' ? 'TESTING…' : 'TEST CONNECTION'}
          </button>
          <button className="metaharness-secondary-button" type="button" onClick={() => void runDoctor()} disabled={Boolean(busy)}>
            {busy === 'doctor' ? 'RUNNING DOCTOR…' : 'RUN DOCTOR'}
          </button>
          {owned && <button className="metaharness-button" type="button" onClick={() => void restartServer()} disabled={Boolean(busy)}>
            {busy === 'restart' ? 'RESTARTING…' : 'RESTART & RELOAD CONFIG'}
          </button>}
          {!connected && status && <button className="metaharness-button" type="button" onClick={() => void startServer()} disabled={Boolean(busy) || !canStart}>
            {busy === 'start' ? 'STARTING…' : 'START METAHARNESS'}
          </button>}
        </div>
        {doctorError && <p className="metaharness-error" role="alert">{doctorError}</p>}
        {doctor && <DoctorReport report={doctor} />}
      </section>

      <section className="metaharness-config-section" aria-labelledby="metaharness-config-effective">
        <h2 id="metaharness-config-effective">Effective configuration</h2>
        <p className="metaharness-config-section__label">Configuration currently loaded by MetaHarness</p>
        <p className="metaharness-muted">
          This shows the configuration currently loaded by the running MetaHarness server.
          If the TOML changed on disk, restart MetaHarness to reload it.
        </p>
        <p className="metaharness-muted">
          Agent routing, checks, approval, revision and publish defaults are defined by the MetaHarness TOML.
          Use New Run to override supported values for a single run.
        </p>
        {configError && <p className="metaharness-error" role="alert">{configError}</p>}
        {config
          ? <EffectiveConfigView config={config} />
          : <p className="metaharness-muted">{connected ? 'Effective configuration not loaded.' : 'Connect to MetaHarness to see its effective configuration.'}</p>}
        <div className="metaharness-settings__actions">
          <button className="metaharness-secondary-button" type="button" onClick={() => void refreshConfig()} disabled={Boolean(busy) || !connected}>
            {busy === 'config' ? 'REFRESHING…' : 'REFRESH EFFECTIVE CONFIG'}
          </button>
        </div>
      </section>
    </section>
  );
}
