import { useEffect, useState } from 'react';
import type { SettingsPanelProps } from '@nimbalyst/extension-sdk';

export interface MetaHarnessSettingsData {
  executable: string;
  configPath: string;
  port: number;
  autoStart: boolean;
  pollIntervalMs: number;
}

export const DEFAULT_SETTINGS: MetaHarnessSettingsData = {
  executable: 'metaharness',
  configPath: '',
  port: 8765,
  autoStart: true,
  pollIntervalMs: 1000,
};

const STORAGE_KEY = 'settings';
type CheckStatus = 'pass' | 'warn' | 'fail';
type Check = { name: string; status: CheckStatus; message: string };

export function validateSettings(settings: MetaHarnessSettingsData): string | undefined {
  if (!settings.executable.trim()) return 'Executable must not be empty.';
  if (!settings.configPath.trim()) return 'Configuration path must not be empty.';
  if (!Number.isInteger(settings.port) || settings.port < 1 || settings.port > 65535) {
    return 'Port must be an integer between 1 and 65535.';
  }
  if (!Number.isInteger(settings.pollIntervalMs)
    || settings.pollIntervalMs < 500 || settings.pollIntervalMs > 30000) {
    return 'Polling interval must be between 500 and 30000 ms.';
  }
  return undefined;
}

function object(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? value as Record<string, unknown> : {};
}

function backendValue(value: unknown): unknown {
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

function statusLabel(status: CheckStatus): string {
  return status === 'pass' ? 'PASS' : status === 'warn' ? 'WARN' : 'FAIL';
}

function doctorChecks(value: unknown): Check[] {
  const response = object(value);
  const raw = response.checks;
  const rows = Array.isArray(raw)
    ? raw.map((check, index) => [String(index + 1), check] as const)
    : Object.entries(object(raw));
  return rows.map(([key, value]) => {
    const check = object(value);
    const rawStatus = String(check.status ?? check.result ?? check.state ?? '').toLowerCase();
    const status: CheckStatus = ['pass', 'passed', 'ok', 'success'].includes(rawStatus)
      ? 'pass'
      : ['warn', 'warning'].includes(rawStatus)
        ? 'warn'
        : ['fail', 'failed', 'error'].includes(rawStatus)
          ? 'fail'
          : check.ok === true ? 'pass' : check.ok === false ? 'fail' : 'warn';
    return {
      name: String(check.name ?? check.check ?? key),
      status,
      message: String(check.message ?? check.detail ?? check.description ?? rawStatus ?? 'No details provided.'),
    };
  });
}

function display(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
}

export function MetaHarnessSettings({ storage, workspacePath, callBackendTool }: SettingsPanelProps) {
  const [settings, setSettings] = useState<MetaHarnessSettingsData>(() => ({
    ...DEFAULT_SETTINGS,
    ...(storage.get<Partial<MetaHarnessSettingsData>>(STORAGE_KEY) ?? {}),
  }));
  const [validationError, setValidationError] = useState('');
  const [actionError, setActionError] = useState('');
  const [connection, setConnection] = useState<Record<string, unknown> | undefined>();
  const [effectiveConfig, setEffectiveConfig] = useState<Record<string, unknown> | undefined>();
  const [doctor, setDoctor] = useState<Check[] | undefined>();
  const [busy, setBusy] = useState<'connection' | 'doctor' | ''>('');

  useEffect(() => {
    void storage.set(STORAGE_KEY, settings)
      .catch((error) => setActionError(`Could not save settings: ${errorMessage(error)}`));
  }, [settings, storage]);

  useEffect(() => {
    let active = true;
    if (!callBackendTool) return () => { active = false; };
    const args = { settings };
    void callBackendTool('metaharness.status', args)
      .then(backendValue)
      .then(async (result) => {
        if (!active) return;
        const status = object(result);
        setConnection(status);
        const suggested = status.recommendedConfigPath;
        let selectedSettings = settings;
        if (!settings.configPath && typeof suggested === 'string' && suggested) {
          selectedSettings = { ...settings, configPath: suggested };
          setSettings((current) => ({ ...current, configPath: suggested }));
        }
        let connected = status.connected === true;
        if (!connected && selectedSettings.autoStart && selectedSettings.configPath.trim()) {
          const started = object(backendValue(await callBackendTool('metaharness.start', { settings: selectedSettings })));
          if (active) setConnection(started);
          connected = started.connected === true;
        }
        if (connected) {
          const config = backendValue(await callBackendTool('metaharness.get_config', { settings: selectedSettings }));
          if (active) setEffectiveConfig(object(config));
        }
      })
      .catch((error) => { if (active) setActionError(errorMessage(error)); });
    return () => { active = false; };
  // Load the initial connection state once; settings are explicitly sent on all calls.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!callBackendTool) return;
    let active = true;
    const timer = window.setInterval(() => {
      void callBackendTool('metaharness.status', { settings })
        .then(backendValue)
        .then((result) => { if (active) setConnection(object(result)); })
        .catch((error) => { if (active) setActionError(errorMessage(error)); });
    }, settings.pollIntervalMs);
    return () => { active = false; window.clearInterval(timer); };
  }, [settings, callBackendTool]);

  function update<K extends keyof MetaHarnessSettingsData>(key: K, value: MetaHarnessSettingsData[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
    setValidationError('');
  }

  async function testConnection() {
    setActionError('');
    setValidationError('');
    const invalid = validateSettings(settings);
    if (invalid) { setValidationError(invalid); return; }
    if (!callBackendTool) {
      setActionError('MetaHarness backend is unavailable. Enable the extension backend and try again.');
      return;
    }
    setBusy('connection');
    try {
      const result = object(backendValue(await callBackendTool('metaharness.status', { settings })));
      setConnection(result);
      if (!result.connected) {
        setEffectiveConfig(undefined);
        setActionError('MetaHarness is not responding on the configured port.');
      } else if (settings.autoStart) {
        const config = backendValue(await callBackendTool('metaharness.get_config', { settings }));
        setEffectiveConfig(object(config));
      }
    } catch (error) {
      setActionError(`Could not test the MetaHarness connection: ${errorMessage(error)}`);
    } finally { setBusy(''); }
  }

  async function runDoctor() {
    setActionError('');
    setValidationError('');
    const invalid = validateSettings(settings);
    if (invalid) { setValidationError(invalid); return; }
    if (!callBackendTool) {
      setActionError('MetaHarness backend is unavailable. Enable the extension backend and try again.');
      return;
    }
    setBusy('doctor');
    setDoctor(undefined);
    try {
      const result = backendValue(await callBackendTool('metaharness.doctor', { settings }));
      const checks = doctorChecks(result);
      if (!checks.length) throw new Error('Doctor returned no individual checks.');
      setDoctor(checks);
    } catch (error) {
      setActionError(`MetaHarness doctor failed: ${errorMessage(error)}`);
    } finally { setBusy(''); }
  }

  const config = object(effectiveConfig);
  const repository = object(config.repository);
  const planning = object(config.planning);
  const approval = object(config.approval);
  const ui = object(config.ui);
  const publish = object(config.publish);
  const checks = config.checks;
  return (
    <section className="metaharness-settings" aria-labelledby="metaharness-settings-title">
      <div className="metaharness-settings__eyebrow">Project settings</div>
      <h1 id="metaharness-settings-title">MetaHarness</h1>
      <div className="metaharness-settings__form">
        <label>Executable
          <input value={settings.executable} onChange={(event) => update('executable', event.target.value)} />
        </label>
        <label>Configuration
          <input value={settings.configPath} placeholder="Absolute path to a MetaHarness TOML file" onChange={(event) => update('configPath', event.target.value)} />
        </label>
        <label>Port
          <input type="number" min="1" max="65535" value={settings.port} onChange={(event) => update('port', Number(event.target.value))} />
        </label>
        <label className="metaharness-settings__checkbox">
          <input type="checkbox" checked={settings.autoStart} onChange={(event) => update('autoStart', event.target.checked)} />
          Start MetaHarness automatically
        </label>
        <label>Polling interval
          <span className="metaharness-settings__input-suffix">
            <input type="number" min="500" max="30000" step="100" value={settings.pollIntervalMs} onChange={(event) => update('pollIntervalMs', Number(event.target.value))} /> ms
          </span>
        </label>
      </div>
      {validationError && <p className="metaharness-settings__error" role="alert">{validationError}</p>}
      <div className="metaharness-settings__actions">
        <button className="metaharness-button" onClick={() => void testConnection()} disabled={Boolean(busy)}>
          {busy === 'connection' ? 'TESTING…' : 'TEST CONNECTION'}
        </button>
        <button className="metaharness-button metaharness-button--secondary" onClick={() => void runDoctor()} disabled={Boolean(busy)}>
          {busy === 'doctor' ? 'RUNNING…' : 'RUN DOCTOR'}
        </button>
      </div>
      {actionError && <p className="metaharness-settings__error" role="alert">{actionError}</p>}
      <section className="metaharness-settings__section" aria-labelledby="metaharness-status-title">
        <h2 id="metaharness-status-title">Status</h2>
        {connection ? <p className={connection.connected ? 'metaharness-settings__connected' : 'metaharness-settings__disconnected'}>
          <span aria-hidden="true">●</span> {connection.connected ? 'Connected' : 'Disconnected'}
        </p> : <p className="metaharness-settings__muted">Not checked</p>}
        {connection?.connected === true && <>
          <p>MetaHarness API v{display(connection.api_version ?? 1)}</p>
          {effectiveConfig && <p>Repository: {display(repository.repo)}<br />Base: {display(repository.base_ref)}</p>}
        </>}
      </section>
      {doctor && <section className="metaharness-settings__section" aria-labelledby="metaharness-doctor-title">
        <h2 id="metaharness-doctor-title">Doctor checks</h2>
        <ul className="metaharness-settings__checks">{doctor.map((check, index) =>
          <li key={`${check.name}-${index}`} data-status={check.status}>
            <strong>{statusLabel(check.status)}</strong><span>{check.name}</span><span>{check.message}</span>
          </li>)}</ul>
      </section>}
      <section className="metaharness-settings__section" aria-labelledby="metaharness-effective-title">
        <h2 id="metaharness-effective-title">Effective MetaHarness configuration</h2>
        {!effectiveConfig && <p className="metaharness-settings__muted">Connect to MetaHarness to load its source-controlled configuration.</p>}
        {effectiveConfig && <dl className="metaharness-settings__details">
          <dt>Repository</dt><dd>{display(repository.repo ?? config.repo)}</dd>
          <dt>Base ref</dt><dd>{display(repository.base_ref ?? config.base_ref)}</dd>
          <dt>Planning protocol</dt><dd>{display(planning.protocol ?? config.planning_protocol)}</dd>
          <dt>Approval required</dt><dd>{display(approval.require_plan_approval ?? config.approval_required)}</dd>
          <dt>Publish enabled</dt><dd>{display(config.publish_enabled ?? publish.enabled)}</dd>
          <dt>Publish mode</dt><dd>{display(config.publish_mode ?? publish.mode)}</dd>
          <dt>Max active runs</dt><dd>{display(ui.max_active_runs ?? config.max_active_runs)}</dd>
          <dt>Configured checks</dt><dd>{Array.isArray(checks) ? checks.map((check) => display(object(check).id ?? check)).join(', ') : display(checks)}</dd>
        </dl>}
      </section>
      {workspacePath && <p className="metaharness-settings__workspace">Workspace: <code>{workspacePath}</code></p>}
    </section>
  );
}
