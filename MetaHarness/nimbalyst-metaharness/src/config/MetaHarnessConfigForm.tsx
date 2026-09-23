import { useEffect, useState } from 'react';
import type { ExtensionStorage } from '@nimbalyst/extension-sdk';
import { isBackendFailure } from '../contract';
import { DEFAULT_SETTINGS, SETTINGS_KEY, validateSettings, type MetaHarnessSettingsData } from './settings';

type BackendCall = (toolName: string, args?: Record<string, unknown>, workspacePath?: string) => Promise<unknown>;

export interface MetaHarnessConfigFormProps {
  storage: ExtensionStorage;
  theme: string;
  workspacePath?: string;
  callBackendTool?: BackendCall;
  onConfigured?: () => void;
  /** Called after every successful persist, tested or not, with the saved settings. */
  onSaved?: (settings: MetaHarnessSettingsData) => void;
  /** Reports whether the inputs differ from the persisted settings. */
  onDirtyChange?: (dirty: boolean) => void;
  mode?: 'initial' | 'settings' | 'embedded';
  showTitle?: boolean;
}

function sameSettings(left: MetaHarnessSettingsData, right: MetaHarnessSettingsData): boolean {
  return (Object.keys(DEFAULT_SETTINGS) as Array<keyof MetaHarnessSettingsData>).every((key) => left[key] === right[key]);
}

function object(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? value as Record<string, unknown> : {};
}

function backendValue(value: unknown): unknown {
  if (isBackendFailure(value)) {
    const error = new Error(typeof value.error.message === 'string' ? value.error.message : 'MetaHarness backend call failed.');
    Object.assign(error, { code: value.error.code, details: value.error.details });
    throw error;
  }
  return value;
}

function friendlyError(error: unknown, port: number): string {
  const detail = error instanceof Error ? error.message : '';
  const normalized = detail.toLowerCase();
  const structured = object(error);
  const code = typeof structured.code === 'string' ? structured.code.toUpperCase() : '';
  const details = object(structured.details);
  if (code.includes('PERMISSION') || code.includes('CONSENT') || code.includes('DENIED')
    || normalized.includes('permission') || normalized.includes('denied') || normalized.includes('consent') || normalized.includes('disabled')) {
    return 'MetaHarness backend permission is required to control local runs.';
  }
  if ((code.includes('MODULE') && (code.includes('UNAVAILABLE') || code.includes('NOT_FOUND') || code.includes('DISABLED')))
    || ((normalized.includes('module') || normalized.includes('backend tool') || normalized.includes('backend module'))
      && (normalized.includes('unavailable') || normalized.includes('not running') || normalized.includes('not found') || normalized.includes('not active') || normalized.includes('not enabled')))) {
    return 'The MetaHarness Nimbalyst backend module is unavailable. Reload the extension and retry.';
  }
  if (code === 'PORT_IN_USE' || code === 'SERVICE_MISMATCH' || normalized.includes('address already in use') || normalized.includes('eaddrinuse') || normalized.includes('port is occupied') || normalized.includes('non-metaharness service')) {
    return `Port ${port} is already in use by a service that is not MetaHarness.`;
  }
  if (code.includes('EXECUTABLE') && (code.includes('NOT_FOUND') || code.includes('MISSING'))
    || (normalized.includes('executable') && (normalized.includes('not found') || normalized.includes('enoent')))
    || (code === 'SPAWN_FAILED' && details.causeCode === 'ENOENT')
    || (normalized.includes('spawn ') && normalized.includes('enoent'))) {
    return 'MetaHarness executable was not found. Check the configured absolute path.';
  }
  if ((code.includes('CONFIG') && (code.includes('NOT_FOUND') || code.includes('MISSING')))
    || (normalized.includes('config') && (normalized.includes('not found') || normalized.includes('enoent') || normalized.includes('does not exist') || normalized.includes('no such file')))) {
    return 'MetaHarness configuration file was not found.';
  }
  return detail || 'MetaHarness backend call failed.';
}

export function MetaHarnessConfigForm({
  storage,
  theme,
  workspacePath,
  callBackendTool,
  onConfigured,
  onSaved,
  onDirtyChange,
  mode = 'initial',
  showTitle = mode !== 'embedded',
}: MetaHarnessConfigFormProps) {
  const [savedSettings, setSavedSettings] = useState<MetaHarnessSettingsData>(() => ({
    ...DEFAULT_SETTINGS,
    ...(storage.get<Partial<MetaHarnessSettingsData>>(SETTINGS_KEY) ?? {}),
  }));
  const [settings, setSettings] = useState<MetaHarnessSettingsData>(savedSettings);
  const [validationError, setValidationError] = useState('');
  const [actionError, setActionError] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const dirty = !sameSettings(settings, savedSettings);

  useEffect(() => { onDirtyChange?.(dirty); }, [dirty, onDirtyChange]);

  function update<K extends keyof MetaHarnessSettingsData>(key: K, value: MetaHarnessSettingsData[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
    setValidationError('');
    setActionError('');
    setStatus('');
  }

  async function save(test: boolean) {
    setValidationError('');
    setActionError('');
    setStatus('');
    const invalid = validateSettings(settings);
    if (invalid) { setValidationError(invalid); return; }
    setBusy(true);
    let persisted = false;
    try {
      await storage.set(SETTINGS_KEY, settings);
      persisted = true;
      setSavedSettings(settings);
      if (!test) {
        setStatus('Saved');
        onConfigured?.();
        return;
      }
      if (!callBackendTool) throw new Error('backend unavailable');
      let result = object(backendValue(await callBackendTool('metaharness.status', { settings }, workspacePath)));
      if (result.connected !== true && settings.autoStart) {
        await callBackendTool('metaharness.start', { settings }, workspacePath).then(backendValue);
        result = object(backendValue(await callBackendTool('metaharness.status', { settings }, workspacePath)));
      }
      if (result.connected !== true) {
        setStatus('Disconnected');
        setActionError('MetaHarness is not connected. Check the configured executable, file, and port.');
        return;
      }
      setStatus('Connected');
      onConfigured?.();
    } catch (error) {
      setActionError(friendlyError(error, settings.port));
    } finally {
      setBusy(false);
      if (persisted) onSaved?.(settings);
    }
  }

  return (
    <section
      className={`metaharness-settings metaharness-settings--${mode}`}
      {...(showTitle ? { 'aria-labelledby': 'metaharness-config-title' } : { 'aria-label': 'MetaHarness connection settings' })}
      data-theme={theme}
    >
      {showTitle && <>
        <div className="metaharness-settings__eyebrow">Project settings</div>
        <h1 id="metaharness-config-title">Configure MetaHarness</h1>
      </>}
      <div className="metaharness-settings__form">
        <label>MetaHarness executable
          <input value={settings.executable} onChange={(event) => update('executable', event.target.value)} />
        </label>
        <label>Configuration file
          <input value={settings.configPath} placeholder="Absolute path to a MetaHarness TOML file" onChange={(event) => update('configPath', event.target.value)} />
        </label>
        <label>Port
          <input type="number" min="1" max="65535" value={settings.port} onChange={(event) => update('port', Number(event.target.value))} />
        </label>
        <label className="metaharness-settings__checkbox">
          <input type="checkbox" checked={settings.autoStart} onChange={(event) => update('autoStart', event.target.checked)} />
          Start automatically
        </label>
        <label>Polling interval
          <span className="metaharness-settings__input-suffix">
            <input type="number" min="500" max="30000" step="100" value={settings.pollIntervalMs} onChange={(event) => update('pollIntervalMs', Number(event.target.value))} /> ms
          </span>
        </label>
      </div>
      {validationError && <p className="metaharness-settings__error" role="alert">{validationError}</p>}
      {actionError && <p className="metaharness-settings__error" role="alert">{actionError}</p>}
      {dirty && !busy && <p className="metaharness-settings__dirty" role="status">Unsaved changes — connection not retested</p>}
      {status && !dirty && <p className={status === 'Connected' ? 'metaharness-settings__connected' : status === 'Saved' ? 'metaharness-settings__saved' : 'metaharness-settings__disconnected'} role="status">{status}</p>}
      <div className="metaharness-settings__actions">
        <button className="metaharness-button" type="button" onClick={() => void save(false)} disabled={busy}>SAVE</button>
        <button className="metaharness-button metaharness-button--secondary" type="button" onClick={() => void save(true)} disabled={busy}>
          {busy ? 'TESTING…' : 'SAVE & TEST CONNECTION'}
        </button>
      </div>
      {workspacePath && <p className="metaharness-settings__workspace">Workspace: <code>{workspacePath}</code></p>}
    </section>
  );
}
