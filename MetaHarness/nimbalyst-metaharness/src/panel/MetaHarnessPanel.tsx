import { useState } from 'react';
import type { PanelHostProps } from '@nimbalyst/extension-sdk';
import { MetaHarnessConfigForm } from '../config/MetaHarnessConfigForm';
import { DEFAULT_SETTINGS, SETTINGS_KEY, type MetaHarnessSettingsData } from '../config/settings';
import { callMetaHarnessBackend } from '../runtime/extensionRuntime';
import { RunsDashboard } from './RunsDashboard';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;

function settingsFromHost(host: PanelHostProps['host']): MetaHarnessSettingsData {
  const saved = host.storage.get<Partial<MetaHarnessSettingsData>>(SETTINGS_KEY) ?? {};
  return { ...DEFAULT_SETTINGS, ...saved };
}

export function MetaHarnessPanel({ host }: PanelHostProps) {
  const [settings, setSettings] = useState(() => settingsFromHost(host));
  const [view, setView] = useState<{ kind: 'dashboard' } | { kind: 'run'; runId: string } | { kind: 'new-run' }>({ kind: 'dashboard' });
  const callBackendTool: BackendCall = (toolName, args) => callMetaHarnessBackend(toolName, args ?? {}, host.workspacePath);

  function handleConfigured() {
    setSettings(settingsFromHost(host));
    setView({ kind: 'dashboard' });
  }

  return (
    <main className="metaharness-panel" aria-label="MetaHarness runs">
      {!settings.configPath.trim()
        ? <MetaHarnessConfigForm storage={host.storage} theme={host.theme} workspacePath={host.workspacePath} callBackendTool={callBackendTool} onConfigured={handleConfigured} />
        : <RunsDashboard
          callBackendTool={callBackendTool}
          settings={settings}
          view={view}
          onViewChange={setView}
          workspacePath={host.workspacePath}
          openFile={(path) => host.openFile(path)}
        />}
    </main>
  );
}
