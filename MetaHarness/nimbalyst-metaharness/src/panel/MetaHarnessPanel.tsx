import { useCallback, useState } from 'react';
import type { PanelHostProps } from '@nimbalyst/extension-sdk';
import { MetaHarnessConfigForm } from '../config/MetaHarnessConfigForm';
import { DEFAULT_SETTINGS, SETTINGS_KEY, type MetaHarnessSettingsData } from '../config/settings';
import { callMetaHarnessBackend } from '../runtime/extensionRuntime';
import { RunsDashboard, type MetaHarnessPanelView } from './RunsDashboard';
import { ConfigurationView } from './configuration/ConfigurationView';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;

function settingsFromHost(host: PanelHostProps['host']): MetaHarnessSettingsData {
  const saved = host.storage.get<Partial<MetaHarnessSettingsData>>(SETTINGS_KEY) ?? {};
  return { ...DEFAULT_SETTINGS, ...saved };
}

export function MetaHarnessPanel({ host }: PanelHostProps) {
  const [settings, setSettings] = useState(() => settingsFromHost(host));
  const [view, setView] = useState<MetaHarnessPanelView>({ kind: 'dashboard' });
  const callBackendTool: BackendCall = useCallback(
    (toolName, args) => callMetaHarnessBackend(toolName, args ?? {}, host.workspacePath),
    [host.workspacePath],
  );

  function handleConfigured() {
    setSettings(settingsFromHost(host));
    setView({ kind: 'dashboard' });
  }

  function renderView() {
    if (!settings.configPath.trim()) {
      return <MetaHarnessConfigForm storage={host.storage} theme={host.theme} workspacePath={host.workspacePath} callBackendTool={callBackendTool} onConfigured={handleConfigured} />;
    }
    if (view.kind === 'configuration') {
      return <ConfigurationView
        settings={settings}
        storage={host.storage}
        theme={host.theme}
        workspacePath={host.workspacePath}
        callBackendTool={callBackendTool}
        openFile={(path) => host.openFile(path)}
        onBack={() => setView({ kind: 'dashboard' })}
        onSettingsSaved={setSettings}
      />;
    }
    return <RunsDashboard
      callBackendTool={callBackendTool}
      settings={settings}
      view={view}
      onViewChange={setView}
      workspacePath={host.workspacePath}
      openFile={(path) => host.openFile(path)}
      onOpenConfiguration={() => setView({ kind: 'configuration' })}
    />;
  }

  return (
    <main className="metaharness-panel" aria-label="MetaHarness runs">
      {renderView()}
    </main>
  );
}
