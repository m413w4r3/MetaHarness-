import { useState } from 'react';
import type { PanelHostProps } from '@nimbalyst/extension-sdk';
import { DEFAULT_SETTINGS, type MetaHarnessSettingsData } from '../settings/MetaHarnessSettings';
import { RunsDashboard } from './RunsDashboard';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type ExtendedPanelHost = PanelHostProps['host'] & { callBackendTool?: BackendCall };

function settingsFromHost(host: ExtendedPanelHost): MetaHarnessSettingsData {
  const saved = host.storage.get<Partial<MetaHarnessSettingsData>>('settings') ?? {};
  return { ...DEFAULT_SETTINGS, ...saved };
}

export function MetaHarnessPanel({ host }: PanelHostProps) {
  const backendCall = (host as ExtendedPanelHost).callBackendTool;
  const [settings] = useState(() => settingsFromHost(host as ExtendedPanelHost));
  const [view, setView] = useState<{ kind: 'dashboard' } | { kind: 'run'; runId: string } | { kind: 'new-run' }>({ kind: 'dashboard' });

  return (
    <main className="metaharness-panel" aria-label="MetaHarness runs">
      <RunsDashboard
        callBackendTool={backendCall}
        settings={settings}
        view={view}
        onViewChange={setView}
        onOpenSettings={() => host.openSettings()}
      />
    </main>
  );
}
