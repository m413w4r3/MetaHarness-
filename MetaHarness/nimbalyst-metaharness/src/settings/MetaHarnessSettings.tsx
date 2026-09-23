import type { SettingsPanelProps } from '@nimbalyst/extension-sdk';
import { MetaHarnessConfigForm } from '../config/MetaHarnessConfigForm';

export function MetaHarnessSettings({ storage, theme, workspacePath, callBackendTool }: SettingsPanelProps) {
  return (
    <MetaHarnessConfigForm
      storage={storage}
      theme={theme}
      workspacePath={workspacePath}
      callBackendTool={callBackendTool
        ? (name, args) => callBackendTool(name, args)
        : undefined}
    />
  );
}

export { DEFAULT_SETTINGS, validateSettings, type MetaHarnessSettingsData } from '../config/settings';
