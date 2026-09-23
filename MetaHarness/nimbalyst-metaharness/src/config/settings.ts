export interface MetaHarnessSettingsData {
  executable: string;
  configPath: string;
  port: number;
  autoStart: boolean;
  pollIntervalMs: number;
}

export const SETTINGS_KEY = 'settings';

export const DEFAULT_SETTINGS: MetaHarnessSettingsData = {
  executable: 'metaharness',
  configPath: '',
  port: 8765,
  autoStart: true,
  pollIntervalMs: 1000,
};

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
