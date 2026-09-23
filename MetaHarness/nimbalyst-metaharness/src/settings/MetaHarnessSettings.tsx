import type { SettingsPanelProps } from '@nimbalyst/extension-sdk';

export function MetaHarnessSettings({ workspacePath }: SettingsPanelProps) {
  const workspace = workspacePath || 'No workspace open';

  return (
    <section className="metaharness-settings" aria-labelledby="metaharness-settings-title">
      <div className="metaharness-settings__eyebrow">Project settings</div>
      <h1 id="metaharness-settings-title">MetaHarness</h1>
      <p className="metaharness-settings__description">
        Configure the MetaHarness service for this project. Connection controls will
        be added in a later iteration.
      </p>
      <div className="metaharness-settings__workspace">
        <span>Workspace</span>
        <code title={workspace}>{workspace}</code>
      </div>
      <div className="metaharness-settings__notice" role="status">
        <strong>Not configured</strong>
        <span>No local MetaHarness service is configured yet.</span>
      </div>
    </section>
  );
}
