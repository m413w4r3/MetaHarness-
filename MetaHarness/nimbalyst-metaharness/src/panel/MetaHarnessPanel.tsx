import type { PanelHostProps } from '@nimbalyst/extension-sdk';

export function MetaHarnessPanel({ host }: PanelHostProps) {
  const workspace = host.workspacePath || 'No workspace open';

  return (
    <main className="metaharness-panel" aria-labelledby="metaharness-title">
      <div className="metaharness-panel__content">
        <div className="metaharness-panel__eyebrow">Nimbalyst extension</div>
        <h1 id="metaharness-title">MetaHarness</h1>
        <p className="metaharness-panel__description">
          Control and inspect MetaHarness runs from this workspace.
        </p>

        <dl className="metaharness-panel__details">
          <div>
            <dt>Workspace</dt>
            <dd title={workspace}>{workspace}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>
              <span className="metaharness-status-dot" aria-hidden="true" />
              Not configured
            </dd>
          </div>
        </dl>

        <button
          className="metaharness-button"
          type="button"
          onClick={() => host.openSettings()}
        >
          Open Settings
        </button>
      </div>
    </main>
  );
}
