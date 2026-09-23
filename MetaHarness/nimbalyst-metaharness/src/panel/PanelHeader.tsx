export function ConfigurationButton({ configPath, onOpen }: { configPath?: string; onOpen: () => void }) {
  const title = configPath ? `MetaHarness configuration\n${configPath}` : 'MetaHarness configuration';
  return (
    <button type="button" className="metaharness-icon-button" aria-label="Open MetaHarness configuration" title={title} onClick={onOpen}>
      <span aria-hidden="true">⚙</span>
    </button>
  );
}

export function PanelHeader({ title = 'MetaHarness', titleId, connection, connected = false, configPath, onOpenConfiguration }: {
  title?: string;
  titleId?: string;
  connection?: string;
  connected?: boolean;
  configPath?: string;
  onOpenConfiguration?: () => void;
}) {
  return (
    <header className="metaharness-dashboard__header metaharness-panel-header">
      <h1 id={titleId}>{title}</h1>
      <div className="metaharness-panel-header__status">
        {connection && <span className={`metaharness-connection ${connected ? 'is-connected' : ''}`}>
          <span className="metaharness-status-dot" aria-hidden="true" />
          {connection}
        </span>}
        {onOpenConfiguration && <ConfigurationButton configPath={configPath} onOpen={onOpenConfiguration} />}
      </div>
    </header>
  );
}
