export type BackendLogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface MetaHarnessStatus {
  configured: false;
  connected: false;
}

export interface BackendToolDescriptor {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, never>;
  };
  scope: 'global';
}

export interface BackendActivateContext {
  services: {
    workspacePath: string;
    extensionPath: string;
    log: (level: BackendLogLevel, message: string, data?: unknown) => void;
    registerMcpTools: (
      tools: BackendToolDescriptor[]
    ) => Promise<{ registered: string[] }>;
  };
}

export interface MetaHarnessBackend {
  methods: {
    status: () => Promise<MetaHarnessStatus>;
  };
  deactivate: () => void | Promise<void>;
}
