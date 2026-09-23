export type BackendLogLevel = 'debug' | 'info' | 'warn' | 'error';

export type JsonObject = Record<string, unknown>;

export interface BackendError {
  code: string;
  message: string;
  httpStatus?: number;
  details?: unknown;
}

export interface RuntimeConfig {
  executable: string;
  configPath: string;
  port: number;
  autoStart: boolean;
}

export interface HealthResponse extends JsonObject {
  service?: string;
  api_version?: number;
  status?: string;
  control_api?: boolean;
}

export type MetaHarnessConfigResponse = JsonObject;
export type ModelProfilesResponse = JsonObject;
export type RunSummary = JsonObject;
export type RunDetail = JsonObject;
export type ProgressResponse = JsonObject;
export type CreateRunInput = JsonObject;
export type CreateRunResponse = JsonObject;

export interface ApproveRunInput extends JsonObject {
  decision: string;
}

export interface ApproveScopeInput extends JsonObject {
  decision: string;
}

export interface RecoverPlanInput extends JsonObject {
  plan: string;
}

export interface MetaHarnessStatus {
  configured: boolean;
  connected: boolean;
  serverOwned: boolean;
}

export interface BackendToolSchema {
  type: 'object';
  properties: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
}

export interface BackendToolDescriptor {
  name: string;
  description: string;
  inputSchema: BackendToolSchema;
  scope: 'global';
}

export interface BackendActivateContext {
  // Older Nimbalyst hosts only supplied workspacePath/extensionPath. Newer
  // hosts may provide dataDir and runtimeConfig.
  dataDir?: string;
  runtimeConfig?: Partial<RuntimeConfig>;
  services: {
    workspacePath: string;
    extensionPath: string;
    dataDir?: string;
    configPath?: string;
    runtimeConfig?: Partial<RuntimeConfig>;
    log: (level: BackendLogLevel, message: string, data?: unknown) => void;
    registerMcpTools: (
      tools: BackendToolDescriptor[]
    ) => Promise<{ registered: string[] } | void>;
  };
}

export type BackendToolFailure = {
  ok: false;
  error: BackendError;
};

export type BackendToolResult<T> = T | BackendToolFailure;

export interface MetaHarnessBackend {
  methods: {
    status: () => Promise<MetaHarnessStatus>;
    start: () => Promise<BackendToolResult<MetaHarnessStatus>>;
    stop: () => Promise<BackendToolResult<MetaHarnessStatus>>;
    get_config: () => Promise<BackendToolResult<MetaHarnessConfigResponse>>;
    model_profiles: () => Promise<BackendToolResult<ModelProfilesResponse>>;
    list_runs: () => Promise<BackendToolResult<RunSummary[]>>;
    get_run: (
      input: { runId: string } | string
    ) => Promise<BackendToolResult<RunDetail>>;
    progress: (
      input: { runId: string; offset: number }
    ) => Promise<BackendToolResult<ProgressResponse>>;
    create_run: (
      input: CreateRunInput | { input: CreateRunInput }
    ) => Promise<BackendToolResult<CreateRunResponse>>;
    approve_run: (
      input: { runId: string; input?: ApproveRunInput; decision?: string } & JsonObject
    ) => Promise<BackendToolResult<JsonObject>>;
    approve_scope: (
      input: { runId: string; input?: ApproveScopeInput; decision?: string } & JsonObject
    ) => Promise<BackendToolResult<JsonObject>>;
    resume_run: (
      input: { runId: string } | string
    ) => Promise<BackendToolResult<JsonObject>>;
    recover_plan: (
      input: { runId: string; input?: RecoverPlanInput; plan?: string } & JsonObject
    ) => Promise<BackendToolResult<JsonObject>>;
    doctor: () => Promise<BackendToolResult<JsonObject>>;
  };
  deactivate: () => void | Promise<void>;
}
