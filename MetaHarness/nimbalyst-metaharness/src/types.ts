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
  pollIntervalMs?: number;
}

export interface RuntimeConfigCall {
  settings?: RuntimeConfig;
}

/** GET /api/v1/health (web/server.py). */
export interface HealthResponse extends JsonObject {
  service: 'metaharness';
  api_version: number;
  status: string;
  control_api?: boolean;
  capabilities_endpoint?: string;
}

export type MetaHarnessConfigResponse = JsonObject;
export type ModelProfilesResponse = JsonObject;

/** One entry of GET /api/v1/runs (web/api.py `_state_summary`). */
export interface RunSummary extends JsonObject {
  run_id: string;
  status?: string | null;
  updated_at?: string | null;
  plan_title?: string | null;
  commit_sha?: string | null;
  candidate?: unknown;
  failure?: unknown;
}
export type RunDetail = JsonObject;

/** GET /api/v1/runs/<id>/progress: byte offset of the next unread event. */
export interface ProgressResponse extends JsonObject {
  next_offset: number;
  events: string[];
}

/** 202 body of POST /api/v1/runs, /resume and /recover-plan. */
export interface MutationAccepted extends JsonObject {
  ok: true;
  run_id: string;
  location: string;
  accepted?: boolean;
}
export type CreateRunResponse = MutationAccepted;

/** GET /api/v1/runs/<id>/artifact (web/api.py `get_artifact`). */
export interface ArtifactResponse extends JsonObject {
  run_id?: string;
  name: string;
  exists: boolean;
  encoding?: string;
  content: string | null;
  truncated: boolean;
  size: number;
}

export interface CreateRunInput extends JsonObject {
  spec: string;
  run_id?: string;
  planner_profile?: string;
  mechanical_profile?: string;
  reasoning_profile?: string;
  agentic_profile?: string;
  final_reviewer_profile?: string;
  semantic_reviser_profile?: string;
  check_repair_profile?: string;
  semantic_revision_enabled?: boolean;
  max_check_repair_attempts?: number;
  max_correction_cycles?: number;
  decomposition?: string;
  execution_mode_policy?: string;
  single_step_max_mutable_paths?: number;
  staged_step_max_mutable_paths?: number;
  repair_scope_policy?: string;
  repair_scope_max_added_paths?: number;
}

export interface ApproveRunInput extends JsonObject {
  decision: string;
  final_reviewer_profile?: string;
  semantic_reviser_profile?: string;
  check_repair_profile?: string;
  step_profiles?: Record<string, string>;
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
  api_version?: number;
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
    status: (input?: RuntimeConfigCall) => Promise<MetaHarnessStatus & { recommendedConfigPath?: string }>;
    start: (input?: RuntimeConfigCall) => Promise<BackendToolResult<MetaHarnessStatus>>;
    stop: () => Promise<BackendToolResult<MetaHarnessStatus>>;
    get_config: (input?: RuntimeConfigCall) => Promise<BackendToolResult<MetaHarnessConfigResponse>>;
    model_profiles: () => Promise<BackendToolResult<ModelProfilesResponse>>;
    list_runs: () => Promise<BackendToolResult<RunSummary[]>>;
    get_run: (
      input: { runId: string } | string
    ) => Promise<BackendToolResult<RunDetail>>;
    get_artifact: (input: { runId: string; name: string }) => Promise<BackendToolResult<ArtifactResponse>>;
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
    ) => Promise<BackendToolResult<MutationAccepted>>;
    recover_plan: (
      input: { runId: string; input?: RecoverPlanInput; plan?: string } & JsonObject
    ) => Promise<BackendToolResult<MutationAccepted>>;
    doctor: (input?: RuntimeConfigCall) => Promise<BackendToolResult<JsonObject>>;
  };
  deactivate: () => void | Promise<void>;
}
