import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type {
  ApproveRunInput,
  ApproveScopeInput,
  BackendActivateContext,
  BackendError,
  BackendToolDescriptor,
  BackendToolResult,
  CreateRunInput,
  CreateRunResponse,
  HealthResponse,
  JsonObject,
  MetaHarnessBackend,
  MetaHarnessConfigResponse,
  MetaHarnessStatus,
  ModelProfilesResponse,
  ProgressResponse,
  RecoverPlanInput,
  RunDetail,
  RunSummary,
  RuntimeConfig,
  RuntimeConfigCall,
} from './types';

const DEFAULT_EXECUTABLE = 'metaharness';
const DEFAULT_PORT = 8765;
const DEFAULT_REQUEST_TIMEOUT_MS = 2_500;
const START_TIMEOUT_MS = 15_000;
const START_POLL_MS = 150;
const STOP_TIMEOUT_MS = 3_000;
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const MAX_TOKEN_BYTES = 4 * 1024;
const MAX_DOCTOR_OUTPUT_BYTES = 4 * 1024 * 1024;
const TOKEN_FILE_NAME = 'metaharness-control.token';

const EMPTY_OBJECT_SCHEMA = (): BackendToolDescriptor['inputSchema'] => ({
  type: 'object',
  properties: {},
  additionalProperties: false,
});

export const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  executable: DEFAULT_EXECUTABLE,
  configPath: '',
  port: DEFAULT_PORT,
  autoStart: true,
  pollIntervalMs: 1_000,
};

function isRecord(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asErrorDetails(value: unknown): unknown {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (Array.isArray(value) || isRecord(value)) return value;
  return String(value);
}

/** An expected, JSON-safe backend error. */
export class MetaHarnessBackendError extends Error implements BackendError {
  readonly code: string;
  readonly httpStatus?: number;
  readonly details?: unknown;

  constructor(
    code: string,
    message: string,
    httpStatus?: number,
    details?: unknown,
  ) {
    super(message);
    this.name = 'MetaHarnessBackendError';
    this.code = code;
    this.httpStatus = httpStatus;
    this.details = asErrorDetails(details);
  }

  toJSON(): BackendError {
    const result: BackendError = { code: this.code, message: this.message };
    if (this.httpStatus !== undefined) result.httpStatus = this.httpStatus;
    if (this.details !== undefined) result.details = this.details;
    return result;
  }
}

function toBackendError(value: unknown, development = false): BackendError {
  if (value instanceof MetaHarnessBackendError) {
    const result = value.toJSON();
    if (development && value.stack) {
      result.details = {
        ...(isRecord(result.details) ? result.details : {}),
        stack: value.stack,
      };
    }
    return result;
  }

  const message = value instanceof Error ? value.message : 'backend operation failed';
  const result: BackendError = { code: 'BACKEND_ERROR', message };
  if (development && value instanceof Error && value.stack) {
    result.details = { stack: value.stack };
  }
  return result;
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

function validatePort(port: number): number {
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new MetaHarnessBackendError(
      'INVALID_CONFIG',
      'port must be an integer between 1 and 65535',
    );
  }
  return port;
}

function validateRuntimeConfig(input: RuntimeConfig): RuntimeConfig {
  if (!input.executable || typeof input.executable !== 'string') {
    throw new MetaHarnessBackendError('INVALID_CONFIG', 'executable must be a non-empty string');
  }
  if (typeof input.configPath !== 'string') {
    throw new MetaHarnessBackendError('INVALID_CONFIG', 'configPath must be a string');
  }
  if (typeof input.autoStart !== 'boolean') {
    throw new MetaHarnessBackendError('INVALID_CONFIG', 'autoStart must be a boolean');
  }
  if (input.pollIntervalMs !== undefined && (!Number.isInteger(input.pollIntervalMs)
    || input.pollIntervalMs < 500 || input.pollIntervalMs > 30_000)) {
    throw new MetaHarnessBackendError('INVALID_CONFIG', 'pollIntervalMs must be between 500 and 30000');
  }
  return { ...input, port: validatePort(input.port) };
}

function mergeRuntimeConfig(
  ...sources: Array<Partial<RuntimeConfig> | undefined>
): RuntimeConfig {
  const merged: RuntimeConfig = { ...DEFAULT_RUNTIME_CONFIG };
  for (const source of sources) {
    if (!source) continue;
    for (const [key, value] of Object.entries(source)) {
      if (value !== undefined) Object.assign(merged, { [key]: value });
    }
  }
  return validateRuntimeConfig(merged);
}

function validateLocalBaseUrl(baseUrl: string): { url: string; port: number } {
  let parsed: URL;
  try {
    parsed = new URL(baseUrl);
  } catch {
    throw new MetaHarnessBackendError(
      'INVALID_BASE_URL',
      'baseUrl must be http://127.0.0.1:<port>',
    );
  }

  const port = Number(parsed.port);
  if (
    parsed.protocol !== 'http:'
    || parsed.hostname !== '127.0.0.1'
    || !parsed.port
    || !Number.isInteger(port)
    || port < 1
    || port > 65_535
    || parsed.username
    || parsed.password
    || parsed.pathname !== '/'
    || parsed.search
    || parsed.hash
  ) {
    throw new MetaHarnessBackendError(
      'INVALID_BASE_URL',
      'baseUrl must be http://127.0.0.1:<port>',
    );
  }
  return { url: `http://127.0.0.1:${port}`, port };
}

function jsonResponseObject(value: unknown, operation = 'response'): JsonObject {
  if (!isRecord(value)) {
    throw new MetaHarnessBackendError(
      'INVALID_RESPONSE',
      `${operation} returned a JSON value that is not an object`,
    );
  }
  return value;
}

function validateRunId(runId: string): string {
  if (
    typeof runId !== 'string'
    || !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(runId)
    || runId === '.'
    || runId === '..'
    || runId.includes('..')
    || runId.endsWith('.')
    || runId.endsWith('.lock')
  ) {
    throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'runId is invalid');
  }
  return runId;
}

function argumentRunId(value: string | { runId: string }): string {
  return validateRunId(typeof value === 'string' ? value : value.runId);
}

function mutationInput<T extends JsonObject>(value: T | { input: T }): T {
  const candidate = value as JsonObject;
  const input = isRecord(candidate.input) && Object.keys(candidate).length === 1
    ? candidate.input
    : candidate;
  if (!isRecord(input)) {
    throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'request input must be a JSON object');
  }
  return input as T;
}

function isConnectionRefused(error: unknown): boolean {
  if (!(error instanceof MetaHarnessBackendError) || error.code !== 'NETWORK_ERROR') return false;
  const details = isRecord(error.details) ? error.details : {};
  return details.causeCode === 'ECONNREFUSED'
    || details.causeCode === 'ERR_CONNECTION_REFUSED'
    || /ECONNREFUSED|connection refused/i.test(error.message);
}

function processIsAlive(child: ChildProcess | undefined): boolean {
  return child !== undefined && child.exitCode === null && child.signalCode === null;
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (!processIsAlive(child)) return Promise.resolve(true);
  return new Promise((resolvePromise) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout>;
    const finish = (exited: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.removeListener('exit', onExit);
      child.removeListener('close', onExit);
      resolvePromise(exited);
    };
    const onExit = () => finish(true);
    timer = setTimeout(() => finish(false), timeoutMs);
    child.once('exit', onExit);
    child.once('close', onExit);
  });
}

export interface MetaHarnessClientOptions {
  baseUrl: string;
  tokenFile: string;
  requestTimeoutMs?: number;
}

/** HTTP-only client for the loopback MetaHarness control API. */
export class MetaHarnessClient {
  readonly baseUrl: string;
  readonly tokenFile: string;
  readonly requestTimeoutMs: number;
  private readonly port: number;

  constructor(options: MetaHarnessClientOptions) {
    const parsed = validateLocalBaseUrl(options.baseUrl);
    if (!options.tokenFile || typeof options.tokenFile !== 'string') {
      throw new MetaHarnessBackendError('INVALID_CONFIG', 'tokenFile must be a non-empty path');
    }
    const requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    if (!Number.isInteger(requestTimeoutMs) || requestTimeoutMs < 1) {
      throw new MetaHarnessBackendError('INVALID_CONFIG', 'requestTimeoutMs must be a positive integer');
    }
    this.baseUrl = parsed.url;
    this.port = parsed.port;
    this.tokenFile = options.tokenFile;
    this.requestTimeoutMs = requestTimeoutMs;
  }

  private endpoint(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private readToken(): string {
    let raw: string;
    try {
      const bytes = readFileSync(this.tokenFile);
      if (bytes.byteLength > MAX_TOKEN_BYTES) {
        throw new MetaHarnessBackendError('TOKEN_INVALID', 'control token file is too large');
      }
      try {
        raw = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
      } catch {
        throw new MetaHarnessBackendError('TOKEN_INVALID', 'control token file is invalid');
      }
    } catch (error) {
      if (error instanceof MetaHarnessBackendError) throw error;
      const code = isRecord(error) && typeof error.code === 'string' ? error.code : undefined;
      if (code === 'ENOENT') {
        throw new MetaHarnessBackendError('TOKEN_MISSING', 'control token file is missing');
      }
      throw new MetaHarnessBackendError('TOKEN_UNREADABLE', 'control token file could not be read');
    }
    const token = raw.endsWith('\n') ? raw.slice(0, -1).replace(/\r$/, '') : raw;
    if (!token || token.includes('\n') || token.includes('\r')) {
      throw new MetaHarnessBackendError('TOKEN_INVALID', 'control token file is invalid');
    }
    return token;
  }

  private async request<T>(
    method: 'GET' | 'POST',
    path: string,
    body?: JsonObject,
    timeoutMs = this.requestTimeoutMs,
  ): Promise<T> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    let serializedBody: string | undefined;
    if (method === 'POST') {
      headers['Content-Type'] = 'application/json';
      headers['X-MetaHarness-Token'] = this.readToken();
      try {
        serializedBody = JSON.stringify(body ?? {});
      } catch {
        throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'request body is not JSON serializable');
      }
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response: Response;
    let text: string;
    try {
      response = await fetch(this.endpoint(path), {
        method,
        headers,
        body: serializedBody,
        signal: controller.signal,
      });
      text = await response.text();
    } catch (error) {
      if (controller.signal.aborted) {
        throw new MetaHarnessBackendError('TIMEOUT', `MetaHarness request timed out after ${timeoutMs}ms`);
      }
      const cause = isRecord(error) && isRecord(error.cause) ? error.cause : undefined;
      const errorMessage = error instanceof Error ? error.message : '';
      const causeCode = cause && typeof cause.code === 'string'
        ? cause.code
        : /ECONNREFUSED|connection refused/i.test(errorMessage)
          ? 'ECONNREFUSED'
          : undefined;
      throw new MetaHarnessBackendError(
        'NETWORK_ERROR',
        'MetaHarness could not be reached',
        undefined,
        causeCode ? { causeCode } : undefined,
      );
    } finally {
      clearTimeout(timer);
    }
    if (Buffer.byteLength(text, 'utf8') > MAX_RESPONSE_BYTES) {
      throw new MetaHarnessBackendError('RESPONSE_TOO_LARGE', 'MetaHarness response is too large');
    }

    let payload: unknown;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      throw new MetaHarnessBackendError(
        'JSON_INVALID',
        'MetaHarness returned invalid JSON',
        response.status,
      );
    }
    if (!response.ok) {
      const message = isRecord(payload) && typeof payload.message === 'string'
        ? payload.message
        : `MetaHarness returned HTTP ${response.status}`;
      throw new MetaHarnessBackendError('HTTP_ERROR', message, response.status, payload);
    }
    return payload as T;
  }

  async health(): Promise<HealthResponse> {
    const payload = jsonResponseObject(
      await this.request<unknown>('GET', '/api/v1/health'),
      'health',
    ) as HealthResponse;
    if (payload.service !== 'metaharness') {
      throw new MetaHarnessBackendError(
        'SERVICE_MISMATCH',
        'loopback service is not MetaHarness',
        undefined,
        { service: payload.service ?? null, port: this.port },
      );
    }
    return payload;
  }

  async getConfig(): Promise<MetaHarnessConfigResponse> {
    return jsonResponseObject(
      await this.request<unknown>('GET', '/api/v1/config'),
      'get_config',
    );
  }

  async modelProfiles(): Promise<ModelProfilesResponse> {
    return jsonResponseObject(
      await this.request<unknown>('GET', '/api/v1/model-profiles'),
      'model_profiles',
    );
  }

  async listRuns(): Promise<RunSummary[]> {
    const payload = jsonResponseObject(
      await this.request<unknown>('GET', '/api/v1/runs'),
      'list_runs',
    );
    if (!Array.isArray(payload.runs)) {
      throw new MetaHarnessBackendError('INVALID_RESPONSE', 'list_runs response has no runs array');
    }
    return payload.runs as RunSummary[];
  }

  async getRun(runId: string): Promise<RunDetail> {
    return jsonResponseObject(
      await this.request<unknown>('GET', `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}`),
      'get_run',
    );
  }

  async getArtifact(runId: string, name: string): Promise<JsonObject> {
    if (!name || name.includes('..') || name.startsWith('/') || name.includes('\\')) {
      throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'artifact name is invalid');
    }
    return jsonResponseObject(
      await this.request<unknown>(
        'GET', `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}/artifact?name=${name}`,
      ),
      'get_artifact',
    );
  }

  async progress(runId: string, offset: number): Promise<ProgressResponse> {
    if (!Number.isInteger(offset) || offset < 0) {
      throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'offset must be a non-negative integer');
    }
    return jsonResponseObject(
      await this.request<unknown>(
        'GET',
        `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}/progress?offset=${offset}`,
      ),
      'progress',
    );
  }

  async createRun(input: CreateRunInput): Promise<CreateRunResponse> {
    return jsonResponseObject(
      await this.request<unknown>('POST', '/api/v1/runs', mutationInput(input)),
      'create_run',
    ) as CreateRunResponse;
  }

  async approveRun(
    runId: string,
    input: ApproveRunInput | string,
    extra: JsonObject = {},
  ): Promise<unknown> {
    const selected: ApproveRunInput = typeof input === 'string'
      ? { ...extra, decision: input }
      : input;
    const body: JsonObject = { ...selected };
    const stepProfiles = body.step_profiles;
    delete body.step_profiles;
    if (stepProfiles && typeof stepProfiles === 'object' && !Array.isArray(stepProfiles)) {
      for (const [stepId, profileId] of Object.entries(stepProfiles as Record<string, unknown>)) {
        body[`step_profile__${stepId}`] = profileId;
      }
    }
    return this.request<unknown>(
      'POST',
      `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}/approval`,
      body,
    );
  }

  async approveScope(
    runId: string,
    input: ApproveScopeInput | string,
  ): Promise<unknown> {
    const body: ApproveScopeInput = typeof input === 'string' ? { decision: input } : input;
    return this.request<unknown>(
      'POST',
      `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}/scope-approval`,
      body,
    );
  }

  async resumeRun(runId: string): Promise<unknown> {
    return this.request<unknown>(
      'POST',
      `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}/resume`,
      {},
    );
  }

  async recoverPlan(runId: string, input: RecoverPlanInput | string): Promise<unknown> {
    const body: RecoverPlanInput = typeof input === 'string' ? { plan: input } : input;
    return this.request<unknown>(
      'POST',
      `/api/v1/runs/${encodeURIComponent(validateRunId(runId))}/recover-plan`,
      body,
    );
  }
}

export interface MetaHarnessRuntimeOptions {
  config: RuntimeConfig;
  dataDir: string;
  development?: boolean;
}

/** Owns the optional MetaHarness child process and never owns an external one. */
export class MetaHarnessRuntime {
  readonly config: RuntimeConfig;
  readonly dataDir: string;
  readonly tokenFile: string;
  readonly client: MetaHarnessClient;
  private readonly development: boolean;
  private ownedChild: ChildProcess | undefined;
  private startPromise: Promise<MetaHarnessStatus> | undefined;
  private readonly spawnErrors = new WeakMap<ChildProcess, MetaHarnessBackendError>();

  constructor(options: MetaHarnessRuntimeOptions) {
    this.config = validateRuntimeConfig(options.config);
    this.dataDir = resolve(options.dataDir);
    this.tokenFile = join(this.dataDir, TOKEN_FILE_NAME);
    this.client = new MetaHarnessClient({
      baseUrl: `http://127.0.0.1:${this.config.port}`,
      tokenFile: this.tokenFile,
    });
    this.development = options.development ?? process.env.NODE_ENV === 'development';
  }

  get isDevelopment(): boolean {
    return this.development;
  }

  get ownedProcess(): ChildProcess | undefined {
    return this.ownedChild;
  }

  private configured(): boolean {
    return this.config.configPath.trim().length > 0;
  }

  async status(): Promise<MetaHarnessStatus> {
    const result: MetaHarnessStatus = {
      configured: this.configured(),
      connected: false,
      serverOwned: processIsAlive(this.ownedChild),
    };
    try {
      const health = await this.client.health();
      result.connected = true;
      result.api_version = health.api_version;
    } catch {
      // Status is a probe, not an error channel: callers only need the three
      // stable state fields. Mutating and control operations return BackendError.
    }
    result.serverOwned = processIsAlive(this.ownedChild);
    return result;
  }

  private async probeBeforeStart(): Promise<boolean> {
    try {
      await this.client.health();
      return true;
    } catch (error) {
      if (isConnectionRefused(error)) return false;
      const probeError = toBackendError(error, this.development);
      throw new MetaHarnessBackendError(
        'PORT_IN_USE',
        `port ${this.config.port} is occupied by a non-MetaHarness service`,
        undefined,
        { probeCode: probeError.code },
      );
    }
  }

  private spawnOwnedProcess(): ChildProcess {
    try {
      mkdirSync(this.dataDir, { recursive: true });
      const child = spawn(
        this.config.executable,
        [
          'web',
          '--config', this.config.configPath,
          '--port', String(this.config.port),
          '--control-token-file', this.tokenFile,
        ],
        {
          shell: false,
          stdio: ['ignore', 'pipe', 'pipe'],
        },
      );
      // Drain output without logging it. It may contain provider diagnostics or
      // other data that must not cross into the panel logs.
      child.stdout?.resume();
      child.stderr?.resume();
      child.once('error', (error) => {
        this.spawnErrors.set(child, new MetaHarnessBackendError(
          'SPAWN_FAILED',
          'MetaHarness could not be started',
          undefined,
          { causeCode: isRecord(error) && typeof error.code === 'string' ? error.code : undefined },
        ));
      });
      child.once('exit', () => {
        if (this.ownedChild === child) this.ownedChild = undefined;
      });
      this.ownedChild = child;
      return child;
    } catch {
      throw new MetaHarnessBackendError('SPAWN_FAILED', 'MetaHarness could not be started');
    }
  }

  private async waitForHealth(child: ChildProcess): Promise<void> {
    const deadline = Date.now() + START_TIMEOUT_MS;
    let lastError: BackendError | undefined;
    while (Date.now() < deadline) {
      const spawnError = this.spawnErrors.get(child);
      if (spawnError) throw spawnError;
      if (!processIsAlive(child)) {
        throw new MetaHarnessBackendError(
          'SPAWN_FAILED',
          'MetaHarness exited before its health endpoint became ready',
          undefined,
          { exitCode: child.exitCode, signal: child.signalCode },
        );
      }
      try {
        await this.client.health();
        return;
      } catch (error) {
        lastError = toBackendError(error, this.development);
      }
      await sleep(Math.min(START_POLL_MS, Math.max(1, deadline - Date.now())));
    }
    throw new MetaHarnessBackendError(
      'START_TIMEOUT',
      `MetaHarness did not become healthy within ${START_TIMEOUT_MS}ms`,
      lastError?.httpStatus,
      lastError ? { lastError: lastError.code } : undefined,
    );
  }

  async start(): Promise<MetaHarnessStatus> {
    if (this.startPromise) return this.startPromise;
    this.startPromise = this.startInternal();
    try {
      return await this.startPromise;
    } finally {
      this.startPromise = undefined;
    }
  }

  private async startInternal(): Promise<MetaHarnessStatus> {
    if (!this.configured()) {
      throw new MetaHarnessBackendError('NOT_CONFIGURED', 'configPath is not configured');
    }
    if (processIsAlive(this.ownedChild)) return this.status();

    const alreadyRunning = await this.probeBeforeStart();
    if (alreadyRunning) return this.status();

    const child = this.spawnOwnedProcess();
    try {
      await this.waitForHealth(child);
      return this.status();
    } catch (error) {
      if (this.ownedChild === child && processIsAlive(child)) {
        child.kill('SIGTERM');
        await waitForExit(child, STOP_TIMEOUT_MS);
      }
      throw error;
    }
  }

  async stop(): Promise<MetaHarnessStatus> {
    const child = this.ownedChild;
    if (child && processIsAlive(child)) {
      child.kill('SIGTERM');
      const exited = await waitForExit(child, STOP_TIMEOUT_MS);
      if (!exited) {
        throw new MetaHarnessBackendError(
          'STOP_TIMEOUT',
          `owned MetaHarness process did not exit within ${STOP_TIMEOUT_MS}ms`,
        );
      }
    }
    return this.status();
  }

  async deactivate(): Promise<void> {
    if (this.ownedChild && processIsAlive(this.ownedChild)) await this.stop();
  }

  async doctor(): Promise<JsonObject> {
    if (!this.configured()) {
      throw new MetaHarnessBackendError('NOT_CONFIGURED', 'configPath is not configured');
    }
    let child: ChildProcess;
    try {
      child = spawn(
        this.config.executable,
        ['doctor', '--config', this.config.configPath, '--json'],
        { shell: false, stdio: ['ignore', 'pipe', 'pipe'] },
      );
    } catch {
      throw new MetaHarnessBackendError('SPAWN_FAILED', 'MetaHarness doctor could not be started');
    }

    let stdout = '';
    let outputTooLarge = false;
    child.stdout?.on('data', (chunk: Buffer | string) => {
      if (outputTooLarge) return;
      const next = stdout + chunk.toString();
      if (Buffer.byteLength(next, 'utf8') > MAX_DOCTOR_OUTPUT_BYTES) {
        outputTooLarge = true;
        child.kill('SIGTERM');
        return;
      }
      stdout = next;
    });
    child.stderr?.resume();

    const exit = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolvePromise, reject) => {
      let settled = false;
      const finish = (value: { code: number | null; signal: NodeJS.Signals | null }) => {
        if (!settled) {
          settled = true;
          resolvePromise(value);
        }
      };
      child.once('error', (error) => {
        if (!settled) {
          settled = true;
          reject(new MetaHarnessBackendError(
            'SPAWN_FAILED',
            'MetaHarness doctor could not be started',
            undefined,
            { causeCode: isRecord(error) && typeof error.code === 'string' ? error.code : undefined },
          ));
        }
      });
      child.once('close', (code, signal) => finish({ code, signal }));
    });

    if (outputTooLarge) {
      throw new MetaHarnessBackendError('DOCTOR_OUTPUT_TOO_LARGE', 'doctor output is too large');
    }
    if (exit.code !== 0) {
      throw new MetaHarnessBackendError(
        'DOCTOR_FAILED',
        'metaharness doctor failed',
        undefined,
        { exitCode: exit.code, signal: exit.signal },
      );
    }
    let payload: unknown;
    try {
      payload = JSON.parse(stdout);
    } catch {
      throw new MetaHarnessBackendError('DOCTOR_INVALID_JSON', 'metaharness doctor returned invalid JSON');
    }
    return jsonResponseObject(payload, 'doctor');
  }
}

function toolError<T>(
  operation: () => Promise<T>,
  development: boolean,
): Promise<BackendToolResult<T>> {
  return operation().catch((error) => ({
    ok: false as const,
    error: toBackendError(error, development),
  }));
}

function descriptor(
  name: string,
  description: string,
  inputSchema = EMPTY_OBJECT_SCHEMA(),
): BackendToolDescriptor {
  return { name, description, inputSchema, scope: 'global' };
}

export const MCP_TOOL_DESCRIPTORS: BackendToolDescriptor[] = [
  descriptor('status', 'READ-ONLY. Return MetaHarness connection and ownership status for the supplied settings.', {
    type: 'object', properties: { settings: { type: 'object', properties: {
      executable: { type: 'string' }, configPath: { type: 'string' }, port: { type: 'integer' },
      autoStart: { type: 'boolean' }, pollIntervalMs: { type: 'integer' },
    }, additionalProperties: false } }, additionalProperties: false,
  }),
  descriptor('get_config', 'READ-ONLY. Return the credential-free MetaHarness configuration description.', {
    type: 'object', properties: { settings: { type: 'object', properties: {
      executable: { type: 'string' }, configPath: { type: 'string' }, port: { type: 'integer' },
      autoStart: { type: 'boolean' }, pollIntervalMs: { type: 'integer' },
    }, additionalProperties: false } }, additionalProperties: false,
  }),
  descriptor('model_profiles', 'READ-ONLY. Return available MetaHarness model profiles.'),
  descriptor('list_runs', 'READ-ONLY. List MetaHarness runs.'),
  descriptor('get_run', 'READ-ONLY. Return one MetaHarness run.', {
    type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'], additionalProperties: false,
  }),
  descriptor('get_artifact', 'READ-ONLY. Return one allowlisted text artifact from a MetaHarness run.', {
    type: 'object', properties: { runId: { type: 'string' }, name: { type: 'string' } },
    required: ['runId', 'name'], additionalProperties: false,
  }),
  descriptor('progress', 'READ-ONLY. Return bounded progress events for one run.', {
    type: 'object', properties: { runId: { type: 'string' }, offset: { type: 'integer', minimum: 0 } },
    required: ['runId', 'offset'], additionalProperties: false,
  }),
  descriptor('create_run', 'MUTATING ACTION. Only use at the user’s explicit request. Never approve or reject a plan automatically. Never recover a plan without supplied plan content and clear intent. Never launch multiple runs to compensate for an error.', {
    type: 'object', properties: {
      spec: { type: 'string', minLength: 1 },
      run_id: { type: 'string' }, planner_profile: { type: 'string' }, mechanical_profile: { type: 'string' },
      reasoning_profile: { type: 'string' }, agentic_profile: { type: 'string' }, final_reviewer_profile: { type: 'string' },
      semantic_reviser_profile: { type: 'string' }, check_repair_profile: { type: 'string' },
      semantic_revision_enabled: { type: 'boolean' }, max_check_repair_attempts: { type: 'integer' },
      max_review_repair_cycles: { type: 'integer' }, decomposition: { type: 'string' },
      execution_mode_policy: { type: 'string' }, single_step_max_mutable_paths: { type: 'integer' },
      staged_step_max_mutable_paths: { type: 'integer' }, repair_scope_policy: { type: 'string' },
      repair_scope_max_added_paths: { type: 'integer' },
    }, required: ['spec'], additionalProperties: false,
  }),
  descriptor('approve_run', 'MUTATING ACTION. Approve or reject a MetaHarness plan. Only use at the user’s explicit request for this exact run. Never approve or reject a plan automatically. Never recover a plan without supplied plan content and clear intent. Never launch multiple runs to compensate for an error. Requires the exact runId and decision; never infer the latest run.', {
    type: 'object', properties: {
      runId: { type: 'string', minLength: 1 }, decision: { type: 'string', enum: ['APPROVE', 'REJECT'] },
      final_reviewer_profile: { type: 'string' }, semantic_reviser_profile: { type: 'string' },
      check_repair_profile: { type: 'string' }, step_profiles: { type: 'object', additionalProperties: { type: 'string' } },
    }, required: ['runId', 'decision'], additionalProperties: false,
  }),
  descriptor('approve_scope', 'MUTATING ACTION. Approve or reject a MetaHarness repair scope. Only use at the user’s explicit request for this exact run. Never approve or reject a plan automatically. Never recover a plan without supplied plan content and clear intent. Never launch multiple runs to compensate for an error. Requires the exact runId and decision; never infer the latest run.', {
    type: 'object', properties: {
      runId: { type: 'string', minLength: 1 }, decision: { type: 'string', enum: ['APPROVE', 'REJECT'] },
    }, required: ['runId', 'decision'], additionalProperties: false,
  }),
  descriptor('resume_run', 'MUTATING ACTION. Resume one MetaHarness run. Only use at the user’s explicit request. Never approve or reject a plan automatically. Never recover a plan without supplied plan content and clear intent. Never launch multiple runs to compensate for an error.', {
    type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'], additionalProperties: false,
  }),
  descriptor('recover_plan', 'MUTATING ACTION. Replace a plan for one recoverable MetaHarness run. Only use at the user’s explicit request. Never approve or reject a plan automatically. Never recover a plan without non-empty plan content supplied by the user and clear intent. Never launch multiple runs to compensate for an error. Requires the exact runId; never infer the latest run.', {
    type: 'object', properties: {
      runId: { type: 'string', minLength: 1 }, input: { type: 'object', properties: {
        plan: { type: 'string', minLength: 1 },
      }, required: ['plan'], additionalProperties: false },
    }, required: ['runId', 'input'], additionalProperties: false,
  }),
];

function runtimeDataDir(context: BackendActivateContext): string {
  return context.dataDir
    ?? context.services.dataDir
    ?? join(context.services.workspacePath || context.services.extensionPath, '.nimbalyst');
}

export async function activate(
  context: BackendActivateContext,
  runtimeConfig?: Partial<RuntimeConfig>,
): Promise<MetaHarnessBackend> {
  const config = mergeRuntimeConfig(
    context.services.runtimeConfig,
    context.runtimeConfig,
    { configPath: context.services.configPath },
    runtimeConfig,
  );
  const runtime = new MetaHarnessRuntime({
    config,
    dataDir: runtimeDataDir(context),
  });
  const workspacePath = context.services.workspacePath || context.services.extensionPath;
  const startedRuntimes = new Map<string, MetaHarnessRuntime>();
  const runtimeForSettings = (input: RuntimeConfigCall) => {
    const selectedConfig = mergeRuntimeConfig(input.settings);
    const key = JSON.stringify(selectedConfig);
    const selected = startedRuntimes.get(key)
      ?? new MetaHarnessRuntime({ config: selectedConfig, dataDir: runtimeDataDir(context) });
    return { key, runtime: selected };
  };
  const forCall = (input?: RuntimeConfigCall) => {
    if (!input?.settings) return runtime;
    return runtimeForSettings(input).runtime;
  };
  const recommendedConfigPath = () => {
    for (const name of ['metaharness.toml', '.metaharness.toml']) {
      const candidate = join(workspacePath, name);
      if (existsSync(candidate)) return candidate;
    }
    return undefined;
  };

  await context.services.registerMcpTools(MCP_TOOL_DESCRIPTORS);
  context.services.log('info', '[metaharness] control tools registered');

  const methods: MetaHarnessBackend['methods'] = {
    status: async (input) => ({
      ...await forCall(input).status(),
      ...(recommendedConfigPath() ? { recommendedConfigPath: recommendedConfigPath() } : {}),
    }),
    start: (input) => {
      if (!input?.settings) {
        return toolError(() => runtime.start(), runtime.isDevelopment);
      }
      const { key, runtime: selected } = runtimeForSettings(input);
      startedRuntimes.set(key, selected);
      return toolError(() => selected.start(), selected.isDevelopment);
    },
    stop: () => toolError(() => runtime.stop(), runtime.isDevelopment),
    get_config: (input) => {
      const selected = forCall(input);
      return toolError(() => selected.client.getConfig(), selected.isDevelopment);
    },
    model_profiles: () => toolError(() => runtime.client.modelProfiles(), runtime.isDevelopment),
    list_runs: () => toolError(() => runtime.client.listRuns(), runtime.isDevelopment),
    get_run: (input) => toolError(
      () => runtime.client.getRun(argumentRunId(input)),
      runtime.isDevelopment,
    ),
    get_artifact: (input) => toolError(
      () => runtime.client.getArtifact(validateRunId(input.runId), String(input.name ?? '')),
      runtime.isDevelopment,
    ),
    progress: (input) => toolError(
      () => runtime.client.progress(validateRunId(input.runId), input.offset),
      runtime.isDevelopment,
    ),
    create_run: (input) => toolError(() => {
      const request = mutationInput<CreateRunInput>(input);
      if (typeof request.spec !== 'string' || !request.spec.trim()) {
        throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'spec is required');
      }
      return runtime.client.createRun(request);
    }, runtime.isDevelopment),
    approve_run: (input) => toolError(
      () => {
        if (typeof input.decision !== 'string' || !input.decision) {
          throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'decision is required');
        }
        const options = Object.fromEntries(Object.entries(input).filter(([key]) => !['runId', 'decision'].includes(key)));
        return runtime.client.approveRun(
          validateRunId(input.runId), { ...options, decision: input.decision },
        ).then((value) => jsonResponseObject(value, 'approve_run'));
      },
      runtime.isDevelopment,
    ),
    approve_scope: (input) => toolError(
      () => {
        if (typeof input.decision !== 'string' || !input.decision) {
          throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'decision is required');
        }
        return runtime.client.approveScope(
          validateRunId(input.runId), { decision: input.decision },
        ).then((value) => jsonResponseObject(value, 'approve_scope'));
      },
      runtime.isDevelopment,
    ),
    resume_run: (input) => toolError(
      () => runtime.client.resumeRun(argumentRunId(input)).then((value) => jsonResponseObject(value, 'resume_run')),
      runtime.isDevelopment,
    ),
    recover_plan: (input) => toolError(
      () => {
        const planInput: JsonObject = isRecord(input.input) ? input.input : {};
        if (typeof planInput.plan !== 'string' || !planInput.plan.trim()) {
          throw new MetaHarnessBackendError('INVALID_ARGUMENT', 'non-empty plan content is required');
        }
        return runtime.client.recoverPlan(
          validateRunId(input.runId), planInput as RecoverPlanInput,
        ).then((value) => jsonResponseObject(value, 'recover_plan'));
      },
      runtime.isDevelopment,
    ),
    doctor: (input) => {
      const selected = forCall(input);
      return toolError(() => selected.doctor(), selected.isDevelopment);
    },
  };

  if (config.autoStart && config.configPath.trim()) {
    try {
      await runtime.start();
    } catch (error) {
      context.services.log('warn', '[metaharness] auto-start failed', {
        code: toBackendError(error).code,
      });
    }
  }

  return {
    methods,
    deactivate: async () => {
      await Promise.all([runtime.deactivate(), ...Array.from(startedRuntimes.values(), (item) => item.deactivate())]);
    },
  };
}
