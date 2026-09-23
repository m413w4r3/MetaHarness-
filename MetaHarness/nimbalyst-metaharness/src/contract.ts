import type {
  ArtifactResponse,
  BackendToolFailure,
  HealthResponse,
  JsonObject,
  MutationAccepted,
  ProgressResponse,
  RunSummary,
} from './types';

/** The only MetaHarness HTTP API major version this extension speaks. */
export const SUPPORTED_API_VERSION = 1;

export function isRecord(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string';
}

function isOffset(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

/**
 * A backend tool failure envelope. It requires a structured `error`, so a
 * MetaHarness payload that merely carries `ok: false` (a failing doctor
 * report) is never mistaken for a transport failure.
 */
export function isBackendFailure(value: unknown): value is BackendToolFailure {
  return isRecord(value) && value.ok === false && isRecord(value.error)
    && typeof value.error.message === 'string';
}

/** GET /api/v1/health */
export function isHealthResponse(value: unknown): value is HealthResponse {
  return isRecord(value) && value.service === 'metaharness'
    && typeof value.api_version === 'number' && typeof value.status === 'string';
}

/** One entry of GET /api/v1/runs (web/api.py `_state_summary`). */
export function isRunSummary(value: unknown): value is RunSummary {
  return isRecord(value) && typeof value.run_id === 'string' && value.run_id.length > 0
    && isNullableString(value.status ?? null)
    && isNullableString(value.updated_at ?? null)
    && isNullableString(value.plan_title ?? null)
    && isNullableString(value.commit_sha ?? null);
}

/** GET /api/v1/runs/<id>/progress?offset=N */
export function isProgressResponse(value: unknown): value is ProgressResponse {
  return isRecord(value) && isOffset(value.next_offset)
    && Array.isArray(value.events) && value.events.every((event) => typeof event === 'string');
}

/** 202 body of POST /api/v1/runs, /resume and /recover-plan. */
export function isMutationAccepted(value: unknown): value is MutationAccepted {
  return isRecord(value) && value.ok === true && typeof value.run_id === 'string'
    && typeof value.location === 'string';
}

/** GET /api/v1/runs/<id>/artifact?name=... */
export function isArtifactResponse(value: unknown): value is ArtifactResponse {
  return isRecord(value) && typeof value.name === 'string' && typeof value.exists === 'boolean'
    && isNullableString(value.content) && typeof value.truncated === 'boolean'
    && typeof value.size === 'number';
}
