import type { CreateRunInput, JsonObject } from '../types';

export const MAX_SPEC_BYTES = 48 * 1024;

export interface ModelProfile {
  id: string;
  display_name: string;
  roles: string[];
  model?: string;
  provider?: string;
  description?: string;
}

export interface RunFormDefaults {
  planner_profile: string;
  mechanical_profile: string;
  reasoning_profile: string;
  agentic_profile: string;
  final_reviewer_profile: string;
  semantic_reviser_profile: string;
  check_repair_profile: string;
  semantic_revision_enabled: boolean;
  max_check_repair_attempts: number;
  max_correction_cycles: number;
  decomposition: string;
  execution_mode_policy: string;
  single_step_max_mutable_paths: number;
  staged_step_max_mutable_paths: number;
  repair_scope_policy: string;
  repair_scope_max_added_paths: number;
}

export interface RunFormState extends RunFormDefaults {
  spec: string;
  run_id: string;
}

const record = (value: unknown): JsonObject => (
  typeof value === 'object' && value !== null && !Array.isArray(value) ? value as JsonObject : {}
);

export function parseProfiles(value: unknown): { profiles: ModelProfile[]; defaults: JsonObject } {
  const data = record(value);
  const profiles = Array.isArray(data.profiles) ? data.profiles.flatMap((entry) => {
    const profile = record(entry);
    if (typeof profile.id !== 'string' || typeof profile.display_name !== 'string' || !Array.isArray(profile.roles)) return [];
    return [{ ...profile, id: profile.id, display_name: profile.display_name, roles: profile.roles.filter((role): role is string => typeof role === 'string') }];
  }) : [];
  return { profiles, defaults: record(data.defaults) };
}

export function profilesForRole(profiles: ModelProfile[], role: string): ModelProfile[] {
  return profiles.filter((profile) => profile.roles.includes(role));
}

function value<T>(candidate: unknown, fallback: T): T {
  return (candidate === undefined || candidate === null ? fallback : candidate) as T;
}

export function defaultsFromServer(profilesResponse: unknown, configResponse: unknown): RunFormDefaults {
  const { defaults, profiles } = parseProfiles(profilesResponse);
  const config = record(configResponse);
  const planning = record(config.planning);
  const revision = record(config.revision);
  const routing = record(config.routing);
  const profileDefault = (key: string, aliases: string[] = []): string => {
    for (const name of [key, ...aliases]) {
      if (typeof defaults[name] === 'string') return defaults[name] as string;
    }
    return '';
  };
  const fallbackProfile = (role: string): string => profiles.find((profile) => profile.roles.includes(role))?.id ?? '';
  return {
    planner_profile: profileDefault('planner_profile', ['planner']) || fallbackProfile('planner'),
    mechanical_profile: profileDefault('mechanical', ['mechanical_profile']) || (typeof routing.mechanical_profile === 'string' ? routing.mechanical_profile : fallbackProfile('implementer')),
    reasoning_profile: profileDefault('reasoning') || (typeof routing.reasoning_profile === 'string' ? routing.reasoning_profile : fallbackProfile('implementer')),
    agentic_profile: profileDefault('agentic') || (typeof routing.agentic_profile === 'string' ? routing.agentic_profile : fallbackProfile('implementer')),
    final_reviewer_profile: profileDefault('final_reviewer_profile') || fallbackProfile('reviewer'),
    semantic_reviser_profile: profileDefault('semantic_reviser_profile'),
    check_repair_profile: profileDefault('check_repair_profile'),
    semantic_revision_enabled: value(revision.enabled, false),
    max_check_repair_attempts: value(revision.max_check_repair_attempts, 0),
    max_correction_cycles: value(revision.max_correction_cycles, 0),
    decomposition: value(planning.decomposition, 'aggressive'),
    execution_mode_policy: value(planning.execution_mode_policy, 'auto'),
    single_step_max_mutable_paths: value(planning.single_step_max_mutable_paths, 2),
    staged_step_max_mutable_paths: value(planning.staged_step_max_mutable_paths, 5),
    repair_scope_policy: 'auto-bounded',
    repair_scope_max_added_paths: 4,
  };
}

export function validateRunForm(form: RunFormState, profiles: ModelProfile[]): string | undefined {
  if (!form.spec.trim()) return 'SPEC must contain non-whitespace text.';
  if (new TextEncoder().encode(form.spec).byteLength > MAX_SPEC_BYTES) return 'SPEC must be at most 48 KiB in UTF-8.';
  if (form.run_id && !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(form.run_id)) return 'Run ID may contain only letters, numbers, dot, underscore, and hyphen.';
  const roles: Array<[keyof RunFormDefaults, string, boolean]> = [
    ['planner_profile', 'planner', true], ['mechanical_profile', 'implementer', true],
    ['reasoning_profile', 'implementer', true], ['agentic_profile', 'implementer', true],
    ['final_reviewer_profile', 'reviewer', true], ['semantic_reviser_profile', 'reviser', false],
    ['check_repair_profile', 'repair', false],
  ];
  for (const [field, role, required] of roles) {
    const id = form[field];
    if (!id && !required) continue;
    if (!profiles.some((profile) => profile.id === id && profile.roles.includes(role))) return `${field.replaceAll('_', ' ')} must use a compatible ${role} profile.`;
  }
  const zeroAllowed = [form.max_check_repair_attempts, form.max_correction_cycles];
  const positive = [form.single_step_max_mutable_paths, form.staged_step_max_mutable_paths, form.repair_scope_max_added_paths];
  if (zeroAllowed.some((number) => !Number.isInteger(number) || number < 0)) return 'Repair budgets must be integers greater than or equal to zero.';
  if (positive.some((number) => !Number.isInteger(number) || number < 1)) return 'Mutable path limits must be positive integers.';
  return undefined;
}

export function buildCreateRunInput(form: RunFormState, advanced: boolean): CreateRunInput {
  const input: CreateRunInput = { spec: form.spec };
  if (form.run_id.trim()) input.run_id = form.run_id.trim();
  for (const key of ['planner_profile', 'mechanical_profile', 'reasoning_profile', 'agentic_profile', 'final_reviewer_profile', 'semantic_reviser_profile', 'check_repair_profile'] as const) {
    if (form[key]) input[key] = form[key];
  }
  if (advanced) {
    input.semantic_revision_enabled = form.semantic_revision_enabled;
    input.max_check_repair_attempts = Number(form.max_check_repair_attempts);
    input.max_correction_cycles = Number(form.max_correction_cycles);
    input.decomposition = form.decomposition;
    input.execution_mode_policy = form.execution_mode_policy;
    input.single_step_max_mutable_paths = Number(form.single_step_max_mutable_paths);
    input.staged_step_max_mutable_paths = Number(form.staged_step_max_mutable_paths);
    input.repair_scope_policy = form.repair_scope_policy;
    input.repair_scope_max_added_paths = Number(form.repair_scope_max_added_paths);
  }
  return input;
}
