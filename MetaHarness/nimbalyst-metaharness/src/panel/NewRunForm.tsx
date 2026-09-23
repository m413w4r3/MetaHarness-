import { useEffect, useRef, useState } from 'react';
import type { MetaHarnessSettingsData } from '../settings/MetaHarnessSettings';
import { buildCreateRunInput, defaultsFromServer, parseProfiles, profilesForRole, validateRunForm, type ModelProfile, type RunFormState } from '../model/runForm';
import { isBackendFailure } from '../contract';

type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type ObjectValue = Record<string, unknown>;
const object = (value: unknown): ObjectValue => typeof value === 'object' && value !== null ? value as ObjectValue : {};

function unwrap(value: unknown): unknown {
  if (isBackendFailure(value)) {
    const { error } = value;
    const caught = new Error(typeof error.message === 'string' ? error.message : 'MetaHarness request failed.') as Error & { httpStatus?: number; code?: string };
    caught.httpStatus = typeof error.httpStatus === 'number' ? error.httpStatus : undefined;
    caught.code = typeof error.code === 'string' ? error.code : undefined;
    throw caught;
  }
  return value;
}

function errorMessage(error: unknown): string {
  const caught = error as Error & { httpStatus?: number; code?: string };
  const detail = `${caught?.code ?? ''} ${caught?.message ?? ''}`.toLowerCase();
  if (caught?.httpStatus === 409 || detail.includes('capacity') || detail.includes('collision') || detail.includes('already exists') || detail.includes('maximum active runs')) {
    return 'MetaHarness cannot create this run because the active-run capacity is full or this Run ID already exists. Choose another Run ID or wait for an active run to finish.';
  }
  return caught?.message || 'Unexpected MetaHarness error.';
}

const PROFILE_FIELDS: Array<{ key: keyof RunFormState; label: string; role: string; required?: boolean }> = [
  { key: 'planner_profile', label: 'Planner', role: 'planner', required: true },
  { key: 'mechanical_profile', label: 'Mechanical', role: 'implementer', required: true },
  { key: 'reasoning_profile', label: 'Reasoning', role: 'implementer', required: true },
  { key: 'agentic_profile', label: 'Agentic', role: 'implementer', required: true },
  { key: 'final_reviewer_profile', label: 'Final reviewer', role: 'reviewer', required: true },
  { key: 'semantic_reviser_profile', label: 'Semantic reviser', role: 'reviser' },
  { key: 'check_repair_profile', label: 'Check repair', role: 'repair' },
];

export function NewRunForm({ callBackendTool, settings, onBack, onCreated }: {
  callBackendTool?: BackendCall;
  settings: MetaHarnessSettingsData;
  onBack: () => void;
  onCreated: (runId: string) => void;
}) {
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [defaults, setDefaults] = useState<RunFormState>();
  const [form, setForm] = useState<RunFormState>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const submitting = useRef(false);
  const [submittingState, setSubmittingState] = useState(false);

  useEffect(() => {
    let active = true;
    if (!callBackendTool) {
      setError('MetaHarness backend is unavailable.');
      setLoading(false);
      return () => { active = false; };
    }
    setLoading(true);
    void Promise.all([
      callBackendTool('metaharness.model_profiles'),
      callBackendTool('metaharness.get_config', { settings }),
    ]).then(([profileResult, configResult]) => {
      if (!active) return;
      const parsed = parseProfiles(unwrap(profileResult));
      const initial = { spec: '', run_id: '', ...defaultsFromServer(profileResult, unwrap(configResult)) };
      setProfiles(parsed.profiles);
      setDefaults(initial);
      setForm(initial);
      setError('');
    }).catch((caught: unknown) => {
      if (active) setError(errorMessage(caught));
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [callBackendTool, settings]);

  if (loading) return <section className="metaharness-dashboard"><button className="metaharness-link-button" type="button" onClick={onBack}>← All runs</button><h1>New Run</h1><p className="metaharness-muted" role="status">Loading MetaHarness defaults…</p></section>;

  const update = (key: keyof RunFormState, value: string | boolean | number) => {
    setForm((current) => current ? { ...current, [key]: value } : current);
    setError('');
  };
  const reset = () => {
    if (!defaults) return;
    setForm({ ...defaults, spec: '', run_id: '' });
    setError('');
  };
  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting.current || !form || !callBackendTool) return;
    const validation = validateRunForm(form, profiles);
    if (validation) { setError(validation); return; }
    submitting.current = true;
    setSubmittingState(true);
    setError('');
    try {
      const result = object(unwrap(await callBackendTool('metaharness.create_run', buildCreateRunInput(form, advanced))));
      const runId = typeof result.run_id === 'string' ? result.run_id : typeof result.id === 'string' ? result.id : form.run_id.trim();
      if (!runId) throw new Error('MetaHarness created a run but did not return its Run ID.');
      onCreated(runId);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      submitting.current = false;
      setSubmittingState(false);
    }
  };

  if (!form) return <section className="metaharness-dashboard"><button className="metaharness-link-button" type="button" onClick={onBack}>← All runs</button><h1>New Run</h1><p className="metaharness-error" role="alert">{error || 'Could not load MetaHarness defaults.'}</p></section>;

  const profileOptions = (role: string) => profilesForRole(profiles, role);
  const field = (key: keyof RunFormState, label: string, type: 'number' | 'text' = 'number') => <label className="metaharness-form__field" key={key}>
    <span>{label}</span><input type={type} min={type === 'number' ? (key === 'max_check_repair_attempts' || key === 'max_review_repair_cycles' ? 0 : 1) : undefined} step={type === 'number' ? 1 : undefined} value={form[key] as string | number} onChange={(event) => update(key, type === 'number' ? (event.target.value === '' ? Number.NaN : Number(event.target.value)) : event.target.value)} />
  </label>;

  return <section className="metaharness-dashboard" aria-labelledby="metaharness-new-run-title">
    <button className="metaharness-link-button" type="button" onClick={onBack}>← All runs</button>
    <h1 id="metaharness-new-run-title">New Run</h1>
    <form className="metaharness-form" noValidate onSubmit={(event) => void submit(event)}>
      <label className="metaharness-form__field"><span>SPEC <b>*</b></span><textarea autoFocus required rows={12} value={form.spec} onChange={(event) => update('spec', event.target.value)} aria-describedby="metaharness-spec-size" /></label>
      <div id="metaharness-spec-size" className="metaharness-form__hint">{new TextEncoder().encode(form.spec).byteLength.toLocaleString()} / 49,152 UTF-8 bytes</div>
      <div className="metaharness-form__profiles">
        {PROFILE_FIELDS.map(({ key, label, role, required }) => <label className="metaharness-form__field" key={key}>
          <span>{label}</span><select required={required} value={form[key] as string} onChange={(event) => update(key, event.target.value)}>
            {!required && <option value="">MetaHarness default</option>}
            {profileOptions(role).map((profile) => <option key={profile.id} value={profile.id}>{profile.display_name} · {profile.id}</option>)}
          </select>
        </label>)}
      </div>
      <section className="metaharness-form__advanced">
        <button className="metaharness-form__advanced-toggle" type="button" aria-expanded={advanced} onClick={() => setAdvanced((open) => !open)}>Advanced</button>
        {advanced && <div className="metaharness-form__advanced-content">
          {field('run_id', 'Run ID (optional)', 'text')}
          <label className="metaharness-form__field"><span>Decomposition</span><select value={form.decomposition} onChange={(event) => update('decomposition', event.target.value)}><option value="aggressive">aggressive</option><option value="balanced">balanced</option></select></label>
          <label className="metaharness-form__field"><span>Execution mode policy</span><select value={form.execution_mode_policy} onChange={(event) => update('execution_mode_policy', event.target.value)}><option value="auto">auto</option><option value="require-staged">staged (require-staged)</option></select></label>
          <label className="metaharness-settings__checkbox"><input type="checkbox" checked={form.semantic_revision_enabled} onChange={(event) => update('semantic_revision_enabled', event.target.checked)} />Semantic revision</label>
          {field('max_check_repair_attempts', 'Max check repair attempts')}
          {field('max_review_repair_cycles', 'Max review repair cycles')}
          {field('single_step_max_mutable_paths', 'Single-step max mutable paths')}
          {field('staged_step_max_mutable_paths', 'Staged-step max mutable paths')}
          <label className="metaharness-form__field"><span>Repair scope policy</span><select value={form.repair_scope_policy} onChange={(event) => update('repair_scope_policy', event.target.value)}><option value="auto-bounded">auto-bounded</option><option value="require-approval">require-approval</option><option value="deny-expansion">deny-expansion</option></select></label>
          {field('repair_scope_max_added_paths', 'Repair scope max added paths')}
        </div>}
      </section>
      {error && <p className="metaharness-error" role="alert">{error}</p>}
      <div className="metaharness-form__actions"><button className="metaharness-secondary-button" type="button" onClick={reset}>Reset to MetaHarness defaults</button><button className="metaharness-button" type="submit" disabled={submittingState}>{submittingState ? 'Creating…' : 'Create Run'}</button></div>
    </form>
  </section>;
}
