import { useEffect, useState } from 'react';
import { parseProfiles, type ModelProfile } from '../../model/runForm';

type Data = Record<string, unknown>;
type BackendCall = (toolName: string, args?: Record<string, unknown>) => Promise<unknown>;
type Step = { id: string; title: string; execution_class: string; profile_id?: string };

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function awaitingPlan(data: Data): boolean {
  const approval = object(data.approval);
  const overview = object(data.overview);
  const resume = object(overview.resume);
  const status = text(data.status)?.toLowerCase();
  const explicit = status === 'awaiting_plan_approval'
    || (resume.resumable === true && resume.phase === 'plan_approval');
  return explicit && approval.recorded !== true && approval.awaiting !== false;
}

function awaitingScope(data: Data): boolean {
  const scopeApproval = object(data.scope_approval);
  const approval = object(data.approval);
  const overview = object(data.overview);
  const resume = object(overview.resume);
  const status = text(data.status)?.toLowerCase();
  const explicit = status === 'waiting_scope_approval'
    || (resume.resumable === true && resume.phase === 'scope_approval')
    || scopeApproval.awaiting === true || approval.scope_awaiting === true;
  return explicit && Object.keys(object(data.scope_delta)).length > 0
    && scopeApproval.recorded !== true && scopeApproval.awaiting !== false;
}

function stepsFrom(data: Data): Step[] {
  const bundle = object(data.implementation_bundle);
  return Array.isArray(bundle.steps) ? bundle.steps.flatMap((value) => {
    const step = object(value);
    if (typeof step.id !== 'string' || typeof step.title !== 'string') return [];
    return [{
      id: step.id,
      title: step.title,
      execution_class: text(step.execution_class)?.toUpperCase() ?? 'MECHANICAL',
      profile_id: text(step.profile_id) ?? text(step.recommended_profile),
    }];
  }) : [];
}

function selectedStepProfiles(data: Data): Record<string, string> {
  const selection = object(data.execution_selection);
  const byId = new Map<string, string>();
  if (Array.isArray(selection.steps)) {
    for (const value of selection.steps) {
      const step = object(value);
      const implementer = object(step.implementer);
      if (typeof step.step_id === 'string' && typeof implementer.profile_id === 'string') {
        byId.set(step.step_id, implementer.profile_id);
      }
    }
  }
  return Object.fromEntries(byId);
}

function profileAllowed(profile: ModelProfile, executionClass: string): boolean {
  if (!profile.roles.includes('implementer')) return false;
  const classes = (profile as ModelProfile & { execution_classes?: unknown; classes?: unknown }).execution_classes
    ?? (profile as ModelProfile & { classes?: unknown }).classes;
  return !Array.isArray(classes) || classes.length === 0
    || classes.some((value) => typeof value === 'string' && value.toUpperCase() === executionClass);
}

function errorFrom(value: unknown): string {
  const result = object(value);
  const error = object(result.error);
  return text(error.message) ?? 'MetaHarness backend call failed.';
}

function requireOk(value: unknown): unknown {
  if (object(value).ok === false) throw new Error(errorFrom(value));
  return value;
}

export function RunApproval({ runId, data, callBackendTool, disabled, onDecision, canApprovePlan, canApproveScope }: {
  runId: string;
  data: Data;
  callBackendTool?: BackendCall;
  disabled: boolean;
  onDecision: (kind: 'plan' | 'scope', decision: 'APPROVE' | 'REJECT', input?: Data) => Promise<void>;
  canApprovePlan?: boolean;
  canApproveScope?: boolean;
}) {
  const planPending = canApprovePlan ?? awaitingPlan(data);
  const scopePending = canApproveScope ?? awaitingScope(data);
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [profileError, setProfileError] = useState('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [rejecting, setRejecting] = useState(false);
  const steps = stepsFrom(data);
  const runOptions = object(data.run_options);
  const pipeline = object(runOptions.pipeline);
  const semanticEnabled = pipeline.semantic_revision_enabled === true;
  const repairEnabled = Number(pipeline.max_check_repair_attempts ?? 0) > 0
    || Number(pipeline.max_review_repair_cycles ?? 0) > 0;

  useEffect(() => {
    if (!planPending || !callBackendTool) return;
    let active = true;
    void callBackendTool('metaharness.model_profiles').then((raw) => {
      const value = requireOk(raw);
      if (!active) return;
      const parsed = parseProfiles(value);
      setProfiles(parsed.profiles);
      const selection = selectedStepProfiles(data);
      const next: Record<string, string> = {};
      const defaults = parsed.defaults;
      const requestedProfiles = object(object(data.run_options).profiles);
      const roleDefault = (key: string, role: string) => {
        const requested = text(requestedProfiles[key]) ?? text(defaults[key]);
        return (requested && parsed.profiles.some((profile) => profile.id === requested && profile.roles.includes(role)))
          ? requested : parsed.profiles.find((profile) => profile.roles.includes(role))?.id ?? '';
      };
      next.final_reviewer_profile = roleDefault('final_reviewer_profile', 'reviewer');
      next.semantic_reviser_profile = roleDefault('semantic_reviser_profile', 'reviser');
      next.check_repair_profile = roleDefault('check_repair_profile', 'repair');
      for (const step of steps) {
        const options = parsed.profiles.filter((profile) => profileAllowed(profile, step.execution_class));
        const route = step.execution_class === 'REASONING' ? 'reasoning_profile'
          : step.execution_class === 'AGENTIC' ? 'agentic_profile' : 'mechanical_profile';
        const selected = selection[step.id] ?? step.profile_id ?? text(requestedProfiles[route]);
        next[`step:${step.id}`] = options.some((profile) => profile.id === selected)
          ? selected! : options[0]?.id ?? '';
      }
      setValues(next);
    }).catch((error: unknown) => {
      if (active) setProfileError(error instanceof Error ? error.message : 'Unable to load MetaHarness profiles.');
    });
    return () => { active = false; };
  }, [callBackendTool, planPending, runId]);

  if (!planPending && !scopePending) return null;

  const compatible = (role: string) => profiles.filter((profile) => profile.roles.includes(role));
  const change = (key: string, value: string) => setValues((current) => ({ ...current, [key]: value }));
  const submitApprove = () => {
    if (planPending) {
      const stepProfiles = Object.fromEntries(steps.map((step) => [step.id, values[`step:${step.id}`] ?? '']));
      const reviewerValid = profiles.some((profile) => profile.id === values.final_reviewer_profile && profile.roles.includes('reviewer'));
      const reviserValid = !values.semantic_reviser_profile || profiles.some((profile) => profile.id === values.semantic_reviser_profile && profile.roles.includes('reviser'));
      const repairValid = !values.check_repair_profile || profiles.some((profile) => profile.id === values.check_repair_profile && profile.roles.includes('repair'));
      if (!reviewerValid || !reviserValid || !repairValid || !steps.length || steps.some((step) => !stepProfiles[step.id]
        || !profiles.some((profile) => profile.id === stepProfiles[step.id] && profileAllowed(profile, step.execution_class)))) return;
      const input: Data = { decision: 'APPROVE', final_reviewer_profile: values.final_reviewer_profile, step_profiles: stepProfiles };
      if (semanticEnabled && values.semantic_reviser_profile) input.semantic_reviser_profile = values.semantic_reviser_profile;
      if (repairEnabled && values.check_repair_profile) input.check_repair_profile = values.check_repair_profile;
      void onDecision('plan', 'APPROVE', input);
    } else {
      void onDecision('scope', 'APPROVE', { decision: 'APPROVE' });
    }
  };
  const delta = object(data.scope_delta);
  const addedPaths = Array.isArray(delta.added_paths) ? delta.added_paths.filter((path): path is string => typeof path === 'string') : [];

  return <section className="metaharness-approval" aria-labelledby="metaharness-approval-title">
    {planPending && <>
      <h2 id="metaharness-approval-title">PLAN APPROVAL REQUIRED</h2>
      {profileError && <p role="alert" className="metaharness-error">{profileError}</p>}
      {!profileError && profiles.length === 0 && <p className="metaharness-muted" role="status">Loading MetaHarness profiles…</p>}
      {profiles.length > 0 && <>
        <div className="metaharness-approval__roles">
          {([['final_reviewer_profile', 'Final reviewer', 'reviewer'], ['semantic_reviser_profile', 'Semantic reviser', 'reviser'], ['check_repair_profile', 'Check repair', 'repair']] as const).map(([key, label, role]) => <label key={key}>
            <span>{label}</span><select aria-label={label} value={values[key] ?? ''} disabled={disabled || (key === 'semantic_reviser_profile' && !semanticEnabled) || (key === 'check_repair_profile' && !repairEnabled)} onChange={(event) => change(key, event.target.value)}>
              {compatible(role).map((profile) => <option key={profile.id} value={profile.id}>{profile.id}</option>)}
            </select>
          </label>)}
        </div>
        <h3>Implementation steps</h3>
        <div className="metaharness-approval__steps">{steps.map((step) => {
          const options = profiles.filter((profile) => profileAllowed(profile, step.execution_class));
          return <label key={step.id}>
            <span><strong>{step.id}</strong> {step.title}</span>
            <span className="metaharness-approval__profile"><span>Profile</span><select aria-label={`${step.id} Profile`} value={values[`step:${step.id}`] ?? ''} disabled={disabled} onChange={(event) => change(`step:${step.id}`, event.target.value)}>
              {options.map((profile) => <option key={profile.id} value={profile.id}>{profile.id}</option>)}
            </select></span>
          </label>;
        })}</div>
      </>}
      <div className="metaharness-approval__actions">
        <button type="button" className="metaharness-button metaharness-button--danger" disabled={disabled || !callBackendTool} onClick={() => setRejecting(true)}>REJECT PLAN</button>
        <button type="button" className="metaharness-button" disabled={disabled || !callBackendTool || profiles.length === 0 || !steps.length || !!profileError} onClick={submitApprove}>APPROVE &amp; CONTINUE</button>
      </div>
    </>}
    {scopePending && <>
      <h2>REPAIR SCOPE EXPANSION</h2>
      <h3>Added paths</h3>
      {addedPaths.length ? <ul>{addedPaths.map((path) => <li key={path}><code>+ {path}</code></li>)}</ul> : <p className="metaharness-muted">No added paths.</p>}
      <div className="metaharness-approval__actions">
        <button type="button" className="metaharness-button metaharness-button--danger" disabled={disabled || !callBackendTool} onClick={() => setRejecting(true)}>REJECT</button>
        <button type="button" className="metaharness-button" disabled={disabled || !callBackendTool} onClick={submitApprove}>APPROVE</button>
      </div>
    </>}
    {rejecting && <div className="metaharness-approval__scrim"><section role="dialog" aria-modal="true" aria-labelledby="metaharness-reject-title" className="metaharness-approval__dialog">
      <h3 id="metaharness-reject-title">Reject this decision?</h3>
      <p>This action is irreversible. The run will record the rejection.</p>
      <div className="metaharness-approval__actions">
        <button type="button" className="metaharness-secondary-button" disabled={disabled} onClick={() => setRejecting(false)}>Cancel</button>
        <button type="button" className="metaharness-button metaharness-button--danger" disabled={disabled} onClick={() => {
          setRejecting(false);
          void onDecision(planPending ? 'plan' : 'scope', 'REJECT', { decision: 'REJECT' });
        }}>Confirm irreversible rejection</button>
      </div>
    </section></div>}
  </section>;
}
