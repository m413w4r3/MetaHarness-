import type { RunDetail } from '../../types';

type Data = Record<string, unknown>;

function object(value: unknown): Data {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Data : {};
}

function firstObject(...values: unknown[]): Data {
  for (const value of values) {
    const result = object(value);
    if (Object.keys(result).length) return result;
  }
  return {};
}

export interface RunActions {
  canResume: boolean;
  resumeLabel?: string;
  canRecoverPlan: boolean;
  canApprovePlan: boolean;
  canApproveScope: boolean;
  canCancel: boolean;
}

/** Derive UI actions only from the authority and capabilities returned by MetaHarness. */
export function deriveRunActions(run: RunDetail): RunActions {
  const data = object(run);
  const overview = object(data.overview);
  const resume = firstObject(data.resume_info, overview.resume, data.resume);
  const capabilities = firstObject(data.capabilities, overview.capabilities);
  const approval = firstObject(data.approval_info, data.approval);
  const scopeApproval = firstObject(data.scope_approval, approval.scope_approval);
  const planRecovery = object(data.plan_recovery);
  const status = typeof data.status === 'string' ? data.status.toLowerCase() : '';
  // MetaHarness accepts a plan decision only in `awaiting_plan_approval` and a
  // scope decision only in `waiting_scope_approval` (web/api.py). A resumable
  // `plan_approval` phase is a Resume action, not an approval form.
  const planPending = status === 'awaiting_plan_approval';
  const scopePending = status === 'waiting_scope_approval';
  const hasScopeDelta = Object.keys(object(data.scope_delta)).length > 0;
  const resumeLabel = typeof resume.label === 'string' && resume.label.trim()
    ? resume.label
    : typeof data.resume_label === 'string' && data.resume_label.trim() ? data.resume_label : undefined;

  return {
    canResume: resume.resumable === true && capabilities.resume !== false,
    ...(resumeLabel ? { resumeLabel } : {}),
    canRecoverPlan: planRecovery.eligible === true && capabilities.recover_plan !== false,
    canApprovePlan: planPending && approval.recorded !== true && approval.awaiting !== false
      && capabilities.plan_approval !== false,
    canApproveScope: scopePending && hasScopeDelta && scopeApproval.recorded !== true && scopeApproval.awaiting !== false
      && capabilities.scope_approval !== false,
    canCancel: capabilities.cancel === true,
  };
}
