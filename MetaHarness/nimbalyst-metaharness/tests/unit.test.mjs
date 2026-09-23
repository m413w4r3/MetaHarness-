import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseProfiles, profilesForRole, validateRunForm } from '../src/model/runForm.ts';
import { classifyRunStatus } from '../src/panel/runStatus.ts';
import { deriveRunActions } from '../src/panel/run/runActions.ts';
import { runDetailPollInterval } from '../src/panel/run/RunDetail.tsx';
import { isBackendFailure, isRunSummary, isProgressResponse, isMutationAccepted, isHealthResponse } from '../src/contract.ts';
import { shortSha } from '../src/panel/run/RunHeader.tsx';
import { workspaceFile } from '../src/panel/run/RunArtifactViews.tsx';

test('API profile decoder accepts known fields and ignores malformed entries', () => {
  const decoded = parseProfiles({
    profiles: [
      { id: 'planner-1', display_name: 'Planner', roles: ['planner', 7] },
      { id: 'bad', display_name: 'Bad', roles: 'reviewer' },
      null,
    ],
    defaults: { planner_profile: 'planner-1' },
  });
  assert.deepEqual(decoded.profiles, [{ id: 'planner-1', display_name: 'Planner', roles: ['planner'] }]);
  assert.equal(decoded.defaults.planner_profile, 'planner-1');
  assert.deepEqual(parseProfiles(null), { profiles: [], defaults: {} });
});

test('run status categories cover active, awaiting, terminal, and unknown API values', () => {
  assert.equal(classifyRunStatus('implementing'), 'active');
  assert.equal(classifyRunStatus('awaiting_plan_approval'), 'awaiting-action');
  assert.equal(classifyRunStatus('published'), 'completed');
  assert.equal(classifyRunStatus('failed'), 'failed');
  assert.equal(classifyRunStatus('future_status'), 'other');
});

test('run actions require server authority and honor disabled capabilities', () => {
  assert.deepEqual(deriveRunActions({
    status: 'failed', resume_info: { resumable: true, label: 'Retry S02' },
    plan_recovery: { eligible: true }, capabilities: { resume: false, recover_plan: true, cancel: true },
  }), {
    canResume: false, resumeLabel: 'Retry S02', canRecoverPlan: true,
    canApprovePlan: false, canApproveScope: false, canCancel: true,
  });
  assert.equal(deriveRunActions({ status: 'awaiting_plan_approval' }).canApprovePlan, true);
});

test('profile filtering returns only profiles that declare the requested role', () => {
  const profiles = parseProfiles({ profiles: [
    { id: 'p', display_name: 'Planner', roles: ['planner'] },
    { id: 'i', display_name: 'Implementer', roles: ['implementer', 'repair'] },
  ] }).profiles;
  assert.deepEqual(profilesForRole(profiles, 'repair').map(({ id }) => id), ['i']);
  assert.deepEqual(profilesForRole(profiles, 'reviewer'), []);
});

test('form validation reports SPEC, Run ID, profile, and numeric errors', () => {
  const roles = ['planner', 'implementer', 'reviewer'].map((role) => ({ id: role, display_name: role, roles: [role] }));
  const valid = {
    spec: 'Add a search page', run_id: '', planner_profile: 'planner', mechanical_profile: 'implementer',
    reasoning_profile: 'implementer', agentic_profile: 'implementer', final_reviewer_profile: 'reviewer',
    semantic_reviser_profile: '', check_repair_profile: '', semantic_revision_enabled: false,
    max_check_repair_attempts: 0, max_review_repair_cycles: 0, decomposition: 'balanced',
    execution_mode_policy: 'auto', single_step_max_mutable_paths: 2, staged_step_max_mutable_paths: 5,
    repair_scope_policy: 'auto-bounded', repair_scope_max_added_paths: 4,
  };
  assert.equal(validateRunForm({ ...valid, spec: '  ' }, roles), 'SPEC must contain non-whitespace text.');
  assert.match(validateRunForm({ ...valid, run_id: '../x' }, roles), /Run ID/);
  assert.match(validateRunForm({ ...valid, planner_profile: 'reviewer' }, roles), /compatible planner profile/);
  assert.match(validateRunForm({ ...valid, max_review_repair_cycles: -1 }, roles), /Repair budgets/);
  assert.equal(validateRunForm(valid, roles), undefined);
});

test('short SHA and workspace file resolution preserve display and reject escapes', () => {
  assert.equal(shortSha('123456789abcdef'), '1234567');
  assert.equal(shortSha(null), '');
  assert.equal(workspaceFile('/workspace/project', 'src/../README.md'), '/workspace/project/README.md');
  for (const input of ['../outside', '/etc/passwd', '\\server\\share', 'C:\\Windows\\system.ini', '../../outside']) {
    assert.equal(workspaceFile('/workspace/project', input), undefined, input);
  }
});

test('contract decoders match the MetaHarness JSON and keep doctor reports distinct from failures', () => {
  assert.equal(isHealthResponse({ service: 'metaharness', api_version: 1, status: 'ok', control_api: true }), true);
  assert.equal(isHealthResponse({ service: 'other', api_version: 1, status: 'ok' }), false);
  assert.equal(isRunSummary({ run_id: 'r', status: 'planning', updated_at: null, plan_title: null, commit_sha: null, candidate: null, failure: null }), true);
  assert.equal(isRunSummary({ status: 'planning' }), false);
  assert.equal(isRunSummary({ run_id: 'r', status: 7 }), false);
  assert.equal(isProgressResponse({ next_offset: 4, events: ['a'] }), true);
  assert.equal(isProgressResponse({ next_offset: 1.5, events: [] }), false);
  assert.equal(isMutationAccepted({ ok: true, run_id: 'r', location: '/runs/r', accepted: true }), true);
  assert.equal(isMutationAccepted({ run_id: 'r' }), false);
  assert.equal(isBackendFailure({ ok: false, error: { code: 'HTTP_ERROR', message: 'x' } }), true);
  assert.equal(isBackendFailure({ ok: false, checks: [], summary: { failed: 1 } }), false);
});

test('run detail polling slows while awaiting a human and stops once terminal', () => {
  assert.equal(runDetailPollInterval('implementing', 1000), 1000);
  assert.equal(runDetailPollInterval('approved', 1000), 1000);
  assert.equal(runDetailPollInterval('awaiting_plan_approval', 1000), 5000);
  assert.equal(runDetailPollInterval('committed', 1000), undefined);
  assert.equal(runDetailPollInterval('failed', 1000), undefined);
  assert.equal(runDetailPollInterval(undefined, 1000), 1000);
});

test('approval actions require the exact server status that accepts the decision', () => {
  const resumablePlanPhase = { status: 'interrupted', overview: { resume: { resumable: true, phase: 'plan_approval' } } };
  assert.equal(deriveRunActions(resumablePlanPhase).canApprovePlan, false);
  assert.equal(deriveRunActions(resumablePlanPhase).canResume, true);
  assert.equal(deriveRunActions({ status: 'awaiting_plan_approval', approval: { recorded: false } }).canApprovePlan, true);
  assert.equal(deriveRunActions({ status: 'failed', scope_delta: { added_paths: ['a'] }, scope_approval: { awaiting: true } }).canApproveScope, false);
});
