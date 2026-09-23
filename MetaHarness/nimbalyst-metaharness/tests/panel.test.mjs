import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { classifyRunStatus } from '../src/panel/runStatus.ts';
import { RunsDashboard } from '../src/panel/RunsDashboard.tsx';
import { RunDetail } from '../src/panel/run/RunDetail.tsx';
import { NewRunForm } from '../src/panel/NewRunForm.tsx';
import { defaultsFromServer, validateRunForm, buildCreateRunInput } from '../src/model/runForm.ts';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { cleanup, fireEvent, render, screen, waitFor } = await import('@testing-library/react');

afterEach(() => cleanup());

const settings = { executable: 'metaharness', configPath: '/work/metaharness.toml', port: 8765, autoStart: false, pollIntervalMs: 1000 };
const modelProfiles = {
  profiles: [
    { id: 'plan-a', display_name: 'Planner A', roles: ['planner'], model: 'p1' },
    { id: 'impl-a', display_name: 'Implementer A', roles: ['implementer'] },
    { id: 'review-a', display_name: 'Reviewer A', roles: ['reviewer'] },
    { id: 'revise-a', display_name: 'Reviser A', roles: ['reviser'] },
    { id: 'repair-a', display_name: 'Repair A', roles: ['repair'] },
  ],
  defaults: { planner_profile: 'plan-a', mechanical: 'impl-a', reasoning: 'impl-a', agentic: 'impl-a', final_reviewer_profile: 'review-a', semantic_reviser_profile: 'revise-a', check_repair_profile: 'repair-a' },
};
const configResponse = { planning: { decomposition: 'balanced', execution_mode_policy: 'require-staged', single_step_max_mutable_paths: 3, staged_step_max_mutable_paths: 6 }, revision: { enabled: true, max_check_repair_attempts: 2, max_review_repair_cycles: 3 } };

function renderNewRun(callBackendTool, onCreated = () => {}) {
  return render(React.createElement(NewRunForm, { callBackendTool, settings, onBack: () => {}, onCreated }));
}

function newRunBackend(overrides = {}) {
  const calls = [];
  const callBackendTool = async (name, args) => {
    if (name === 'metaharness.model_profiles') return modelProfiles;
    if (name === 'metaharness.get_config') return configResponse;
    if (name === 'metaharness.create_run') return { run_id: 'created-001' };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  return { calls, callBackendTool: async (name, args) => {
    calls.push([name, args]);
    return overrides[name] ? overrides[name](args, calls) : callBackendTool(name, args);
  } };
}

function renderDashboard(callBackendTool, overrides = {}) {
  return render(React.createElement(RunsDashboard, {
    callBackendTool,
    settings,
    view: { kind: 'dashboard' },
    onViewChange: () => {},
    onOpenSettings: () => {},
    ...overrides,
  }));
}

function backend({ connected = true, configured = true, runs = [] } = {}) {
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.status') return { configured, connected };
    if (name === 'metaharness.list_runs') return runs;
    throw new Error(`Unexpected backend call: ${name}`);
  };
  return { callBackendTool, calls };
}

test('loading is shown while status is pending', async () => {
  let resolveStatus;
  const callBackendTool = (name) => name === 'metaharness.status'
    ? new Promise((resolve) => { resolveStatus = resolve; })
    : Promise.resolve([]);
  renderDashboard(callBackendTool);
  assert.ok(screen.getByRole('status').textContent.includes('Loading runs'));
  resolveStatus({ configured: true, connected: true });
  await waitFor(() => assert.ok(screen.getByText('No runs in this workspace yet.')));
});

test('disconnected configured workspace shows a reconnect message', async () => {
  const { callBackendTool, calls } = backend({ connected: false });
  renderDashboard(callBackendTool);
  await screen.findByText('MetaHarness is not connected. Check your configuration or try again.');
  assert.deepEqual(calls.map(([name]) => name), ['metaharness.status']);
});

test('autoStart explicitly starts the service then reloads status', async () => {
  const calls = [];
  let statusChecks = 0;
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.status') return { configured: true, connected: ++statusChecks > 1 };
    if (name === 'metaharness.start') return { connected: true };
    if (name === 'metaharness.list_runs') return [];
    throw new Error(`Unexpected backend call: ${name}`);
  };
  renderDashboard(callBackendTool, { settings: { ...settings, autoStart: true } });
  await screen.findByText('No runs in this workspace yet.');
  assert.deepEqual(calls.map(([name]) => name), [
    'metaharness.status', 'metaharness.start', 'metaharness.status', 'metaharness.list_runs',
  ]);
  assert.deepEqual(calls[1][1], { settings: { ...settings, autoStart: true } });
});

test('unconfigured workspace shows settings instead of a technical error', async () => {
  const { callBackendTool } = backend({ configured: false, connected: false });
  renderDashboard(callBackendTool);
  await screen.findByRole('heading', { name: 'Configure MetaHarness' });
  assert.equal(screen.queryByRole('alert'), null);
});

test('empty run list gets the empty state', async () => {
  const { callBackendTool } = backend();
  renderDashboard(callBackendTool);
  await screen.findByText('No runs in this workspace yet.');
});

test('active run is rendered with its exposed summary fields', async () => {
  const run = { run_id: 'example-014', status: 'implementing', plan_title: 'Implement user account search', updated_at: '2026-09-23T10:42:00Z' };
  const { callBackendTool } = backend({ runs: [run] });
  renderDashboard(callBackendTool);
  await screen.findByText('example-014');
  assert.ok(screen.getByRole('region', { name: 'Active' }));
  assert.ok(screen.getByText('Implement user account search'));
  assert.ok(screen.getByText('IMPLEMENTING', { exact: false }));
  assert.equal(document.querySelector('[role="progressbar"]'), null);
});

test('failed run shows its failure alongside the true status', async () => {
  const run = { run_id: 'example-013', status: 'failed', failure: { reason: 'review failed' } };
  const { callBackendTool } = backend({ runs: [run] });
  renderDashboard(callBackendTool);
  await screen.findByText('example-013');
  assert.ok(screen.getByRole('region', { name: 'Failed' }));
  assert.ok(screen.getByText('review failed'));
});

test('new status values classify as other and remain visible', async () => {
  assert.equal(classifyRunStatus('future_status'), 'other');
  assert.equal(classifyRunStatus('IMPLEMENTING'), 'active');
  const { callBackendTool } = backend({ runs: [{ run_id: 'future-run', status: 'future_status' }] });
  renderDashboard(callBackendTool);
  await screen.findByText('future-run');
  assert.ok(screen.getByRole('region', { name: 'Other' }));
  assert.ok(screen.getByText('future_status'));
});

const runFixtures = [
  ['planning', { status: 'planning', planner_raw: '# Draft plan', spec: 'Add search', implementation_bundle: null }],
  ['awaiting approval', { status: 'awaiting_approval', planner_raw: '# Proposed plan', implementation_contract: 'Contract text' }],
  ['implementing', { status: 'implementing', implementation_bundle: { steps: [{ id: 'S03', title: 'API search endpoint', execution_class: 'reasoning' }] }, cycle_artifacts: [{ number: 1, steps: [{ id: 'S03', title: 'API search endpoint', status: 'accepted', execution_class: 'reasoning', profile_id: 'codex-luna-xhigh', commit_sha: '8c12d77abcdef', changed_files: ['a', 'b', 'c', 'd'] }] }] }],
  ['checking', { status: 'checking', cycle_artifacts: [{ number: 1, steps: [{ id: 'S01', title: 'Check step', status: 'completed', checks: { passed: true } }] }] }],
  ['review', { status: 'review', implementation_bundle: { steps: [{ id: 'S02', title: 'Review step' }] }, cycle_artifacts: [{ number: 1, steps: [{ id: 'S02', title: 'Review step', status: 'completed' }] }] }],
  ['failed', { status: 'failed', failure: { reason: 'workspace setup failed' }, workspace_setup: null }],
  ['published', { status: 'published', publish: { mode: 'pull_request', status: 'published' }, commit_sha: '123456789abcdef' }],
];

for (const [name, payload] of runFixtures) {
  test(`run detail fixture renders ${name} from durable state`, async () => {
    const calls = [];
    const callBackendTool = async (tool, args) => {
      calls.push([tool, args]);
      if (tool === 'metaharness.get_run') return { run_id: `fixture-${name}`, updated_at: '2026-09-23T10:42:00Z', ...payload };
      throw new Error(`Unexpected backend call: ${tool}`);
    };
    render(React.createElement(RunDetail, { runId: `fixture-${name}`, callBackendTool, onBack: () => {} }));
    await screen.findByText(name === 'awaiting approval' ? 'awaiting_approval' : name, { exact: false });
    assert.deepEqual(calls[0], ['metaharness.get_run', { runId: `fixture-${name}` }]);
    assert.ok(screen.getByRole('button', { name: 'Refresh' }));
    if (name === 'implementing') {
      assert.ok(screen.getByText('API search endpoint'));
      assert.equal(screen.getByText('8c12d77').title, '8c12d77abcdef');
      assert.ok(screen.getByText('4'));
    }
    if (name === 'failed') assert.ok(screen.getByText('workspace setup failed'));
    if (name === 'planning') assert.ok(screen.getByText('# Draft plan'));
    if (name === 'published') assert.ok(screen.getByText('pull_request'));
  });
}

test('polling timer is cleared on unmount and overlapping polls are skipped', async () => {
  const originalSetInterval = window.setInterval;
  const originalClearInterval = window.clearInterval;
  let poll;
  let cleared;
  let releaseStatus;
  const calls = [];
  const callBackendTool = (name) => {
    calls.push(name);
    if (name === 'metaharness.status' && calls.filter((call) => call === name).length === 1) {
      return new Promise((resolve) => { releaseStatus = resolve; });
    }
    if (name === 'metaharness.status') return Promise.resolve({ configured: true, connected: true });
    return Promise.resolve([]);
  };
  window.setInterval = (callback, interval) => { poll = callback; assert.equal(interval, 1000); return 123; };
  window.clearInterval = (id) => { cleared = id; };
  try {
    const view = renderDashboard(callBackendTool);
    poll();
    assert.equal(calls.filter((call) => call === 'metaharness.status').length, 1);
    releaseStatus({ configured: true, connected: true });
    await waitFor(() => assert.ok(calls.includes('metaharness.list_runs')));
    view.unmount();
    assert.equal(cleared, 123);
  } finally {
    window.setInterval = originalSetInterval;
    window.clearInterval = originalClearInterval;
  }
});

test('server profile and config defaults populate the form reset state', async () => {
  const defaults = defaultsFromServer(modelProfiles, configResponse);
  assert.equal(defaults.decomposition, 'balanced');
  assert.equal(defaults.execution_mode_policy, 'require-staged');
  assert.equal(defaults.semantic_revision_enabled, true);
  assert.equal(defaults.max_review_repair_cycles, 3);
  const { callBackendTool, calls } = newRunBackend();
  renderNewRun(callBackendTool);
  await screen.findByRole('heading', { name: 'New Run' });
  assert.deepEqual(calls.slice(0, 2).map(([name]) => name), ['metaharness.model_profiles', 'metaharness.get_config']);
  assert.equal(screen.getByLabelText('Planner').value, 'plan-a');
  assert.equal(screen.getByLabelText('Mechanical').value, 'impl-a');
});

test('each profile selector contains only profiles with its required role', async () => {
  const { callBackendTool } = newRunBackend();
  renderNewRun(callBackendTool);
  await screen.findByRole('heading', { name: 'New Run' });
  assert.deepEqual([...screen.getByLabelText('Planner').options].map((option) => option.value), ['plan-a']);
  assert.deepEqual([...screen.getByLabelText('Mechanical').options].map((option) => option.value), ['impl-a']);
  assert.deepEqual([...screen.getByLabelText('Final reviewer').options].map((option) => option.value), ['review-a']);
  assert.deepEqual([...screen.getByLabelText('Semantic reviser').options].map((option) => option.value), ['', 'revise-a']);
  assert.deepEqual([...screen.getByLabelText('Check repair').options].map((option) => option.value), ['', 'repair-a']);
});

test('invalid blank spec is rejected before create_run', async () => {
  const { callBackendTool, calls } = newRunBackend();
  renderNewRun(callBackendTool);
  await screen.findByRole('heading', { name: 'New Run' });
  fireEvent.click(screen.getByRole('button', { name: 'Create Run' }));
  assert.equal(calls.some(([name]) => name === 'metaharness.create_run'), false);
  assert.ok(screen.getByRole('alert').textContent.includes('SPEC'));
  const defaults = defaultsFromServer(modelProfiles, configResponse);
  assert.ok(validateRunForm({ spec: '  ', run_id: '', ...defaults }, modelProfiles));
  assert.ok(validateRunForm({ spec: 'é'.repeat(24 * 1024 + 1), run_id: '', ...defaults }, modelProfiles));
});

test('minimal form submits spec and compatible selected profile fields', async () => {
  const { callBackendTool, calls } = newRunBackend();
  let created;
  renderNewRun(callBackendTool, (runId) => { created = runId; });
  await screen.findByRole('heading', { name: 'New Run' });
  fireEvent.change(screen.getByLabelText(/SPEC/), { target: { value: 'Implement the requested feature.' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create Run' }));
  await waitFor(() => assert.equal(created, 'created-001'));
  const payload = calls.find(([name]) => name === 'metaharness.create_run')[1];
  assert.deepEqual(payload, {
    spec: 'Implement the requested feature.', planner_profile: 'plan-a', mechanical_profile: 'impl-a',
    reasoning_profile: 'impl-a', agentic_profile: 'impl-a', final_reviewer_profile: 'review-a',
    semantic_reviser_profile: 'revise-a', check_repair_profile: 'repair-a',
  });
  assert.equal(Object.keys(payload).some((key) => /key|secret|token/i.test(key)), false);
});

test('advanced form submits all supported run options', async () => {
  const { callBackendTool, calls } = newRunBackend();
  renderNewRun(callBackendTool);
  await screen.findByRole('heading', { name: 'New Run' });
  fireEvent.change(screen.getByLabelText(/SPEC/), { target: { value: 'Do work.' } });
  fireEvent.click(screen.getByText('Advanced'));
  fireEvent.change(screen.getByLabelText('Run ID (optional)'), { target: { value: 'custom-run-2' } });
  fireEvent.change(screen.getByLabelText('Decomposition'), { target: { value: 'aggressive' } });
  fireEvent.change(screen.getByLabelText('Execution mode policy'), { target: { value: 'auto' } });
  fireEvent.click(screen.getByLabelText('Semantic revision'));
  fireEvent.change(screen.getByLabelText('Max check repair attempts'), { target: { value: '0' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create Run' }));
  await waitFor(() => assert.ok(calls.some(([name]) => name === 'metaharness.create_run')));
  const payload = calls.find(([name]) => name === 'metaharness.create_run')[1];
  assert.equal(payload.run_id, 'custom-run-2');
  assert.equal(payload.decomposition, 'aggressive');
  assert.equal(payload.execution_mode_policy, 'auto');
  assert.equal(payload.semantic_revision_enabled, false);
  assert.equal(payload.max_check_repair_attempts, 0);
  assert.equal(payload.repair_scope_policy, 'auto-bounded');
});

test('backend failure leaves spec in place and reports capacity conflicts', async () => {
  const { callBackendTool } = newRunBackend({
    'metaharness.create_run': async () => ({ ok: false, error: { code: 'RUN_CAPACITY', message: 'maximum active runs reached', httpStatus: 409 } }),
  });
  renderNewRun(callBackendTool);
  await screen.findByRole('heading', { name: 'New Run' });
  const spec = 'Please keep this exact SPEC.';
  fireEvent.change(screen.getByLabelText(/SPEC/), { target: { value: spec } });
  fireEvent.click(screen.getByRole('button', { name: 'Create Run' }));
  const alert = await screen.findByRole('alert');
  assert.ok(alert.textContent.includes('capacity is full'));
  assert.equal(screen.getByLabelText(/SPEC/).value, spec);
});

test('submitting twice while create_run is pending makes one backend request', async () => {
  let resolveCreate;
  const { callBackendTool, calls } = newRunBackend({
    'metaharness.create_run': async () => new Promise((resolve) => { resolveCreate = resolve; }),
  });
  renderNewRun(callBackendTool);
  await screen.findByRole('heading', { name: 'New Run' });
  fireEvent.change(screen.getByLabelText(/SPEC/), { target: { value: 'Do work.' } });
  const button = screen.getByRole('button', { name: 'Create Run' });
  fireEvent.click(button);
  fireEvent.click(button);
  assert.equal(calls.filter(([name]) => name === 'metaharness.create_run').length, 1);
  resolveCreate({ run_id: 'created-001' });
  await waitFor(() => assert.equal(screen.getByRole('button', { name: 'Create Run' }).disabled, false));
});
