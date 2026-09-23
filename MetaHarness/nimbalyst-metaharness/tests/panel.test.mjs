import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { classifyRunStatus } from '../src/panel/runStatus.ts';
import { RunsDashboard } from '../src/panel/RunsDashboard.tsx';
import { RunDetail } from '../src/panel/run/RunDetail.tsx';
import { deriveRunActions } from '../src/panel/run/runActions.ts';
import { NewRunForm } from '../src/panel/NewRunForm.tsx';
import { MetaHarnessPanel } from '../src/panel/MetaHarnessPanel.tsx';
import { bindExtensionRuntime, unbindExtensionRuntime } from '../src/runtime/extensionRuntime.ts';
import { defaultsFromServer, validateRunForm, buildCreateRunInput } from '../src/model/runForm.ts';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { cleanup, fireEvent, render, screen, waitFor } = await import('@testing-library/react');

afterEach(() => cleanup());

test('run actions use MetaHarness resume, approval and capability authority', () => {
  const base = { status: 'failed', capabilities: { resume: true, recover_plan: true, plan_approval: true, scope_approval: true, cancel: false } };
  assert.deepEqual(deriveRunActions({ ...base, overview: { resume: { resumable: true, phase: 'implement_step', label: 'Retry S02' } } }), {
    canResume: true, resumeLabel: 'Retry S02', canRecoverPlan: false,
    canApprovePlan: false, canApproveScope: false, canCancel: false,
  });
  assert.equal(deriveRunActions({ ...base, overview: { resume: { resumable: false, phase: 'implementation', reason: 'integrity failure' } } }).canResume, false);
  const recovered = deriveRunActions({
    status: 'failed', capabilities: { recover_plan: true }, plan_recovery: { eligible: true },
  });
  assert.equal(recovered.canRecoverPlan, true);
  assert.equal(deriveRunActions({ status: 'implementing', capabilities: { cancel: false } }).canCancel, false);
});

test('plan recovery displays rejected plan, counts UTF-8 bytes and submits confirmed replacement', async () => {
  const calls = [];
  const run = { run_id: 'recover-001', status: 'failed', planner_raw: 'invalid plan', plan_recovery: { eligible: true, reason: 'planner output invalid', max_bytes: 5 } };
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.get_run') return run;
    if (name === 'metaharness.get_config') return { capabilities: { recover_plan: true } };
    if (name === 'metaharness.recover_plan') return { ok: true };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'recover-001', callBackendTool, onBack: () => {} }));
  await screen.findByRole('heading', { name: 'REPLACE PLAN' });
  fireEvent.click(screen.getByText('Rejected / invalid plan'));
  assert.ok(screen.getByText('invalid plan'));
  const textarea = screen.getByLabelText('Replacement META PLAN v2');
  fireEvent.change(textarea, { target: { value: 'ééé' } });
  assert.ok(screen.getByText('6 / 5 bytes UTF-8'));
  assert.equal(screen.getByRole('button', { name: 'Review replacement' }).disabled, true);
  fireEvent.change(textarea, { target: { value: 'Plan' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review replacement' }));
  fireEvent.click(screen.getByRole('button', { name: 'REPLACE PLAN & CONTINUE' }));
  await waitFor(() => assert.ok(calls.some(([name]) => name === 'metaharness.recover_plan')));
  assert.deepEqual(calls.find(([name]) => name === 'metaharness.recover_plan')[1], {
    runId: 'recover-001', input: { plan: 'Plan' },
  });
});

test('resume stale state 409 refreshes run and reports the fixed stale-state message', async () => {
  let reads = 0;
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.get_run') return ++reads === 1
      ? { run_id: 'resume-001', status: 'failed', overview: { resume: { resumable: true, label: 'Retry S02' } } }
      : { run_id: 'resume-001', status: 'implementing', overview: { resume: { resumable: false } } };
    if (name === 'metaharness.get_config') return { capabilities: { resume: true } };
    if (name === 'metaharness.resume_run') return { ok: false, error: { httpStatus: 409, message: 'run is not resumable' } };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'resume-001', callBackendTool, onBack: () => {} }));
  const button = await screen.findByRole('button', { name: 'Retry S02' });
  fireEvent.click(button);
  await screen.findByText('The run changed before this action could be applied. Data has been refreshed.');
  assert.ok(screen.getByText('implementing'));
  assert.equal(screen.queryByRole('button', { name: 'Retry S02' }), null);
  assert.equal(calls.filter(([name]) => name === 'metaharness.get_run').length, 2);
});

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
    ...overrides,
  }));
}

test('unconfigured fullscreen panel shows the inline form and switches to dashboard after connection', async () => {
  const saved = {};
  const storage = {
    get: (key) => saved[key],
    set: async (key, value) => { saved[key] = value; },
  };
  const calls = [];
  const context = { services: { ai: { callBackendTool: async (name, args, workspacePath) => {
    calls.push([name, args, workspacePath]);
    if (name === 'metaharness.status') return { configured: true, connected: true };
    if (name === 'metaharness.list_runs') return [];
    throw new Error(`Unexpected backend call: ${name}`);
  } } } };
  bindExtensionRuntime(context);
  render(React.createElement(MetaHarnessPanel, { host: {
    storage, theme: 'dark', workspacePath: '/work', openFile: () => {},
  } }));
  assert.ok(screen.getByRole('heading', { name: 'Configure MetaHarness' }));
  assert.ok(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  assert.equal(screen.queryByRole('button', { name: 'Open Settings' }), null);
  fireEvent.change(screen.getByLabelText('Configuration file'), { target: { value: '/work/config.toml' } });
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('No runs in this workspace yet.');
  assert.deepEqual(calls.map(([name]) => name), ['metaharness.status', 'metaharness.status', 'metaharness.list_runs']);
  assert.ok(calls.every(([, , workspacePath]) => workspacePath === '/work'));
  assert.equal(saved.settings.configPath, '/work/config.toml');
  unbindExtensionRuntime();
});

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
    if (name === 'awaiting approval') {
      assert.equal(screen.queryByRole('button', { name: 'APPROVE & CONTINUE' }), null);
      assert.equal(screen.queryByRole('button', { name: 'REJECT PLAN' }), null);
    }
    if (name === 'planning') {
      fireEvent.click(screen.getByRole('tab', { name: 'Plan' }));
      assert.ok(screen.getByText('# Draft plan'));
    }
    if (name === 'implementing') {
      fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
      assert.ok(screen.getByText('API search endpoint'));
      assert.equal(screen.getByText('8c12d77').title, '8c12d77abcdef');
      assert.ok(screen.getByText('4'));
    }
    if (name === 'failed') assert.ok(screen.getByText('workspace setup failed'));
    if (name === 'planning') assert.ok(screen.getByText('# Draft plan'));
    if (name === 'published') assert.ok(screen.getByText('pull_request'));
  });
}

test('run detail tabs show empty states when each optional artifact is absent', async () => {
  const callBackendTool = async (tool) => tool === 'metaharness.get_run' ? { run_id: 'empty-views', status: 'published' } : tool === 'metaharness.progress' ? { events: [], next_offset: 0 } : {};
  render(React.createElement(RunDetail, { runId: 'empty-views', callBackendTool, onBack: () => {} }));
  await screen.findByRole('tab', { name: 'Overview' });
  for (const [tab, expected] of [['Plan', 'Plan has not been produced.'], ['Steps', 'No implementation steps are available yet.'], ['Checks', 'No data available yet.'], ['Review', 'No data available yet.'], ['Diff', 'Diff is not available.'], ['Usage', 'No data available yet.'], ['Logs', 'Run is terminal; live polling stopped.'], ['Diagnostics', 'No data available yet.'], ['Results', 'No data available yet.']]) {
    fireEvent.click(screen.getByRole('tab', { name: tab }));
    assert.ok(await screen.findByText(expected), `${tab} should have an empty state`);
  }
});

test('run tabs support roving focus with arrow keys', async () => {
  const callBackendTool = async (tool) => tool === 'metaharness.get_run' ? { run_id: 'keyboard-tabs', status: 'published' } : {};
  render(React.createElement(RunDetail, { runId: 'keyboard-tabs', callBackendTool, onBack: () => {} }));
  const overview = await screen.findByRole('tab', { name: 'Overview' });
  fireEvent.keyDown(overview, { key: 'ArrowRight' });
  const plan = screen.getByRole('tab', { name: 'Plan' });
  assert.equal(plan.getAttribute('aria-selected'), 'true');
  assert.equal(document.activeElement, plan);
  fireEvent.keyDown(plan, { key: 'End' });
  assert.equal(screen.getByRole('tab', { name: 'Results' }).getAttribute('aria-selected'), 'true');
});

test('run tabs display partial check, review, diff, usage and diagnostic artifacts', async () => {
  const callBackendTool = async (tool) => tool === 'metaharness.get_run' ? {
    run_id: 'partial-views', status: 'implementing', cycle: 2, planner_raw: 'draft plan',
    implementation_bundle: { steps: [{ id: 'S01', title: 'Implement API' }] },
    checks: { results: [{ id: 'lint', passed: true, duration_seconds: 4.2 }] },
    review: { verdict: 'APPROVE', category: 'IMPLEMENTATION', reviewer_profile: 'review-a', cycle: 2, candidate_sha: 'abc123' },
    reviewer_raw: 'raw reviewer note',
    candidate: { changed_files: ['src/new.ts'], diff_tail: 'diff contents', diff_truncated: true },
    usage: { phases: [{ phase: 'planner', input_tokens: 10, output_tokens: 5, total_tokens: 15 }] },
    diagnostics: { content: 'diagnostic report' },
    results: { status: 'passed' },
  } : tool === 'metaharness.progress' ? { events: ['step started'], next_offset: 20 } : {};
  const opened = [];
  render(React.createElement(RunDetail, { runId: 'partial-views', callBackendTool, workspacePath: '/workspace/project', openFile: (path) => opened.push(path), onBack: () => {} }));
  await screen.findByRole('tab', { name: 'Overview' });
  fireEvent.click(screen.getByRole('tab', { name: 'Plan' }));
  assert.ok(await screen.findByText('draft plan'));
  fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
  assert.ok(await screen.findByText('Implement API'));
  fireEvent.click(screen.getByRole('tab', { name: 'Checks' }));
  assert.ok(await screen.findByText('lint'));
  fireEvent.click(screen.getByRole('tab', { name: 'Review' }));
  assert.ok(await screen.findByText('IMPLEMENTATION'));
  assert.ok(screen.getByText('raw reviewer note'));
  fireEvent.click(screen.getByRole('tab', { name: 'Diff' }));
  assert.ok(await screen.findByText('diff contents'));
  fireEvent.click(screen.getByRole('button', { name: 'Open file' }));
  assert.deepEqual(opened, ['/workspace/project/src/new.ts']);
  fireEvent.click(screen.getByRole('tab', { name: 'Usage' }));
  assert.ok(await screen.findByText(/Input: 10/));
  fireEvent.click(screen.getByRole('tab', { name: 'Diagnostics' }));
  assert.ok(await screen.findByText('diagnostic report'));
  fireEvent.click(screen.getByRole('tab', { name: 'Results' }));
  assert.ok(await screen.findByText(/passed/));
  fireEvent.click(screen.getByRole('tab', { name: 'Logs' }));
  assert.ok(await screen.findByText('step started'));
});

test('Open file rejects traversal and absolute paths', async () => {
  const callBackendTool = async (tool) => tool === 'metaharness.get_run' ? { candidate: { changed_files: ['../outside', '/etc/passwd', 'safe/file.ts'], diff_tail: '' } } : {};
  const opened = [];
  render(React.createElement(RunDetail, { runId: 'paths', callBackendTool, workspacePath: '/workspace', openFile: (path) => opened.push(path), onBack: () => {} }));
  await screen.findByRole('tab', { name: 'Diff' });
  fireEvent.click(screen.getByRole('tab', { name: 'Diff' }));
  const buttons = screen.getAllByRole('button', { name: 'Open file' });
  buttons.forEach((button) => fireEvent.click(button));
  assert.deepEqual(opened, ['/workspace/safe/file.ts']);
});

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

const approvalRun = (extra = {}) => ({
  run_id: 'approval-001', status: 'awaiting_plan_approval', approval: { recorded: false },
  run_options: { profiles: {}, pipeline: { semantic_revision_enabled: true, max_check_repair_attempts: 1 } },
  implementation_bundle: { steps: [
    { id: 'S01', title: 'Database model', execution_class: 'MECHANICAL' },
    { id: 'S02', title: 'API layer', execution_class: 'REASONING' },
  ] },
  ...extra,
});

test('plan approval loads metadata-compatible profiles, maps steps, submits once, and reloads the run', async () => {
  const calls = [];
  let detail = approvalRun();
  let finishMutation;
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.get_run') return detail;
    if (name === 'metaharness.model_profiles') return {
      profiles: [
        ...modelProfiles.profiles,
        { id: 'impl-b', display_name: 'Implementer B', roles: ['implementer'], execution_classes: ['REASONING'] },
        { id: 'not-impl', display_name: 'Reviewer only', roles: ['reviewer'] },
      ],
      defaults: modelProfiles.defaults,
    };
    if (name === 'metaharness.approve_run') {
      await new Promise((resolve) => { finishMutation = resolve; });
      detail = { ...detail, status: 'implementing', approval: { recorded: true, decision: 'APPROVE' } };
      return { ok: true };
    }
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'approval-001', callBackendTool, onBack: () => {} }));
  await screen.findByRole('heading', { name: 'PLAN APPROVAL REQUIRED' });
  await screen.findByLabelText('S02 Profile');
  const s01 = screen.getByLabelText('S01 Profile');
  const s02 = screen.getByLabelText('S02 Profile');
  assert.deepEqual([...s01.options].map((option) => option.value), ['impl-a']);
  assert.deepEqual([...s02.options].map((option) => option.value), ['impl-a', 'impl-b']);
  assert.equal([...s01.options].some((option) => option.value === 'not-impl'), false);
  fireEvent.change(s01, { target: { value: 'impl-a' } });
  fireEvent.change(s02, { target: { value: 'impl-b' } });
  const approve = screen.getByRole('button', { name: 'APPROVE & CONTINUE' });
  fireEvent.click(approve);
  fireEvent.click(approve);
  await waitFor(() => assert.equal(calls.filter(([name]) => name === 'metaharness.approve_run').length, 1));
  assert.equal(approve.disabled, true);
  assert.deepEqual(calls.find(([name]) => name === 'metaharness.approve_run')[1], {
    runId: 'approval-001',
    decision: 'APPROVE', final_reviewer_profile: 'review-a', semantic_reviser_profile: 'revise-a',
    check_repair_profile: 'repair-a', step_profiles: { S01: 'impl-a', S02: 'impl-b' },
  });
  finishMutation();
  await waitFor(() => assert.equal(calls.filter(([name]) => name === 'metaharness.get_run').length, 2));
  await waitFor(() => assert.equal(screen.queryByRole('heading', { name: 'PLAN APPROVAL REQUIRED' }), null));
});

test('plan reject confirms irreversible choice and omits profile fields', async () => {
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.get_run') return approvalRun();
    if (name === 'metaharness.model_profiles') return modelProfiles;
    if (name === 'metaharness.approve_run') return { ok: true };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'approval-001', callBackendTool, onBack: () => {} }));
  await screen.findByRole('heading', { name: 'PLAN APPROVAL REQUIRED' });
  fireEvent.click(screen.getByRole('button', { name: 'REJECT PLAN' }));
  assert.ok(screen.getByRole('dialog').textContent.includes('irreversible'));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm irreversible rejection' }));
  await waitFor(() => assert.ok(calls.some(([name]) => name === 'metaharness.approve_run')));
  assert.deepEqual(calls.find(([name]) => name === 'metaharness.approve_run')[1], {
    runId: 'approval-001', decision: 'REJECT',
  });
});

test('scope expansion approval displays added paths and sends only the decision', async () => {
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.get_run') return {
      run_id: 'scope-001', status: 'waiting_scope_approval',
      scope_delta: { added_paths: ['src/foo.py', 'tests/test_foo.py'] },
    };
    if (name === 'metaharness.approve_scope') return { ok: true };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'scope-001', callBackendTool, onBack: () => {} }));
  await screen.findByRole('heading', { name: 'REPAIR SCOPE EXPANSION' });
  assert.ok(screen.getByText('+ src/foo.py'));
  assert.ok(screen.getByText('+ tests/test_foo.py'));
  fireEvent.click(screen.getByRole('button', { name: 'APPROVE' }));
  await waitFor(() => assert.ok(calls.some(([name]) => name === 'metaharness.approve_scope')));
  assert.deepEqual(calls.find(([name]) => name === 'metaharness.approve_scope')[1], {
    runId: 'scope-001', decision: 'APPROVE',
  });
});

test('approval is hidden when API reports a recorded decision even if status is stale', async () => {
  const callBackendTool = async (name) => {
    if (name === 'metaharness.get_run') return approvalRun({ approval: { recorded: true, decision: 'REJECT' } });
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'approval-001', callBackendTool, onBack: () => {} }));
  await screen.findByText('awaiting_plan_approval');
  assert.equal(screen.queryByRole('button', { name: 'APPROVE & CONTINUE' }), null);
  assert.equal(screen.queryByRole('button', { name: 'REJECT PLAN' }), null);
});

test('HTTP conflict reports the error and refreshes the run state', async () => {
  const calls = [];
  let reads = 0;
  const callBackendTool = async (name) => {
    calls.push(name);
    if (name === 'metaharness.get_run') return ++reads === 1 ? approvalRun() : { ...approvalRun(), status: 'implementing' };
    if (name === 'metaharness.model_profiles') return modelProfiles;
    if (name === 'metaharness.approve_run') return { ok: false, error: { code: 'HTTP_ERROR', httpStatus: 409, message: 'run is not awaiting plan approval' } };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(RunDetail, { runId: 'approval-001', callBackendTool, onBack: () => {} }));
  await screen.findByRole('heading', { name: 'PLAN APPROVAL REQUIRED' });
  await waitFor(() => assert.equal(screen.getByRole('button', { name: 'APPROVE & CONTINUE' }).disabled, false));
  fireEvent.click(screen.getByRole('button', { name: 'APPROVE & CONTINUE' }));
  await screen.findByText('The run changed before this action could be applied. Data has been refreshed.');
  assert.equal(screen.getByText('The run changed before this action could be applied. Data has been refreshed.').getAttribute('role'), 'alert');
  await waitFor(() => assert.equal(calls.filter((name) => name === 'metaharness.get_run').length, 2));
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

test('the dashboard does not keep its own poll while a run detail is shown', async () => {
  const originalSetInterval = window.setInterval;
  const originalClearInterval = window.clearInterval;
  const active = new Map();
  let nextId = 1000;
  window.setInterval = (_callback, interval) => { nextId += 1; active.set(nextId, interval); return nextId; };
  window.clearInterval = (id) => { active.delete(id); };
  try {
    const callBackendTool = async (name) => {
      if (name === 'metaharness.status') return { configured: true, connected: true };
      if (name === 'metaharness.list_runs') return [];
      if (name === 'metaharness.get_run') return { run_id: 'r1', status: 'committed' };
      return {};
    };
    renderDashboard(callBackendTool, { view: { kind: 'run', runId: 'r1' } });
    await screen.findByText('r1', { exact: false });
    await new Promise((resolve) => setTimeout(resolve, 50));
    // Terminal run: neither the dashboard nor RunDetail keeps a state poll.
    assert.deepEqual([...active.values()], []);
  } finally {
    window.setInterval = originalSetInterval;
    window.clearInterval = originalClearInterval;
  }
});
