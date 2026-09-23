import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { RunsDashboard } from '../src/panel/RunsDashboard.tsx';
import { MetaHarnessPanel } from '../src/panel/MetaHarnessPanel.tsx';
import { ConfigurationView } from '../src/panel/configuration/ConfigurationView.tsx';
import { bindExtensionRuntime, unbindExtensionRuntime } from '../src/runtime/extensionRuntime.ts';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { cleanup, fireEvent, render, screen, waitFor } = await import('@testing-library/react');

afterEach(() => { cleanup(); unbindExtensionRuntime(); });

const settings = { executable: '/opt/metaharness', configPath: '/work/examples/autowork.toml', port: 8766, autoStart: false, pollIntervalMs: 1000 };
const effectiveConfig = {
  repository: { repo: '/work/AutoWork', base_ref: 'main', remote: 'origin' },
  planning: { protocol: 'v2', decomposition: 'aggressive', execution_mode_policy: 'auto' },
  routing: { mechanical_profile: 'codex-luna-high', reasoning_profile: 'codex-luna-xhigh', agentic_profile: 'codex-deepseek-flash-max' },
  approval: { require_plan_approval: true },
  publish: { enabled: false },
  checks: [{ id: 'lint', description: 'Lint' }, { id: 'typecheck' }],
};

function storage(saved = { settings }) {
  return { saved, get: (key) => saved[key], set: async (key, value) => { saved[key] = value; } };
}

function backend(status, overrides = {}) {
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (overrides[name]) return overrides[name](args);
    if (name === 'metaharness.status') return { configured: true, ...status };
    if (name === 'metaharness.get_config') return effectiveConfig;
    if (name === 'metaharness.list_runs') return [];
    if (name === 'metaharness.stop') return { configured: true, connected: false, serverOwned: false };
    if (name === 'metaharness.start') return { configured: true, connected: true, serverOwned: true };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  return { calls, names: () => calls.map(([name]) => name), callBackendTool };
}

function renderConfiguration(callBackendTool, props = {}) {
  return render(React.createElement(ConfigurationView, {
    settings, storage: storage(), theme: 'dark', workspacePath: '/work', callBackendTool,
    onBack: () => {}, onSettingsSaved: () => {}, ...props,
  }));
}

test('dashboard renders permanent configuration button in connected, disconnected and error states', async () => {
  const cases = [
    [backend({ connected: true }).callBackendTool, 'No runs in this workspace yet.'],
    [backend({ connected: false }).callBackendTool, 'MetaHarness is not connected. Check your configuration or try again.'],
    [async () => ({ ok: false, error: { code: 'X', message: 'backend exploded' } }), 'backend exploded'],
  ];
  for (const [callBackendTool, text] of cases) {
    let opened = 0;
    render(React.createElement(RunsDashboard, {
      callBackendTool, settings, view: { kind: 'dashboard' }, onViewChange: () => {}, workspacePath: '/work',
      openFile: () => {}, onOpenConfiguration: () => { opened += 1; },
    }));
    await screen.findByText(text);
    const button = screen.getByRole('button', { name: 'Open MetaHarness configuration' });
    assert.ok(button.title.startsWith('MetaHarness configuration'));
    assert.ok(button.title.includes(settings.configPath));
    fireEvent.click(button);
    assert.equal(opened, 1);
    cleanup();
  }
});

test('configuration button opens ConfigurationView with persisted settings and Back returns to runs', async () => {
  const { callBackendTool } = backend({ connected: true, serverOwned: true });
  bindExtensionRuntime({ services: { ai: { callBackendTool } } });
  render(React.createElement(MetaHarnessPanel, { host: { storage: storage(), theme: 'dark', workspacePath: '/work', openFile: () => {} } }));
  await screen.findByText('No runs in this workspace yet.');
  fireEvent.click(screen.getByRole('button', { name: 'Open MetaHarness configuration' }));
  assert.ok(screen.getByRole('heading', { name: 'MetaHarness Configuration' }));
  assert.equal(screen.queryAllByRole('heading', { name: 'Configure MetaHarness' }).length, 0);
  assert.equal(screen.getByLabelText('MetaHarness executable').value, '/opt/metaharness');
  assert.equal(screen.getByLabelText('Configuration file').value, '/work/examples/autowork.toml');
  assert.equal(screen.getByLabelText('Port').value, '8766');
  await screen.findByText('codex-luna-high');
  fireEvent.click(screen.getByRole('button', { name: '← Back to runs' }));
  await screen.findByText('No runs in this workspace yet.');
  assert.ok(screen.getByRole('button', { name: 'Open MetaHarness configuration' }));
});

test('effective config renders server values and Refresh calls get_config again', async () => {
  const { callBackendTool, names } = backend({ connected: true, serverOwned: true });
  renderConfiguration(callBackendTool);
  await screen.findByText('codex-luna-high');
  for (const value of ['/work/AutoWork', 'main', 'origin', 'v2', 'aggressive', 'auto', 'codex-luna-xhigh', 'codex-deepseek-flash-max', 'lint', 'typecheck']) {
    assert.ok(screen.getByText(value), value);
  }
  const approval = screen.getByText('Plan approval').parentElement;
  assert.ok(approval.textContent.includes('Enabled'));
  assert.ok(screen.getByText('Publish', { selector: 'span' }).parentElement.textContent.includes('Disabled'));
  assert.equal(screen.getByText('Ownership').nextSibling.textContent, 'Nimbalyst');
  assert.deepEqual(names(), ['metaharness.status', 'metaharness.get_config']);
  fireEvent.click(screen.getByRole('button', { name: 'REFRESH EFFECTIVE CONFIG' }));
  await waitFor(() => assert.deepEqual(names(), ['metaharness.status', 'metaharness.get_config', 'metaharness.get_config']));
});

test('owned server restart runs stop, start, status, get_config in order', async () => {
  const { callBackendTool, names, calls } = backend({ connected: true, serverOwned: true });
  renderConfiguration(callBackendTool);
  const restart = await screen.findByRole('button', { name: 'RESTART & RELOAD CONFIG' });
  await screen.findByText('codex-luna-high');
  fireEvent.click(restart);
  assert.ok(screen.getByRole('button', { name: 'RESTARTING…' }).disabled);
  await waitFor(() => assert.equal(calls.length, 6));
  assert.deepEqual(names().slice(2), ['metaharness.stop', 'metaharness.start', 'metaharness.status', 'metaharness.get_config']);
  assert.deepEqual(calls[3][1], { settings });
  await screen.findByRole('button', { name: 'RESTART & RELOAD CONFIG' });
});

test('external server is never stopped or started and shows the external notice', async () => {
  const { callBackendTool, names } = backend({ connected: true, serverOwned: false });
  renderConfiguration(callBackendTool);
  await screen.findByText('External MetaHarness process');
  assert.ok(screen.getByText(/was not started by Nimbalyst/));
  assert.equal(screen.getByText('Ownership').nextSibling.textContent, 'External');
  assert.equal(screen.queryByRole('button', { name: 'RESTART & RELOAD CONFIG' }), null);
  assert.equal(screen.queryByRole('button', { name: 'START METAHARNESS' }), null);
  fireEvent.click(screen.getByRole('button', { name: 'REFRESH EFFECTIVE CONFIG' }));
  await waitFor(() => assert.equal(names().filter((name) => name === 'metaharness.get_config').length, 2));
  assert.ok(!names().includes('metaharness.stop'));
  assert.ok(!names().includes('metaharness.start'));
});

test('disconnected server keeps the configuration editable and can be started explicitly', async () => {
  let connected = false;
  const { callBackendTool, names } = backend({}, {
    'metaharness.status': () => ({ configured: true, connected, serverOwned: connected }),
    'metaharness.start': () => { connected = true; return { connected: true, serverOwned: true }; },
  });
  renderConfiguration(callBackendTool);
  await screen.findByText('Disconnected');
  assert.equal(screen.getByText('Ownership').nextSibling.textContent, '—');
  for (const label of ['MetaHarness executable', 'Configuration file', 'Port', 'Start automatically', /Polling interval/]) {
    assert.equal(screen.getByLabelText(label).disabled, false, label);
  }
  assert.ok(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  fireEvent.change(screen.getByLabelText('Port'), { target: { value: '9000' } });
  assert.ok(screen.getByText('Unsaved changes — connection not retested'));
  assert.ok(!names().includes('metaharness.start'));
  fireEvent.click(screen.getByRole('button', { name: 'START METAHARNESS' }));
  await screen.findByText('codex-luna-high');
  assert.deepEqual(names(), ['metaharness.status', 'metaharness.start', 'metaharness.status', 'metaharness.get_config']);
  assert.equal(screen.getByText('Ownership').nextSibling.textContent, 'Nimbalyst');
});

test('start failure stays in the configuration view with a readable error', async () => {
  let backed = 0;
  const { callBackendTool } = backend({ connected: false, serverOwned: false }, {
    'metaharness.start': () => ({ ok: false, error: { code: 'SPAWN_FAILED', message: 'MetaHarness could not be started' } }),
  });
  renderConfiguration(callBackendTool, { onBack: () => { backed += 1; } });
  fireEvent.click(await screen.findByRole('button', { name: 'START METAHARNESS' }));
  await screen.findByText('MetaHarness could not be started');
  assert.ok(screen.getByRole('heading', { name: 'MetaHarness Configuration' }));
  assert.equal(backed, 0);
});

test('doctor report renders structured pass and fail checks', async () => {
  const { callBackendTool, calls } = backend({ connected: true, serverOwned: true }, {
    'metaharness.doctor': () => ({
      ok: false,
      checks: [
        { id: 'repo', status: 'pass', message: 'repository valid' },
        { id: 'credentials', status: 'fail', message: 'credential missing' },
      ],
    }),
  });
  renderConfiguration(callBackendTool);
  fireEvent.click(await screen.findByRole('button', { name: 'RUN DOCTOR' }));
  await screen.findByText('repository valid');
  assert.ok(screen.getByText('credential missing'));
  assert.ok(screen.getByText('Doctor — 1 failing check'));
  assert.ok(screen.getByText('pass'));
  assert.ok(screen.getByText('fail'));
  assert.ok(document.querySelector('.metaharness-doctor__check--fail'));
  assert.equal(document.querySelector('.metaharness-doctor__raw'), null);
  assert.deepEqual(calls.find(([name]) => name === 'metaharness.doctor')[1], { settings });
});

test('saving embedded settings persists them and reloads status with the new settings', async () => {
  const saved = { settings };
  const received = [];
  const { callBackendTool, calls } = backend({ connected: false, serverOwned: false });
  function Harness() {
    const [current, setCurrent] = React.useState(settings);
    return React.createElement(ConfigurationView, {
      settings: current, storage: storage(saved), theme: 'dark', workspacePath: '/work', callBackendTool,
      onBack: () => {}, onSettingsSaved: (next) => { received.push(next); setCurrent(next); },
    });
  }
  render(React.createElement(Harness));
  await screen.findByText('Disconnected');
  fireEvent.change(screen.getByLabelText('Port'), { target: { value: '9001' } });
  fireEvent.click(screen.getByRole('button', { name: 'SAVE' }));
  await waitFor(() => assert.equal(saved.settings.port, 9001));
  await waitFor(() => assert.equal(calls.filter(([name]) => name === 'metaharness.status').length, 2));
  assert.equal(received[0].port, 9001);
  assert.equal(calls.at(-1)[1].settings.port, 9001);
  assert.ok(!calls.some(([name]) => name === 'metaharness.start'));
});

test('config file is opened only inside the workspace; copy path is always offered', async () => {
  const opened = [];
  const { callBackendTool } = backend({ connected: false, serverOwned: false });
  renderConfiguration(callBackendTool, { openFile: (path) => opened.push(path) });
  fireEvent.click(await screen.findByRole('button', { name: 'OPEN CONFIG FILE' }));
  assert.deepEqual(opened, ['/work/examples/autowork.toml']);
  assert.ok(screen.getByRole('button', { name: 'COPY PATH' }));
  cleanup();
  renderConfiguration(callBackendTool, { openFile: () => {}, settings: { ...settings, configPath: '/elsewhere/a.toml' } });
  await screen.findByText('Disconnected');
  assert.equal(screen.queryByRole('button', { name: 'OPEN CONFIG FILE' }), null);
  assert.ok(screen.getByRole('button', { name: 'COPY PATH' }));
});
