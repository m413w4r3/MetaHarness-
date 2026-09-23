import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { DEFAULT_SETTINGS, MetaHarnessSettings, validateSettings } from '../src/settings/MetaHarnessSettings.tsx';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { cleanup, fireEvent, render, screen, waitFor } = await import('@testing-library/react');

afterEach(() => cleanup());

function fakeStorage(saved = {}) {
  return {
    saved,
    get: (key) => key === 'settings' ? saved.settings : undefined,
    set: async (key, value) => { saved[key] = value; },
  };
}

test('settings defaults validate and persist in project ExtensionStorage', async () => {
  const storage = fakeStorage();
  render(React.createElement(MetaHarnessSettings, { storage, workspacePath: '/work', callBackendTool: undefined }));
  await waitFor(() => assert.deepEqual(storage.saved.settings, DEFAULT_SETTINGS));
  assert.equal(screen.getByLabelText('Executable').value, 'metaharness');
  assert.equal(screen.getByLabelText('Configuration').value, '');
  fireEvent.change(screen.getByLabelText('Configuration'), {
    target: { value: '/home/user/dev/MetaHarness-/MetaHarness/examples/autowork.toml' },
  });
  await waitFor(() => assert.equal(storage.saved.settings.configPath,
    '/home/user/dev/MetaHarness-/MetaHarness/examples/autowork.toml'));
  assert.equal(validateSettings(DEFAULT_SETTINGS), 'Configuration path must not be empty.');
  const valid = { ...DEFAULT_SETTINGS, configPath: '/home/user/dev/MetaHarness-/MetaHarness/examples/autowork.toml' };
  assert.equal(validateSettings(valid), undefined);
  assert.equal(validateSettings({ ...valid, port: 65536 }), 'Port must be an integer between 1 and 65535.');
  assert.equal(validateSettings({ ...valid, pollIntervalMs: 499 }), 'Polling interval must be between 500 and 30000 ms.');
});

test('test connection uses explicit settings and does not start when autoStart is false', async () => {
  const storage = fakeStorage({ settings: { ...DEFAULT_SETTINGS, configPath: '/work/custom.toml', autoStart: false } });
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.status') return { connected: true, configured: true };
    return { repo: '/work' };
  };
  render(React.createElement(MetaHarnessSettings, { storage, workspacePath: '/work', callBackendTool }));
  await waitFor(() => assert.ok(calls.some(([name]) => name === 'metaharness.status')));
  calls.length = 0;
  fireEvent.click(screen.getByRole('button', { name: 'TEST CONNECTION' }));
  await waitFor(() => assert.match(screen.getByText('Connected').textContent, /Connected/));
  assert.deepEqual(calls.map(([name]) => name), ['metaharness.status']);
  assert.deepEqual(calls[0][1].settings, storage.saved.settings);
});

test('autoStart starts MetaHarness with the project settings when the status probe is down', async () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' };
  const calls = [];
  const callBackendTool = async (name, args) => {
    calls.push([name, args]);
    if (name === 'metaharness.status') return { connected: false };
    if (name === 'metaharness.start') return { connected: true, serverOwned: true };
    return { repository: { repo: '/work', base_ref: 'main' } };
  };
  render(React.createElement(MetaHarnessSettings, { storage: fakeStorage({ settings }), callBackendTool }));
  await waitFor(() => assert.ok(calls.some(([name]) => name === 'metaharness.start')));
  const startCall = calls.find(([name]) => name === 'metaharness.start');
  assert.deepEqual(startCall[1].settings, settings);
});

test('doctor displays each check on success and readable failure on backend rejection', async () => {
  const storage = fakeStorage({ settings: { ...DEFAULT_SETTINGS, configPath: '/work/custom.toml' } });
  let fail = false;
  const callBackendTool = async (name) => {
    if (name === 'metaharness.status') return { connected: false };
    if (name === 'metaharness.start') return { connected: true };
    if (name === 'metaharness.get_config') return { repository: { repo: '/work' } };
    if (fail) throw new Error('doctor could not read config');
    return { checks: [
      { name: 'config', status: 'pass', message: 'Configuration loaded' },
      { name: 'git', status: 'warn', message: 'Branch is dirty' },
      { name: 'publish', status: 'fail', message: 'Remote missing' },
    ] };
  };
  render(React.createElement(MetaHarnessSettings, { storage, callBackendTool }));
  fireEvent.click(screen.getByRole('button', { name: 'RUN DOCTOR' }));
  await screen.findByText('Configuration loaded');
  assert.equal(screen.getByText('PASS').textContent, 'PASS');
  assert.equal(screen.getByText('WARN').textContent, 'WARN');
  assert.equal(screen.getByText('FAIL').textContent, 'FAIL');
  cleanup();
  fail = true;
  render(React.createElement(MetaHarnessSettings, { storage, callBackendTool }));
  fireEvent.click(screen.getByRole('button', { name: 'RUN DOCTOR' }));
  await screen.findByText('MetaHarness doctor failed: doctor could not read config');
});

test('missing backend shows an actionable message instead of throwing', async () => {
  const storage = fakeStorage({ settings: { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' } });
  render(React.createElement(MetaHarnessSettings, { storage }));
  fireEvent.click(screen.getByRole('button', { name: 'TEST CONNECTION' }));
  await screen.findByText('MetaHarness backend is unavailable. Enable the extension backend and try again.');
  fireEvent.click(screen.getByRole('button', { name: 'RUN DOCTOR' }));
  await screen.findByText('MetaHarness backend is unavailable. Enable the extension backend and try again.');
});

test('a failing doctor report (ok: false) is rendered as structured checks, not as a backend error', async () => {
  const storage = fakeStorage({ settings: { ...DEFAULT_SETTINGS, configPath: '/work/custom.toml' } });
  const callBackendTool = async () => ({
    ok: false,
    checks: [{ id: 'credentials', status: 'fail', message: 'required credential is missing or invalid' }],
    summary: { passed: 0, failed: 1, warnings: 0 },
  });
  render(React.createElement(MetaHarnessSettings, { storage, callBackendTool }));
  fireEvent.click(screen.getByRole('button', { name: 'RUN DOCTOR' }));
  await screen.findByText('required credential is missing or invalid');
  assert.ok(screen.getByText('credentials'));
  assert.equal(screen.queryByText(/MetaHarness doctor failed/), null);
});
