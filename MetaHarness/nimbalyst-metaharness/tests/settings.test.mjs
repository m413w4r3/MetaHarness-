import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { MetaHarnessConfigForm } from '../src/config/MetaHarnessConfigForm.tsx';
import { DEFAULT_SETTINGS, validateSettings } from '../src/config/settings.ts';
import { MetaHarnessSettings } from '../src/settings/MetaHarnessSettings.tsx';

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

function fillConfig() {
  fireEvent.change(screen.getByLabelText('Configuration file'), { target: { value: '/work/config.toml' } });
}

test('shared config form shows defaults, validates and persists on Save', async () => {
  const storage = fakeStorage();
  render(React.createElement(MetaHarnessConfigForm, { storage, theme: 'dark', workspacePath: '/work' }));
  assert.equal(screen.getByLabelText('MetaHarness executable').value, 'metaharness');
  assert.equal(screen.getByLabelText('Configuration file').value, '');
  assert.equal(validateSettings(DEFAULT_SETTINGS), 'Configuration path must not be empty.');
  fillConfig();
  fireEvent.click(screen.getByRole('button', { name: 'SAVE' }));
  await waitFor(() => assert.deepEqual(storage.saved.settings, { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' }));
  assert.equal(validateSettings({ ...DEFAULT_SETTINGS, configPath: '/work/config.toml', port: 65536 }), 'Port must be an integer between 1 and 65535.');
  assert.equal(validateSettings({ ...DEFAULT_SETTINGS, configPath: '/work/config.toml', pollIntervalMs: 499 }), 'Polling interval must be between 500 and 30000 ms.');
});

test('Save & Test calls status, starts when configured, then checks status and configures', async () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' };
  const storage = fakeStorage({ settings });
  const calls = [];
  let statuses = 0;
  let configured = 0;
  const callBackendTool = async (name, args, workspacePath) => {
    calls.push([name, args, workspacePath]);
    if (name === 'metaharness.status') return { connected: ++statuses > 1 };
    if (name === 'metaharness.start') return { connected: true };
    throw new Error(`Unexpected backend call: ${name}`);
  };
  render(React.createElement(MetaHarnessConfigForm, { storage, theme: 'dark', workspacePath: '/work', callBackendTool, onConfigured: () => { configured += 1; } }));
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('Connected');
  assert.deepEqual(calls.map(([name]) => name), ['metaharness.status', 'metaharness.start', 'metaharness.status']);
  assert.equal(calls[0][2], '/work');
  assert.equal(configured, 1);
  assert.deepEqual(storage.saved.settings, settings);
});

test('successful connected status saves and configures without starting', async () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml', autoStart: false };
  const calls = [];
  let configured = false;
  render(React.createElement(MetaHarnessConfigForm, {
    storage: fakeStorage({ settings }), theme: 'light',
    callBackendTool: async (name) => { calls.push(name); return { connected: true }; },
    onConfigured: () => { configured = true; },
  }));
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('Connected');
  assert.deepEqual(calls, ['metaharness.status']);
  assert.equal(configured, true);
});

test('permission rejection shows friendly copy and keeps the form visible', async () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' };
  render(React.createElement(MetaHarnessConfigForm, {
    storage: fakeStorage({ settings }), theme: 'dark',
    callBackendTool: async () => { throw new Error('Backend permission denied'); },
  }));
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('MetaHarness backend permission is required to control local runs.');
  assert.ok(screen.getByLabelText('Configuration file'));
});

test('port and missing executable backend failures use friendly messages', async () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' };
  const storage = fakeStorage({ settings });
  render(React.createElement(MetaHarnessConfigForm, {
    storage, theme: 'dark',
    callBackendTool: async () => ({ ok: false, error: { code: 'PORT_IN_USE', message: 'port 8765 is occupied by a non-MetaHarness service' } }),
  }));
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('Port 8765 is already in use by a service that is not MetaHarness.');
  cleanup();

  render(React.createElement(MetaHarnessConfigForm, {
    storage, theme: 'dark',
    callBackendTool: async () => ({ ok: false, error: { code: 'SPAWN_FAILED', message: 'MetaHarness could not be started', details: { causeCode: 'ENOENT' } } }),
  }));
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('MetaHarness executable was not found. Check the configured absolute path.');
});

test('unavailable backend module has friendly copy', async () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' };
  render(React.createElement(MetaHarnessConfigForm, {
    storage: fakeStorage({ settings }), theme: 'dark',
    callBackendTool: async () => { throw new Error('backend module metaharness-runtime is not running'); },
  }));
  fireEvent.click(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
  await screen.findByText('The MetaHarness Nimbalyst backend module is unavailable. Reload the extension and retry.');
});

test('settings route is a wrapper around the shared configuration form', () => {
  const settings = { ...DEFAULT_SETTINGS, configPath: '/work/config.toml' };
  render(React.createElement(MetaHarnessSettings, { storage: fakeStorage({ settings }), theme: 'dark', workspacePath: '/work' }));
  assert.ok(screen.getByLabelText('MetaHarness executable'));
  assert.ok(screen.getByRole('button', { name: 'SAVE & TEST CONNECTION' }));
});
