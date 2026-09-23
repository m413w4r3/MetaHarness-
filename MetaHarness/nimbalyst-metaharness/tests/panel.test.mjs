import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { classifyRunStatus } from '../src/panel/runStatus.ts';
import { RunsDashboard } from '../src/panel/RunsDashboard.tsx';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { cleanup, fireEvent, render, screen, waitFor } = await import('@testing-library/react');

afterEach(() => cleanup());

const settings = { executable: 'metaharness', configPath: '/work/metaharness.toml', port: 8765, autoStart: false, pollIntervalMs: 1000 };

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
