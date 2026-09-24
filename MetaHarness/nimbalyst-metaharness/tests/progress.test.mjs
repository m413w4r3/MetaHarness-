import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import React from 'react';
import { useRunProgress } from '../src/hooks/useRunProgress.ts';
import { ProgressView } from '../src/panel/run/ProgressView.tsx';
import { eventMatchesFilter } from '../src/panel/run/EventRow.tsx';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { act, cleanup, fireEvent, render, screen, waitFor } = await import('@testing-library/react');

afterEach(() => cleanup());

test('structured categories drive filters and Errors includes an LLM 503', () => {
  const event = '15:12:09 [recovery] [S04] LLM_FAILURE · HTTP 503 after 3 attempts';
  assert.equal(eventMatchesFilter(event, 'Recovery'), true);
  assert.equal(eventMatchesFilter(event, 'Errors'), true);
  assert.equal(eventMatchesFilter('15:12:09 [step] [S04] step started', 'Steps'), true);
  assert.equal(eventMatchesFilter('15:12:09 [planner] planning started', 'Planner'), true);
});

test('progress view renders returned rows, empty state and backend errors visibly', async () => {
  const callBackendTool = async () => ({ events: ['15:12:09 [step] [S04] step started'], next_offset: 40 });
  render(React.createElement(ProgressView, { runId: 'p', status: 'waiting_external', callBackendTool }));
  assert.ok(await screen.findByText(/\[S04\] step started/));
  cleanup();
  render(React.createElement(ProgressView, { runId: 'empty', status: 'published', callBackendTool: async () => ({ events: [], next_offset: 0 }) }));
  assert.ok(screen.getByText('No progress events have been recorded yet.'));
  cleanup();
  render(React.createElement(ProgressView, { runId: 'broken', status: 'implementing', callBackendTool: async () => ({ ok: false, error: { message: 'progress backend unavailable' } }) }));
  assert.ok(await screen.findByRole('alert'));
  assert.match(screen.getByRole('alert').textContent, /progress backend unavailable/);
});

function Harness({ callBackendTool, enabled = true }) {
  const progress = useRunProgress({ runId: 'progress-1', enabled, intervalMs: 100, callBackendTool });
  return React.createElement('div', null,
    React.createElement('output', { 'data-testid': 'offset' }, progress.nextOffset),
    React.createElement('output', { 'data-testid': 'count' }, progress.events.length),
    React.createElement('output', { 'data-testid': 'dropped' }, progress.droppedCount),
    React.createElement('output', { 'data-testid': 'error' }, progress.error),
    React.createElement('button', { onClick: progress.reloadFromBeginning }, 'Reload'),
    React.createElement('ul', null, progress.events.map((event, index) => React.createElement('li', { key: index }, event))),
  );
}

test('polling uses server byte offsets, appends once, and ignores repeated offset batches', async () => {
  const calls = [];
  const callBackendTool = async (_name, args) => {
    calls.push(args);
    if (calls.length === 1) return { next_offset: 12, events: ['planner started'] };
    if (calls.length === 2) return { next_offset: 12, events: ['planner started'] };
    return { next_offset: 24, events: ['plan accepted'] };
  };
  render(React.createElement(Harness, { callBackendTool }));
  await screen.findByText('plan accepted');
  assert.deepEqual(calls.slice(0, 3).map(({ offset }) => offset), [0, 12, 12]);
  assert.equal(screen.getAllByText('planner started').length, 1);
  assert.equal(screen.getByTestId('offset').textContent, '24');
});

test('polling does not issue a second request while the current request is in flight', async () => {
  let release;
  let concurrent = 0;
  let active = 0;
  const callBackendTool = async () => {
    concurrent += 1;
    active += 1;
    await new Promise((resolve) => { release = resolve; });
    active -= 1;
    return { next_offset: 1, events: ['one'] };
  };
  const view = render(React.createElement(Harness, { callBackendTool }));
  await new Promise((resolve) => setTimeout(resolve, 250));
  assert.equal(concurrent, 1);
  assert.equal(active, 1);
  await act(async () => release());
  await screen.findByText('one');
  view.unmount();
});

test('reload from beginning resets events and asks the server from offset zero', async () => {
  const offsets = [];
  const callBackendTool = async (_name, args) => {
    offsets.push(args.offset);
    return offsets.length === 1
      ? { next_offset: 8, events: ['old event'] }
      : { next_offset: 4, events: ['reloaded event'] };
  };
  render(React.createElement(Harness, { callBackendTool }));
  await screen.findByText('old event');
  fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
  await screen.findByText('reloaded event');
  assert.deepEqual(offsets.slice(0, 2), [0, 0]);
  assert.equal(screen.queryByText('old event'), null);
});

test('errors recover using the same offset and keep previously received events', async () => {
  const offsets = [];
  const callBackendTool = async (_name, args) => {
    offsets.push(args.offset);
    if (offsets.length === 1) return { next_offset: 4, events: ['first'] };
    if (offsets.length === 2) throw new Error('temporary outage');
    return { next_offset: 8, events: ['second'] };
  };
  render(React.createElement(Harness, { callBackendTool }));
  await screen.findByText('second');
  assert.ok(offsets.includes(4));
  assert.deepEqual([...document.querySelectorAll('li')].map((row) => row.textContent), ['first', 'second']);
});

test('rendered event memory is capped and older event count is retained', async () => {
  const events = Array.from({ length: 2105 }, (_, index) => `event-${index}`);
  render(React.createElement(Harness, { callBackendTool: async () => ({ next_offset: 10, events }) }));
  await waitFor(() => assert.equal(screen.getByTestId('count').textContent, '2000'));
  assert.equal(screen.getByTestId('dropped').textContent, '105');
  assert.equal(document.querySelectorAll('li').length, 2000);
  assert.equal(screen.queryByText('event-0'), null);
  assert.ok(screen.getByText('event-2104'));
});

test('terminal runs read the log up to its end once, then stop polling', async () => {
  const offsets = [];
  const callBackendTool = async (_name, { offset }) => {
    offsets.push(offset);
    if (offset === 0) return { next_offset: 10, events: ['planner started'] };
    if (offset === 10) return { next_offset: 20, events: ['run committed'] };
    return { next_offset: 20, events: [] };
  };
  render(React.createElement(ProgressView, { runId: 'done', status: 'published', intervalMs: 100, callBackendTool }));
  assert.ok(screen.getByText('Run is terminal; live polling stopped.'));
  await screen.findByText('run committed');
  await new Promise((resolve) => setTimeout(resolve, 350));
  assert.deepEqual(offsets, [0, 10, 20]);
  assert.equal(screen.getAllByText('planner started').length, 1);
});

test('unmount stops future polling and ignores the pending response', async () => {
  let release;
  let calls = 0;
  const callBackendTool = async () => {
    calls += 1;
    return new Promise((resolve) => { release = resolve; });
  };
  const view = render(React.createElement(Harness, { callBackendTool }));
  await waitFor(() => assert.equal(calls, 1));
  view.unmount();
  await act(async () => release({ next_offset: 1, events: ['after unmount'] }));
  await new Promise((resolve) => setTimeout(resolve, 150));
  assert.equal(calls, 1);
});
