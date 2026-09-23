import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { activate, CONTROL_TOOL_DESCRIPTORS } from '../dist/backend-runtime.js';

const root = mkdtempSync(join(tmpdir(), 'nimbalyst-metaharness-runtime-'));

test('Nimbalyst runtime entrypoint registers UI process-control tools', async () => {
  const registered = [];
  const backend = await activate({
    services: {
      workspacePath: root,
      extensionPath: root,
      dataDir: join(root, 'data'),
      log: () => undefined,
      registerMcpTools: async (tools) => {
        registered.push(...tools.map((tool) => tool.name));
        return { registered: tools.map((tool) => tool.name) };
      },
    },
  }, { port: 65534, autoStart: false });

  for (const name of ['status', 'start', 'stop', 'doctor', 'list_runs', 'create_run']) {
    assert.ok(registered.includes(name), `missing registered backend tool: ${name}`);
  }
  assert.equal(new Set(registered).size, registered.length, 'backend tool names must be unique');
  assert.deepEqual(CONTROL_TOOL_DESCRIPTORS.map((tool) => tool.name), ['start', 'stop', 'doctor']);

  await backend.deactivate();
});
