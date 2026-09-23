import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  activate,
  MetaHarnessBackendError,
  MetaHarnessClient,
  MetaHarnessRuntime,
} from '../dist/backend.js';

const root = mkdtempSync(join(tmpdir(), 'nimbalyst-metaharness-'));

function json(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

async function listen(handler) {
  const server = createServer(handler);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return server;
}

function port(server) {
  return server.address().port;
}

async function close(server) {
  await new Promise((resolve) => server.close(resolve));
}

function tokenFile(name, value = 'test-token\n') {
  const path = join(root, name);
  writeFileSync(path, value);
  return path;
}

function errorCode(code, extra = {}) {
  return (error) => error instanceof MetaHarnessBackendError
    && error.code === code
    && Object.entries(extra).every(([key, value]) => error[key] === value);
}

test('health OK, reads do not send the mutation token, and mutation headers are sent', async () => {
  let seen;
  const server = await listen((req, res) => {
    seen = { method: req.method, path: req.url, token: req.headers['x-metaharness-token'] };
    if (req.url === '/api/v1/health') return json(res, 200, { service: 'metaharness', status: 'ok' });
    if (req.method === 'GET' && req.url === '/api/v1/runs') return json(res, 200, { runs: [] });
    return json(res, 404, { message: 'not found' });
  });
  try {
    const client = new MetaHarnessClient({
      baseUrl: `http://127.0.0.1:${port(server)}`,
      tokenFile: tokenFile('health.token'),
    });
    assert.equal((await client.health()).service, 'metaharness');
    await client.listRuns();
    assert.equal(seen.method, 'GET');
    assert.equal(seen.token, undefined);
    await assert.rejects(client.createRun({ spec: 'x' }), errorCode('HTTP_ERROR', { httpStatus: 404 }));
  } finally {
    await close(server);
  }
});

test('health timeout is bounded', async () => {
  const server = await listen((_req, _res) => undefined);
  try {
    const client = new MetaHarnessClient({
      baseUrl: `http://127.0.0.1:${port(server)}`,
      tokenFile: tokenFile('timeout.token'),
      requestTimeoutMs: 40,
    });
    await assert.rejects(client.health(), errorCode('TIMEOUT'));
  } finally {
    await close(server);
  }
});

test('get_artifact reads a named text artifact without mutation credentials', async () => {
  let seen;
  const server = await listen((req, res) => {
    seen = { method: req.method, path: req.url, token: req.headers['x-metaharness-token'] };
    return json(res, 200, { run_id: 'run-1', name: 'diagnostics.json', content: '{"ok":true}', truncated: false });
  });
  try {
    const client = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('artifact.token') });
    const artifact = await client.getArtifact('run-1', 'diagnostics.json');
    assert.equal(artifact.content, '{"ok":true}');
    assert.deepEqual(seen, { method: 'GET', path: '/api/v1/runs/run-1/artifact?name=diagnostics.json', token: undefined });
    await assert.rejects(client.getArtifact('run-1', '../state.json'), errorCode('INVALID_ARGUMENT'));
  } finally { await close(server); }
});

test('403 mutation, invalid JSON, and missing token are standardized', async () => {
  const forbidden = await listen((_req, res) => json(res, 403, { message: 'mutation token required' }));
  try {
    const client = new MetaHarnessClient({
      baseUrl: `http://127.0.0.1:${port(forbidden)}`,
      tokenFile: tokenFile('forbidden.token'),
    });
    await assert.rejects(client.resumeRun('run-1'), errorCode('HTTP_ERROR', { httpStatus: 403 }));
  } finally {
    await close(forbidden);
  }

  const invalid = await listen((_req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{invalid');
  });
  try {
    const client = new MetaHarnessClient({
      baseUrl: `http://127.0.0.1:${port(invalid)}`,
      tokenFile: tokenFile('invalid.token'),
    });
    await assert.rejects(client.health(), errorCode('JSON_INVALID'));
  } finally {
    await close(invalid);
  }

  const absent = new MetaHarnessClient({
    baseUrl: 'http://127.0.0.1:8766',
    tokenFile: join(root, 'does-not-exist.token'),
  });
  await assert.rejects(absent.resumeRun('run-1'), errorCode('TOKEN_MISSING'));
  assert.throws(
    () => new MetaHarnessClient({ baseUrl: 'http://localhost:8766', tokenFile: tokenFile('host.token') }),
    errorCode('INVALID_BASE_URL'),
  );
});

test('approval bridge flattens conceptual step profile map and keeps reject body minimal', async () => {
  const seen = [];
  const server = await listen(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    seen.push({ path: req.url, body: JSON.parse(Buffer.concat(chunks).toString('utf8')) });
    return json(res, 200, { ok: true });
  });
  try {
    const client = new MetaHarnessClient({
      baseUrl: `http://127.0.0.1:${port(server)}`,
      tokenFile: tokenFile('approval.token'),
    });
    await client.approveRun('run-1', {
      decision: 'APPROVE', final_reviewer_profile: 'reviewer',
      step_profiles: { S01: 'mechanical', S02: 'reasoning' },
    });
    await client.approveRun('run-1', { decision: 'REJECT' });
    assert.deepEqual(seen, [
      { path: '/api/v1/runs/run-1/approval', body: {
        decision: 'APPROVE', final_reviewer_profile: 'reviewer',
        step_profile__S01: 'mechanical', step_profile__S02: 'reasoning',
      } },
      { path: '/api/v1/runs/run-1/approval', body: { decision: 'REJECT' } },
    ]);
  } finally {
    await close(server);
  }
});

function fakeExecutable(name, doctorOutput = null) {
  const script = join(root, `${name}.mjs`);
  writeFileSync(script, `#!/usr/bin/env node
import { createServer } from 'node:http';
import { writeFileSync } from 'node:fs';
const args = process.argv.slice(2);
if (args[0] === 'doctor') {
  process.stdout.write(${JSON.stringify(doctorOutput ?? '{"ok":true}')} + '\\n');
  process.exit(0);
}
const port = Number(args[args.indexOf('--port') + 1]);
const tokenFile = args[args.indexOf('--control-token-file') + 1];
writeFileSync(tokenFile, 'fake-token\\n');
const server = createServer((req, res) => {
  if (req.url === '/api/v1/health') {
    const body = JSON.stringify({ service: 'metaharness', status: 'ok' });
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
    res.end(body);
    return;
  }
  const body = JSON.stringify({ ok: true });
  res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
});
server.listen(port, '127.0.0.1');
process.on('SIGTERM', () => server.close(() => process.exit(0)));
`);
  chmodSync(script, 0o755);
  return script;
}

async function freePort() {
  const server = await listen((_req, res) => res.end());
  const value = port(server);
  await close(server);
  return value;
}

test('start spawns when absent, attaches when present, and stop does not kill an external server', async () => {
  const executable = fakeExecutable('web-server');
  const runtime = new MetaHarnessRuntime({
    config: { executable, configPath: join(root, 'config.toml'), port: await freePort(), autoStart: false },
    dataDir: join(root, 'owned-data'),
  });
  const started = await runtime.start();
  assert.equal(started.connected, true);
  assert.equal(started.serverOwned, true);
  await runtime.stop();
  assert.equal(runtime.ownedProcess, undefined);

  const external = await listen((req, res) => {
    if (req.url === '/api/v1/health') return json(res, 200, { service: 'metaharness', status: 'ok' });
    return json(res, 200, { ok: true });
  });
  try {
    const attached = new MetaHarnessRuntime({
      config: { executable: fakeExecutable('must-not-start'), configPath: 'config.toml', port: port(external), autoStart: false },
      dataDir: join(root, 'external-data'),
    });
    const status = await attached.start();
    assert.equal(status.connected, true);
    assert.equal(status.serverOwned, false);
    await attached.stop();
    assert.equal((await attached.client.health()).service, 'metaharness');
  } finally {
    await close(external);
  }
});

test('start refuses a port occupied by a different service', async () => {
  const foreign = await listen((_req, res) => {
    res.writeHead(200, { 'content-type': 'text/plain' });
    res.end('not MetaHarness');
  });
  try {
    const runtime = new MetaHarnessRuntime({
      config: { executable: fakeExecutable('must-not-start-foreign'), configPath: 'config.toml', port: port(foreign), autoStart: false },
      dataDir: join(root, 'foreign-data'),
    });
    await assert.rejects(runtime.start(), errorCode('PORT_IN_USE'));
  } finally {
    await close(foreign);
  }
});

test('doctor rejects invalid JSON', async () => {
  const runtime = new MetaHarnessRuntime({
    config: {
      executable: fakeExecutable('bad-doctor', 'not-json'),
      configPath: join(root, 'config.toml'),
      port: 8766,
      autoStart: false,
    },
    dataDir: join(root, 'doctor-data'),
  });
  await assert.rejects(runtime.doctor(), errorCode('DOCTOR_INVALID_JSON'));
});

test('activate registers the complete tool surface with default config safely', async () => {
  const registered = [];
  const backend = await activate({
    services: {
      workspacePath: root,
      extensionPath: root,
      log: () => undefined,
      registerMcpTools: async (tools) => {
        registered.push(...tools.map((tool) => tool.name));
        return { registered: tools.map((tool) => tool.name) };
      },
    },
  }, { port: await freePort() });
  assert.deepEqual(registered, [
    'status', 'start', 'stop', 'get_config', 'model_profiles', 'list_runs',
    'get_run', 'get_artifact', 'progress', 'create_run', 'approve_run', 'approve_scope',
    'resume_run', 'recover_plan', 'doctor',
  ]);
  assert.deepEqual(await backend.methods.status(), {
    configured: false,
    connected: false,
    serverOwned: false,
  });
  await backend.deactivate();
});
