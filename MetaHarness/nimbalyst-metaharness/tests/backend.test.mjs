import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  activate,
  MCP_TOOL_DESCRIPTORS,
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
    if (req.url === '/api/v1/health') return json(res, 200, { service: 'metaharness', api_version: 1, status: 'ok' });
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
    return json(res, 200, { run_id: 'run-1', name: 'diagnostics.json', exists: true, encoding: 'utf-8', content: '{"ok":true}', truncated: false, size: 11 });
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
    await assert.rejects(client.resumeRun('run-1'), errorCode('TOKEN_REJECTED', { httpStatus: 403 }));
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

function fakeExecutable(name, doctorOutput = null, doctorExit = 0) {
  const script = join(root, `${name}.mjs`);
  writeFileSync(script, `#!/usr/bin/env node
import { createServer } from 'node:http';
import { writeFileSync } from 'node:fs';
const args = process.argv.slice(2);
if (args[0] === 'doctor') {
  process.stdout.write(${JSON.stringify(doctorOutput ?? '{"ok":true}')} + '\\n');
  process.exit(${doctorExit});
}
const port = Number(args[args.indexOf('--port') + 1]);
const tokenFile = args[args.indexOf('--control-token-file') + 1];
writeFileSync(tokenFile, 'fake-token\\n');
const server = createServer((req, res) => {
  if (req.url === '/api/v1/health') {
    const body = JSON.stringify({ service: 'metaharness', api_version: 1, status: 'ok' });
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
    if (req.url === '/api/v1/health') return json(res, 200, { service: 'metaharness', api_version: 1, status: 'ok' });
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

test('doctor returns the decoded JSON report from the local executable', async () => {
  const report = { ok: true, checks: [{ name: 'config', status: 'pass' }] };
  const runtime = new MetaHarnessRuntime({
    config: { executable: fakeExecutable('good-doctor', JSON.stringify(report)), configPath: join(root, 'doctor.toml'), port: await freePort(), autoStart: false },
    dataDir: join(root, 'doctor-success-data'),
  });
  assert.deepEqual(await runtime.doctor(), report);
});

test('token file decoder rejects multiline and oversized credentials without echoing them', async () => {
  const server = await listen((_req, res) => json(res, 200, { ok: true }));
  try {
    const multiline = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('multiline.token', 'line-one\nline-two\n') });
    await assert.rejects(multiline.resumeRun('run-1'), errorCode('TOKEN_INVALID'));
    const oversized = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('oversized.token', `${'x'.repeat(5000)}\n`) });
    await assert.rejects(oversized.resumeRun('run-1'), errorCode('TOKEN_INVALID'));
  } finally { await close(server); }
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
    'status', 'get_config', 'model_profiles', 'list_runs',
    'get_run', 'get_artifact', 'progress', 'create_run', 'approve_run', 'approve_scope',
    'resume_run', 'recover_plan',
  ]);
  assert.deepEqual(await backend.methods.status(), {
    configured: false,
    connected: false,
    serverOwned: false,
  });
  await backend.deactivate();
});

test('MCP descriptors classify reads and require explicit mutation arguments', () => {
  const descriptors = new Map(MCP_TOOL_DESCRIPTORS.map((tool) => [tool.name, tool]));
  const readOnly = ['status', 'get_config', 'model_profiles', 'list_runs', 'get_run', 'progress', 'get_artifact'];
  const mutations = ['create_run', 'approve_run', 'approve_scope', 'resume_run', 'recover_plan'];
  for (const name of readOnly) assert.match(descriptors.get(name).description, /^READ-ONLY\./);
  for (const name of mutations) {
    const description = descriptors.get(name).description;
    assert.match(description, /MUTATING ACTION/);
    assert.match(description, /user’s explicit request/i);
    assert.match(description, /Never approve or reject a plan automatically/i);
    assert.match(description, /Never recover a plan without .*plan content.*intent/i);
    assert.match(description, /Never launch multiple runs to compensate for an error/i);
  }
  assert.deepEqual(descriptors.get('create_run').inputSchema.required, ['spec']);
  assert.deepEqual(descriptors.get('approve_run').inputSchema.required, ['runId', 'decision']);
  assert.deepEqual(descriptors.get('approve_scope').inputSchema.required, ['runId', 'decision']);
  assert.deepEqual(descriptors.get('resume_run').inputSchema.required, ['runId']);
  assert.deepEqual(descriptors.get('recover_plan').inputSchema.required, ['runId', 'input']);
  assert.deepEqual(descriptors.get('recover_plan').inputSchema.properties.input.required, ['plan']);
  assert.equal(descriptors.has('start'), false);
  assert.equal(descriptors.has('stop'), false);
  assert.equal(descriptors.has('doctor'), false);
});

test('API decoders reject non-object payloads and malformed run collections', async () => {
  const server = await listen((_req, res) => json(res, 200, { runs: 'not-an-array' }));
  try {
    const client = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('decode.token') });
    await assert.rejects(client.listRuns(), errorCode('INVALID_RESPONSE'));
  } finally { await close(server); }

  const scalar = await listen((_req, res) => json(res, 200, ['not', 'an object']));
  try {
    const client = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(scalar)}`, tokenFile: tokenFile('scalar.token') });
    await assert.rejects(client.getConfig(), errorCode('INVALID_RESPONSE'));
  } finally { await close(scalar); }
});

test('mutations send the control token from the token file and never expose it in errors', async () => {
  let seen;
  const server = await listen((req, res) => {
    seen = { method: req.method, token: req.headers['x-metaharness-token'] };
    return json(res, 202, { ok: true, run_id: 'created-1', location: '/runs/created-1', accepted: true });
  });
  try {
    const secret = 'only-in-token-file';
    const client = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('auth.token', `${secret}\n`) });
    assert.deepEqual(await client.createRun({ spec: 'small test' }), { ok: true, run_id: 'created-1', location: '/runs/created-1', accepted: true });
    assert.deepEqual(seen, { method: 'POST', token: secret });
  } finally { await close(server); }
});

test('health rejects an unsupported MetaHarness API version', async () => {
  const server = await listen((_req, res) => json(res, 200, { service: 'metaharness', api_version: 2, status: 'ok' }));
  try {
    const client = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('version.token') });
    await assert.rejects(client.health(), errorCode('API_VERSION_MISMATCH'));
  } finally {
    await close(server);
  }
});

test('run summaries, progress and mutation acknowledgements are decoded against the server contract', async () => {
  const responses = {
    '/api/v1/runs': { runs: [{ run_id: 'ok-1', status: 'planning', updated_at: null, plan_title: null, commit_sha: null, candidate: null, failure: null }, { status: 'failed' }] },
    '/api/v1/runs/ok-1/progress?offset=0': { next_offset: -1, events: [] },
    '/api/v1/runs/ok-1/resume': { run_id: 'ok-1' },
  };
  const server = await listen((req, res) => json(res, req.method === 'POST' ? 202 : 200, responses[req.url] ?? {}));
  try {
    const client = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${port(server)}`, tokenFile: tokenFile('contract.token') });
    await assert.rejects(client.listRuns(), errorCode('INVALID_RESPONSE'));
    await assert.rejects(client.progress('ok-1', 0), errorCode('INVALID_RESPONSE'));
    await assert.rejects(client.resumeRun('ok-1'), errorCode('INVALID_RESPONSE'));
  } finally {
    await close(server);
  }
});

test('doctor keeps the structured report when a check fails with a non-zero exit', async () => {
  const report = { ok: false, checks: [{ id: 'credentials', status: 'fail', message: 'required credential is missing or invalid' }], summary: { passed: 0, failed: 1, warnings: 0 } };
  const runtime = new MetaHarnessRuntime({
    config: { executable: fakeExecutable('failing-doctor', JSON.stringify(report), 1), configPath: join(root, 'doctor.toml'), port: await freePort(), autoStart: false },
    dataDir: join(root, 'doctor-failing-data'),
  });
  assert.deepEqual(await runtime.doctor(), report);
  const broken = new MetaHarnessRuntime({
    config: { executable: fakeExecutable('broken-doctor', 'Traceback', 2), configPath: join(root, 'doctor.toml'), port: await freePort(), autoStart: false },
    dataDir: join(root, 'doctor-broken-data'),
  });
  await assert.rejects(broken.doctor(), errorCode('DOCTOR_FAILED'));
});

function registerOnly() {
  return { workspacePath: root, extensionPath: root, log: () => undefined, registerMcpTools: async () => undefined };
}

test('read and mutation tools follow the service selected by the panel settings', async () => {
  const hits = [];
  const serve = (name) => listen((req, res) => {
    hits.push(`${name} ${req.url}`);
    if (req.url === '/api/v1/health') return json(res, 200, { service: 'metaharness', api_version: 1, status: 'ok' });
    return json(res, 200, { runs: [] });
  });
  const first = await serve('default');
  const second = await serve('selected');
  try {
    const backend = await activate({ dataDir: join(root, 'routing-data'), services: registerOnly() }, { port: port(first), autoStart: false });
    const settings = { executable: 'metaharness', configPath: '/work/metaharness.toml', port: port(second), autoStart: true, pollIntervalMs: 1000 };
    assert.equal((await backend.methods.status({ settings })).connected, true);
    assert.deepEqual(await backend.methods.list_runs(), []);
    assert.ok(hits.includes('selected /api/v1/runs'));
    assert.ok(!hits.includes('default /api/v1/runs'));
    // A poll-interval change is a client preference, not another service.
    assert.equal((await backend.methods.status({ settings: { ...settings, pollIntervalMs: 5000 } })).connected, true);
    assert.deepEqual(await backend.methods.list_runs(), []);
    assert.ok(!hits.includes('default /api/v1/runs'));
    await backend.deactivate();
  } finally {
    await close(first);
    await close(second);
  }
});

test('reconfiguring the same port replaces only the owned process and deactivate leaves none running', async () => {
  const executable = fakeExecutable('reconfigured-server');
  const servicePort = await freePort();
  const backend = await activate({ dataDir: join(root, 'reconfigure-data'), services: registerOnly() }, { port: servicePort, autoStart: false });
  const settings = { executable, configPath: join(root, 'one.toml'), port: servicePort, autoStart: true, pollIntervalMs: 1000 };
  const started = await backend.methods.start({ settings });
  assert.equal(started.serverOwned, true);
  // A second start with the same identity attaches to the owned process.
  assert.equal((await backend.methods.start({ settings: { ...settings, pollIntervalMs: 2000 } })).serverOwned, true);
  const replaced = await backend.methods.start({ settings: { ...settings, configPath: join(root, 'two.toml') } });
  assert.equal(replaced.connected, true);
  assert.equal(replaced.serverOwned, true);
  await backend.deactivate();
  const probe = new MetaHarnessClient({ baseUrl: `http://127.0.0.1:${servicePort}`, tokenFile: tokenFile('after.token') });
  await assert.rejects(probe.health(), errorCode('NETWORK_ERROR'));
});

test('synchronous argument errors are returned as failure envelopes', async () => {
  const backend = await activate({ dataDir: join(root, 'sync-error-data'), services: registerOnly() }, { port: await freePort(), autoStart: false });
  assert.deepEqual(await backend.methods.create_run({ spec: '   ' }), { ok: false, error: { code: 'INVALID_ARGUMENT', message: 'spec is required' } });
  assert.equal((await backend.methods.get_run({ runId: '../x' })).error.code, 'INVALID_ARGUMENT');
  assert.equal((await backend.methods.progress({ runId: 'r', offset: -1 })).error.code, 'INVALID_ARGUMENT');
  await backend.deactivate();
});
