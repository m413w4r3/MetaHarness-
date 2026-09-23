import { activate as activateBase } from './backend';
import type {
  BackendActivateContext,
  BackendToolDescriptor,
  MetaHarnessBackend,
  RuntimeConfig,
} from './types';

const SETTINGS_PROPERTIES = {
  executable: { type: 'string' },
  configPath: { type: 'string' },
  port: { type: 'integer' },
  autoStart: { type: 'boolean' },
  pollIntervalMs: { type: 'integer' },
} as const;

function settingsSchema(): BackendToolDescriptor['inputSchema'] {
  return {
    type: 'object',
    properties: {
      settings: {
        type: 'object',
        properties: SETTINGS_PROPERTIES,
        additionalProperties: false,
      },
    },
    additionalProperties: false,
  };
}

export const CONTROL_TOOL_DESCRIPTORS: BackendToolDescriptor[] = [
  {
    name: 'start',
    description:
      'LOCAL PROCESS ACTION. Start or attach to MetaHarness for the supplied settings. ' +
      'Intended for an explicit UI connection or configured auto-start. AI agents must not ' +
      'call this unless the user explicitly asks to start MetaHarness.',
    inputSchema: settingsSchema(),
    scope: 'global',
  },
  {
    name: 'stop',
    description:
      'LOCAL PROCESS ACTION. Stop only a MetaHarness process owned by this extension. ' +
      'AI agents must not call this unless the user explicitly asks to stop MetaHarness.',
    inputSchema: {
      type: 'object',
      properties: {},
      additionalProperties: false,
    },
    scope: 'global',
  },
  {
    name: 'doctor',
    description:
      'READ-ONLY DIAGNOSTIC. Run metaharness doctor --json for the supplied settings and ' +
      'return its structured report. This does not mutate a MetaHarness run.',
    inputSchema: settingsSchema(),
    scope: 'global',
  },
];

function withControlTools(tools: BackendToolDescriptor[]): BackendToolDescriptor[] {
  const merged = new Map<string, BackendToolDescriptor>();
  for (const tool of [...tools, ...CONTROL_TOOL_DESCRIPTORS]) merged.set(tool.name, tool);
  return [...merged.values()];
}

/**
 * Nimbalyst-facing backend entrypoint.
 *
 * The core backend deliberately keeps process-control helpers out of its agent-facing
 * descriptor list. The Nimbalyst renderer bridge can only invoke methods registered
 * through registerMcpTools, however, so this host adapter advertises the three UI
 * control methods that already exist in MetaHarnessBackend.methods.
 */
export async function activate(
  context: BackendActivateContext,
  runtimeConfig?: Partial<RuntimeConfig>,
): Promise<MetaHarnessBackend> {
  const registerMcpTools = context.services.registerMcpTools;
  const wrappedContext: BackendActivateContext = {
    ...context,
    services: {
      ...context.services,
      registerMcpTools: (tools) => registerMcpTools(withControlTools(tools)),
    },
  };
  return activateBase(wrappedContext, runtimeConfig);
}
