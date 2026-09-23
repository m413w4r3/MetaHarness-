import type {
  BackendActivateContext,
  MetaHarnessBackend,
  MetaHarnessStatus,
} from './types';

const STATUS: MetaHarnessStatus = {
  configured: false,
  connected: false,
};

export async function activate(
  context: BackendActivateContext
): Promise<MetaHarnessBackend> {
  await context.services.registerMcpTools([
    {
      name: 'status',
      description: 'Return the current MetaHarness configuration and connection status.',
      inputSchema: {
        type: 'object',
        properties: {},
      },
      scope: 'global',
    },
  ]);

  context.services.log('info', '[metaharness] status tool registered');

  return {
    methods: {
      status: async (): Promise<MetaHarnessStatus> => ({ ...STATUS }),
    },
    deactivate: async () => undefined,
  };
}
