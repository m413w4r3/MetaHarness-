import type { ExtensionAIService, ExtensionContext } from '@nimbalyst/extension-sdk';

let aiService: ExtensionAIService | undefined;

export function bindExtensionRuntime(context: ExtensionContext): void {
  aiService = context.services.ai;
}

export function unbindExtensionRuntime(): void {
  aiService = undefined;
}

export async function callMetaHarnessBackend(
  toolName: string,
  args: Record<string, unknown> = {},
  workspacePath?: string,
): Promise<unknown> {
  if (!aiService) {
    throw new Error('Nimbalyst AI service is unavailable. Ensure permissions.ai is enabled.');
  }
  return aiService.callBackendTool(toolName, args, workspacePath);
}
