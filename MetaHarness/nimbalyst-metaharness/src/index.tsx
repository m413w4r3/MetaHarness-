import type { ExtensionContext, ExtensionModule } from '@nimbalyst/extension-sdk';
import './styles.css';
import { MetaHarnessPanel } from './panel/MetaHarnessPanel';
import { MetaHarnessSettings } from './settings/MetaHarnessSettings';
import { bindExtensionRuntime, unbindExtensionRuntime } from './runtime/extensionRuntime';

export function activate(context: ExtensionContext): void {
  bindExtensionRuntime(context);
}

export function deactivate(): void {
  unbindExtensionRuntime();
}

export const panels = {
  metaharness: {
    component: MetaHarnessPanel,
  },
};

export const settingsPanel = {
  MetaHarnessSettings,
};

const extension: ExtensionModule = {
  activate,
  deactivate,
  panels,
  settingsPanel,
};

export default extension;
