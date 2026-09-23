import type { ExtensionModule } from '@nimbalyst/extension-sdk';
import './styles.css';
import { MetaHarnessPanel } from './panel/MetaHarnessPanel';
import { MetaHarnessSettings } from './settings/MetaHarnessSettings';

export const panels = {
  metaharness: {
    component: MetaHarnessPanel,
  },
};

export const settingsPanel = {
  MetaHarnessSettings,
};

const extension: ExtensionModule = {
  panels,
  settingsPanel,
};

export default extension;
