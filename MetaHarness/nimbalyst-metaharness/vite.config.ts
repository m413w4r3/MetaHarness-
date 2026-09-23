import react from '@vitejs/plugin-react';
import { createExtensionConfig } from '@nimbalyst/extension-sdk/vite';

export default createExtensionConfig({
  entry: './src/index.tsx',
  plugins: [react({ jsxRuntime: 'automatic' })],
});
