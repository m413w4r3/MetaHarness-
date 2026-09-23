import { builtinModules } from 'node:module';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  mode: 'production',
  build: {
    lib: {
      entry: resolve(process.cwd(), 'src/runtimeBackend.ts'),
      formats: ['es'],
      fileName: () => 'backend-runtime.js',
    },
    rollupOptions: {
      external: [/^node:/, ...builtinModules],
      output: {
        inlineDynamicImports: true,
      },
    },
    target: 'node18',
    outDir: 'dist',
    emptyOutDir: false,
    sourcemap: true,
    minify: false,
  },
});
