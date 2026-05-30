/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` keeps asset URLs relative so the build works both at a domain
// root (Vercel) and on a subpath (GitHub Pages project page).
export default defineConfig({
  base: "./",
  plugins: [react()],
  // onnxruntime-web ships its own wasm and does not play well with Vite's
  // dep pre-bundling; load it as-is at runtime.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    testTimeout: 30000,
  },
});
