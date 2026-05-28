/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static SPA. `base: "./"` keeps asset URLs relative so the build works whether
// served from a domain root (Vercel) or a subpath (GitHub Pages project page).
export default defineConfig({
  base: "./",
  plugins: [react()],
  // onnxruntime-web ships its own wasm and does not play well with Vite's dep
  // pre-bundling; exclude it so it is loaded as-is at runtime.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
  test: {
    // Parity tests are pure compute (encoding + ONNX inference) — no DOM needed.
    environment: "node",
    include: ["src/**/*.test.ts"],
    testTimeout: 30000,
  },
});
