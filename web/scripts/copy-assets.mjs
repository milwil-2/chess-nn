// Copies the standard runtime binaries (onnxruntime-web wasm + Stockfish) from
// node_modules into public/ so they are self-hosted (no CDN dependency, works
// offline) WITHOUT committing ~50MB of third-party binaries to git.
//
// public/ort and public/stockfish are gitignored; this regenerates them. Wired
// to run automatically via `predev` and `prebuild` in package.json, so Vercel /
// any CI that runs `npm install && npm run build` gets them too.
//
// The custom model (public/model/*) is produced by web/scripts/export_onnx.py
// and IS committed — it is not touched here.

import { mkdirSync, copyFileSync, existsSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nm = join(root, "node_modules");

/** [sourceAbsPath, destAbsPath] pairs. */
const jobs = [];

function add(srcRel, destRel) {
  jobs.push([join(nm, srcRel), join(root, destRel)]);
}

// onnxruntime-web wasm is loaded from jsDelivr at runtime (see net.ts) — not
// copied here. ort dynamically imports its .mjs glue, which Vite won't serve
// from /public, and self-hosting the ~26MB runtime bloats the deploy.

// Stockfish 16, single-threaded build (no COOP/COEP headers required) + its
// NNUE weights. The engine fetches the .wasm and the .nnue relative to the
// worker script, so all three must sit together in public/stockfish/.
const sfSrc = "stockfish/src";
for (const f of [
  "stockfish-nnue-16-single.js",
  "stockfish-nnue-16-single.wasm",
  "nn-5af11540bbfe.nnue",
]) {
  add(`${sfSrc}/${f}`, `public/stockfish/${f}`);
}

let copied = 0;
let bytes = 0;
for (const [src, dest] of jobs) {
  if (!existsSync(src)) {
    console.warn(`[copy-assets] MISSING source, skipping: ${src}`);
    continue;
  }
  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(src, dest);
  copied++;
  bytes += statSync(dest).size;
}
console.log(
  `[copy-assets] copied ${copied}/${jobs.length} files (${(bytes / 1e6).toFixed(1)} MB) into public/`,
);
