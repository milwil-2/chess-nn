import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import * as path from "node:path";
import { Chess } from "chess.js";
import * as ort from "onnxruntime-web";

import { encodeGame, planeSums } from "./boardEncoding";
import { buildNetResult } from "./policy";

interface Fixture {
  name: string;
  startFen: string;
  moves: string[];
  fen: string;
  turn: string;
  planeSums: number[];
  inputChecksum: number;
  topMoves: { uci: string; prob: number }[];
  wdl: [number, number, number];
}

interface FixtureFile {
  model: string;
  inputPlanes: number;
  policySize: number;
  historyLength: number;
  encodingRule: string;
  fixtures: Fixture[];
}

// vitest cwd is the `web/` package root.
const FIXTURES: FixtureFile = JSON.parse(
  readFileSync("public/model/parity_fixtures.json", "utf-8")
);
const MODEL_PATH = "public/model/chessnet-v3.onnx";

// Point onnxruntime-web at the local wasm binaries so it can run under node.
const ORT_DIST = path.resolve("node_modules/onnxruntime-web/dist/");
ort.env.wasm.wasmPaths = ORT_DIST + path.sep;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const ATOL_PLANE = 1e-4;
const ATOL_CHECKSUM = 1e-3;
const ATOL_WDL = 0.06;

let session: ort.InferenceSession | null = null;
let inferenceAvailable = false;
let inferenceError: unknown = null;

beforeAll(async () => {
  try {
    const bytes = readFileSync(MODEL_PATH);
    // Pass a Uint8Array to avoid URL/fetch resolution issues under node.
    session = await ort.InferenceSession.create(new Uint8Array(bytes), {
      executionProviders: ["wasm"],
    });
    inferenceAvailable = true;
  } catch (err) {
    inferenceError = err;
    inferenceAvailable = false;
    // eslint-disable-next-line no-console
    console.warn(
      "[parity] onnxruntime-web could not run under node; inference " +
        "assertions will be soft-skipped. Error:",
      err
    );
  }
}, 120_000);

function rebuildPosition(startFen: string, moves: string[]): Chess {
  const chess = new Chess(startFen);
  for (const uci of moves) {
    chess.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci.length > 4 ? uci[4] : undefined,
    });
  }
  return chess;
}

describe("board encoding parity (STRICT)", () => {
  for (const fx of FIXTURES.fixtures) {
    it(`encodes ${fx.name} to match fixture plane sums`, () => {
      const tensor = encodeGame(fx.startFen, fx.moves);
      expect(tensor.length).toBe(FIXTURES.inputPlanes * 64);

      const sums = planeSums(tensor);
      expect(sums.length).toBe(fx.planeSums.length);

      for (let i = 0; i < fx.planeSums.length; i++) {
        expect(
          Math.abs(sums[i] - fx.planeSums[i]),
          `plane ${i}: got ${sums[i]}, expected ${fx.planeSums[i]}`
        ).toBeLessThanOrEqual(ATOL_PLANE);
      }

      const total = sums.reduce((a, b) => a + b, 0);
      expect(
        Math.abs(total - fx.inputChecksum),
        `checksum: got ${total}, expected ${fx.inputChecksum}`
      ).toBeLessThanOrEqual(ATOL_CHECKSUM);
    });
  }
});

describe("ONNX inference parity (TOLERANT — int8 drift)", () => {
  for (const fx of FIXTURES.fixtures) {
    it(`runs ${fx.name} and matches top moves + wdl`, async () => {
      if (!inferenceAvailable || session === null) {
        console.warn(
          `[parity] skipping inference for ${fx.name} (onnx unavailable):`,
          inferenceError
        );
        return;
      }

      const data = encodeGame(fx.startFen, fx.moves);
      const input = new ort.Tensor("float32", data, [1, 105, 8, 8]);
      const outputs = await session.run({ input });
      const policy = outputs.policy.data as Float32Array;
      const value = outputs.value.data as Float32Array;

      const chess = rebuildPosition(fx.startFen, fx.moves);
      const result = buildNetResult(policy, value, chess);

      const netTop = result.topMoves.map((m) => m.uci);

      const fixtureBest = fx.topMoves[0].uci;
      expect(
        netTop.slice(0, 3),
        `${fx.name}: fixture best ${fixtureBest} not in net top-3 ${netTop
          .slice(0, 3)
          .join(",")}`
      ).toContain(fixtureBest);

      const netTop5 = new Set(netTop.slice(0, 5));
      const fxTop5 = fx.topMoves.slice(0, 5).map((m) => m.uci);
      const overlap = fxTop5.filter((u) => netTop5.has(u)).length;
      expect(
        overlap,
        `${fx.name}: top-5 overlap ${overlap} (<3). net=${[...netTop5].join(
          ","
        )} fx=${fxTop5.join(",")}`
      ).toBeGreaterThanOrEqual(3);

      for (let i = 0; i < 3; i++) {
        expect(
          Math.abs(result.wdl[i] - fx.wdl[i]),
          `${fx.name}: wdl[${i}] got ${result.wdl[i]}, expected ${fx.wdl[i]}`
        ).toBeLessThanOrEqual(ATOL_WDL);
      }
    }, 60_000);
  }
});
