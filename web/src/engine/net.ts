// =====================================================================
// NetEngine — in-browser inference of the chess-nn v3 ONNX model via
// onnxruntime-web. Replays the move history into the 105-plane tensor,
// runs the model, and post-processes policy/value into a NetResult.
// =====================================================================
import * as ort from "onnxruntime-web";
import { Chess } from "chess.js";
import type { NetEngine, NetResult } from "./types";
import { encodeGame } from "./boardEncoding";
import { buildNetResult } from "./policy";

const BASE = (import.meta.env.BASE_URL ?? "/") as string;
const DEFAULT_MODEL_URL = BASE + "model/chessnet-v3.onnx";

export function createNetEngine(modelUrl?: string): NetEngine {
  const url = modelUrl ?? DEFAULT_MODEL_URL;

  // onnxruntime-web wasm config (browser). Single-threaded keeps it simple and
  // avoids cross-origin-isolation requirements for SharedArrayBuffer. The wasm
  // runtime is loaded from jsDelivr (pinned to the installed version): ort
  // dynamically imports its .mjs glue, which Vite refuses to serve from /public,
  // and self-hosting the ~26MB runtime bloats the deploy. The custom model
  // itself is still self-hosted (DEFAULT_MODEL_URL above).
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
  ort.env.wasm.numThreads = 1;

  let sessionPromise: Promise<ort.InferenceSession> | null = null;

  // The single-threaded wasm session is not safe to run concurrently. React
  // StrictMode fires effects twice on mount, which otherwise launches two
  // overlapping evaluate() calls and makes one of them throw. Serialize every
  // run through a queue so calls execute one at a time.
  let queue: Promise<unknown> = Promise.resolve();
  const serialize = <T>(fn: () => Promise<T>): Promise<T> => {
    const run = queue.then(fn, fn);
    queue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  };

  const getSession = (): Promise<ort.InferenceSession> => {
    if (sessionPromise === null) {
      sessionPromise = ort.InferenceSession.create(url, {
        executionProviders: ["wasm"],
      });
    }
    return sessionPromise;
  };

  return {
    async ready(): Promise<void> {
      await getSession();
    },

    async evaluate(startFen: string, movesUci: string[]): Promise<NetResult> {
      return serialize(async () => {
        const session = await getSession();

        const data = encodeGame(startFen, movesUci);
        const inputTensor = new ort.Tensor("float32", data, [1, 105, 8, 8]);

        const outputs = await session.run({ input: inputTensor });
        const policy = outputs.policy.data as Float32Array;
        const value = outputs.value.data as Float32Array;

        // Reconstruct the current position so we can legal-mask + produce SAN.
        const chess = new Chess(startFen);
        for (const uci of movesUci) {
          chess.move({
            from: uci.slice(0, 2),
            to: uci.slice(2, 4),
            promotion: uci.length > 4 ? uci[4] : undefined,
          });
        }

        return buildNetResult(policy, value, chess);
      });
    },
  };
}
