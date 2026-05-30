// Stockfish 16 single-threaded WASM UCI wrapper. The single-threaded build
// runs WITHOUT COOP/COEP headers, so it works on a plain static host. Assets
// (worker .js, .wasm, NNUE) ship together in public/stockfish/.
import { Chess } from "chess.js";
import type { MoveSuggestion, SfAnalysis, StockfishEngine } from "./types";

/** SF16 honours UCI_Elo in this band. Below the floor we fall back to Skill Level. */
const UCI_ELO_MIN = 1320;
const UCI_ELO_MAX = 3190;

type Listener = (line: string) => void;

function uciToSan(fen: string, uci: string): string {
  if (!uci || uci.length < 4) return uci;
  try {
    const game = new Chess(fen);
    const move = game.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci.length > 4 ? uci[4] : undefined,
    });
    return move.san;
  } catch {
    return uci;
  }
}

export function createStockfish(): StockfishEngine {
  let worker: Worker | null = null;
  let bootPromise: Promise<void> | null = null;

  const listeners = new Set<Listener>();

  // Serialize commands so two overlapping bestMove/analyze calls can't
  // interleave their info/bestmove lines on the shared UCI stream.
  let queue: Promise<unknown> = Promise.resolve();

  function post(cmd: string) {
    worker?.postMessage(cmd);
  }

  function send<T>(cmd: string | string[], predicate: (line: string) => T | undefined): Promise<T> {
    return new Promise<T>((resolve) => {
      const listener: Listener = (line) => {
        const result = predicate(line);
        if (result !== undefined) {
          listeners.delete(listener);
          resolve(result);
        }
      };
      listeners.add(listener);
      const cmds = Array.isArray(cmd) ? cmd : [cmd];
      for (const c of cmds) post(c);
    });
  }

  function enqueue<T>(job: () => Promise<T>): Promise<T> {
    const run = queue.then(job, job);
    // Keep the queue alive regardless of individual job outcome.
    queue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  function boot(): Promise<void> {
    if (bootPromise) return bootPromise;
    bootPromise = (async () => {
      const base = import.meta.env.BASE_URL ?? "/";
      worker = new Worker(base + "stockfish/stockfish-nnue-16-single.js");
      worker.onmessage = (e: MessageEvent) => {
        const line = typeof e.data === "string" ? e.data : String(e.data ?? "");
        for (const l of [...listeners]) l(line);
      };
      await send("uci", (l) => (l.startsWith("uciok") ? true : undefined));
      // SF16 resolves EvalFile relative to the worker script, so the bare
      // filename matches our public/stockfish/ layout.
      post("setoption name EvalFile value nn-5af11540bbfe.nnue");
      post("setoption name Use NNUE value true");
      await send("isready", (l) => (l.startsWith("readyok") ? true : undefined));
    })();
    return bootPromise;
  }

  async function ready(): Promise<void> {
    await boot();
  }

  async function setElo(elo: number | null): Promise<void> {
    await boot();
    return enqueue(async () => {
      if (elo === null) {
        post("setoption name UCI_LimitStrength value false");
        post("setoption name Skill Level value 20");
      } else if (elo >= UCI_ELO_MIN) {
        const clamped = Math.min(UCI_ELO_MAX, Math.round(elo));
        post("setoption name Skill Level value 20");
        post("setoption name UCI_LimitStrength value true");
        post(`setoption name UCI_Elo value ${clamped}`);
      } else {
        // Below the UCI_Elo floor: map onto Skill Level 0..20.
        // ~800 Elo -> skill 0, the 1320 floor -> skill ~8.
        const t = Math.max(0, (elo - 600) / (UCI_ELO_MIN - 600));
        const skill = Math.max(0, Math.min(20, Math.round(t * 8)));
        post("setoption name UCI_LimitStrength value false");
        post(`setoption name Skill Level value ${skill}`);
      }
      await send("isready", (l) => (l.startsWith("readyok") ? true : undefined));
    });
  }

  async function bestMove(fen: string, opts?: { movetimeMs?: number }): Promise<MoveSuggestion> {
    await boot();
    const movetime = opts?.movetimeMs ?? 800;
    return enqueue(async () => {
      const uci = await send<string>(
        [`position fen ${fen}`, `go movetime ${movetime}`],
        (line) => {
          if (line.startsWith("bestmove")) {
            return line.split(/\s+/)[1] ?? "";
          }
          return undefined;
        },
      );
      return { uci, san: uciToSan(fen, uci) };
    });
  }

  async function analyze(fen: string, opts?: { depth?: number }): Promise<SfAnalysis> {
    await boot();
    const depth = opts?.depth ?? 15;
    return enqueue(async () => {
      let scoreCp: number | null = null;
      let mate: number | null = null;
      let reachedDepth = 0;
      let pv: string[] = [];

      const bestUci = await send<string>(
        [`position fen ${fen}`, `go depth ${depth}`],
        (line) => {
          if (line.startsWith("info") && line.includes(" pv ")) {
            const depthMatch = line.match(/\bdepth (\d+)/);
            const d = depthMatch ? Number(depthMatch[1]) : 0;
            if (d >= reachedDepth) {
              reachedDepth = d;
              const cpMatch = line.match(/score cp (-?\d+)/);
              const mateMatch = line.match(/score mate (-?\d+)/);
              if (mateMatch) {
                mate = Number(mateMatch[1]);
                scoreCp = null;
              } else if (cpMatch) {
                scoreCp = Number(cpMatch[1]);
                mate = null;
              }
              const pvMatch = line.match(/ pv (.+)$/);
              pv = pvMatch ? pvMatch[1].trim().split(/\s+/) : pv;
            }
          }
          if (line.startsWith("bestmove")) {
            return line.split(/\s+/)[1] ?? "";
          }
          return undefined;
        },
      );

      const best: MoveSuggestion = { uci: bestUci, san: uciToSan(fen, bestUci) };
      if (pv.length && pv[0] !== bestUci) {
        best.uci = pv[0];
        best.san = uciToSan(fen, pv[0]);
      }
      return { best, scoreCp, mate, depth: reachedDepth, pv };
    });
  }

  function dispose(): void {
    worker?.terminate();
    worker = null;
    bootPromise = null;
    listeners.clear();
    queue = Promise.resolve();
  }

  return { ready, setElo, bestMove, analyze, dispose };
}
