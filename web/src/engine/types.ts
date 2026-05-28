// =====================================================================
// Shared contracts between the engine layer (agent B / agent C) and the UI.
// These types are the integration glue — do not change a field without
// updating both the producer and the consumer.
// =====================================================================

/** A single move suggestion from a recommender (the net or Stockfish). */
export interface MoveSuggestion {
  /** Long algebraic / UCI, e.g. "e2e4", "e7e8q". */
  uci: string;
  /** Standard algebraic notation in the current position, e.g. "e4", "Qxe7+". */
  san: string;
  /** Network policy probability in [0,1]. Present for net suggestions only. */
  prob?: number;
}

/** Result of running the chess-nn network on a position (raw policy, no search). */
export interface NetResult {
  /** Legal moves only, sorted descending by `prob`. */
  topMoves: MoveSuggestion[];
  /** [win, draw, loss] probabilities (sum to 1), from the side-to-move POV. */
  wdl: [number, number, number];
  /** P(win) - P(loss), in [-1, 1]. Convenience scalar for an eval bar. */
  scalar: number;
}

/** Stockfish analysis of a position (the "strong engine" side of the compare). */
export interface SfAnalysis {
  /** Stockfish's best move in the position. */
  best: MoveSuggestion;
  /** Centipawn score from the side-to-move POV; null when forced mate. */
  scoreCp: number | null;
  /** Mate-in-N (signed) from the side-to-move POV; null when no forced mate. */
  mate: number | null;
  /** Search depth reached. */
  depth: number;
  /** Principal variation as UCI strings. */
  pv: string[];
}

// ---------------------------------------------------------------------
// Engine factory contracts.
// ---------------------------------------------------------------------

/**
 * The chess-nn network engine. IMPLEMENTED BY agent B in `net.ts` as:
 *   export function createNetEngine(modelUrl?: string): NetEngine
 *
 * `evaluate` replays `movesUci` from `startFen` to rebuild the 8-frame board
 * history exactly like training (current frame first, then most-recent-first,
 * truncated to 8), encodes the 105-plane tensor, runs the ONNX model, then
 * legal-masks + softmaxes the policy. This signature intentionally mirrors the
 * parity-fixture format `{ startFen, moves }` so the parity test is a direct
 * call. For a freshly loaded FEN with no move history, pass `movesUci = []`.
 */
export interface NetEngine {
  ready(): Promise<void>;
  evaluate(startFen: string, movesUci: string[]): Promise<NetResult>;
}

/**
 * A Stockfish (WASM) worker wrapper. IMPLEMENTED BY agent C in `stockfish.ts` as:
 *   export function createStockfish(): StockfishEngine
 *
 * Two instances are used in the UI: an "opponent" with setElo(<n>) that plays
 * the visitor via bestMove(), and an "analyst" with setElo(null) (full
 * strength) queried via analyze() for the side-by-side comparison + eval bar.
 */
export interface StockfishEngine {
  ready(): Promise<void>;
  /** null => full strength; a number => UCI_LimitStrength + UCI_Elo (or Skill Level fallback below the UCI_Elo floor). */
  setElo(elo: number | null): Promise<void>;
  /** Engine's move for actually playing, time-bounded. */
  bestMove(fen: string, opts?: { movetimeMs?: number }): Promise<MoveSuggestion>;
  /** Fixed-depth analysis for the comparison panel. */
  analyze(fen: string, opts?: { depth?: number }): Promise<SfAnalysis>;
  /** Release the worker. */
  dispose(): void;
}
