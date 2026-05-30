// Shared contracts between the engine layer and the UI components.

export interface MoveSuggestion {
  /** Long algebraic / UCI, e.g. "e2e4", "e7e8q". */
  uci: string;
  /** SAN in the position, e.g. "e4", "Qxe7+". */
  san: string;
  /** Policy probability in [0,1]; only set on net suggestions. */
  prob?: number;
}

/** Output of the chess-nn network on a position (raw policy, no search). */
export interface NetResult {
  /** Legal moves only, sorted descending by `prob`. */
  topMoves: MoveSuggestion[];
  /** [win, draw, loss] from the side-to-move POV; sums to 1. */
  wdl: [number, number, number];
  /** P(win) − P(loss) ∈ [−1, 1]. */
  scalar: number;
}

/** One line from a MultiPV-mode Stockfish analysis. */
export interface SfLine {
  uci: string;
  san: string;
  scoreCp: number | null;
  mate: number | null;
  pv: string[];
}

/** Stockfish analysis of a position (the reference side of the compare). */
export interface SfAnalysis {
  /** Best line (multipv slot 1). */
  best: MoveSuggestion;
  /** Centipawns from the STM POV; null when the score is a forced mate. */
  scoreCp: number | null;
  /** Signed mate-in-N from the STM POV; null when not a forced mate. */
  mate: number | null;
  depth: number;
  /** Principal variation of the best line, as UCI strings. */
  pv: string[];
  /** Top-N lines sorted by multipv index (slot 1 = best). Includes `best` at [0]. */
  alternatives: SfLine[];
}

export interface NetEngine {
  ready(): Promise<void>;
  /**
   * Replays `movesUci` from `startFen` to rebuild the 8-frame history (current
   * frame first, most-recent-first, truncated to 8), encodes the 105-plane
   * tensor, runs the ONNX model, then legal-masks + softmaxes the policy.
   * Signature mirrors the parity-fixture format so the test is a direct call.
   */
  evaluate(startFen: string, movesUci: string[]): Promise<NetResult>;
}

export interface StockfishEngine {
  ready(): Promise<void>;
  /** null = full strength; a number sets UCI_LimitStrength + UCI_Elo (Skill Level fallback under the floor). */
  setElo(elo: number | null): Promise<void>;
  /** Time-bounded best move for actually playing. */
  bestMove(fen: string, opts?: { movetimeMs?: number }): Promise<MoveSuggestion>;
  /** Analysis of the position, bounded by depth or movetime (movetime wins when both given). Always returns the top lines from MultiPV mode in `alternatives`. */
  analyze(fen: string, opts?: { depth?: number; movetimeMs?: number }): Promise<SfAnalysis>;
  dispose(): void;
}
