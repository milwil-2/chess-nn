// =====================================================================
// Policy + value post-processing — TypeScript port of
//   models/v3_vast/chess_nn/move_encoding.py :: policy_to_moves
//   models/v3_vast/chess_nn/model.py        :: wdl_to_scalar
//
// Given raw policy logits (4672) and value logits (3) plus the current
// position, mask illegal moves, softmax over the legal set, and return the
// top moves sorted by probability, together with the WDL distribution.
// =====================================================================
import type { Chess } from "chess.js";
import type { MoveSuggestion, NetResult } from "./types";
import { legalMoveEntries } from "./moveEncoding";

/** Softmax of WDL value logits [win, draw, loss]. */
export function softmax3(
  logits: ArrayLike<number>
): [number, number, number] {
  const m = Math.max(logits[0], logits[1], logits[2]);
  const e0 = Math.exp(logits[0] - m);
  const e1 = Math.exp(logits[1] - m);
  const e2 = Math.exp(logits[2] - m);
  const sum = e0 + e1 + e2;
  return [e0 / sum, e1 / sum, e2 / sum];
}

/**
 * Build a NetResult from raw network outputs.
 *
 * @param policy logits, length 4672
 * @param value  WDL logits, length 3 ([win, draw, loss] from side-to-move POV)
 * @param chess  current position (used for legal masking + SAN)
 * @param topK   number of suggestions to return (default 10)
 */
export function buildNetResult(
  policy: ArrayLike<number>,
  value: ArrayLike<number>,
  chess: Chess,
  topK = 10
): NetResult {
  const wdl = softmax3(value);
  const scalar = wdl[0] - wdl[2];

  const legal = legalMoveEntries(chess);
  if (legal.length === 0) {
    return { topMoves: [], wdl, scalar };
  }

  // Masked softmax over legal move indices only (mirrors policy_to_moves:
  // subtract the max legal logit, exp, normalize by the legal-set sum).
  let maxLogit = -Infinity;
  for (const e of legal) {
    const v = policy[e.index];
    if (v > maxLogit) maxLogit = v;
  }

  let expSum = 0;
  const exps = new Array<number>(legal.length);
  for (let i = 0; i < legal.length; i++) {
    const v = Math.exp(policy[legal[i].index] - maxLogit);
    exps[i] = v;
    expSum += v;
  }

  const suggestions: MoveSuggestion[] = legal.map((e, i) => ({
    uci: e.uci,
    san: e.san,
    prob: exps[i] / expSum,
  }));

  suggestions.sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));

  return {
    topMoves: suggestions.slice(0, topK),
    wdl,
    scalar,
  };
}
