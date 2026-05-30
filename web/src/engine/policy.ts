// Port of models/v3_vast/chess_nn :: policy_to_moves + wdl_to_scalar.
import type { Chess } from "chess.js";
import type { MoveSuggestion, NetResult } from "./types";
import { legalMoveEntries } from "./moveEncoding";

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

  // Masked softmax over legal move indices only (mirrors policy_to_moves).
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
