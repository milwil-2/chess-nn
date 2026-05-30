// Port of models/v3_vast/chess_nn/move_encoding.py — AlphaZero 73*64 policy.
import { Chess } from "chess.js";
import type { Color, PieceSymbol } from "chess.js";

export const POLICY_SIZE = 4672;

// Order MUST match the Python DIRECTIONS list.
const DIRECTIONS: ReadonlyArray<[number, number]> = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [1, -1],
];

const KNIGHT_DELTAS: ReadonlyArray<[number, number]> = [
  [2, 1],
  [2, -1],
  [-2, 1],
  [-2, -1],
  [1, 2],
  [1, -2],
  [-1, 2],
  [-1, -2],
];

const UNDERPROMO_PIECES: ReadonlyArray<PieceSymbol> = ["n", "b", "r"];
// From the current player's POV (pawn always moves "up" => dr = +1).
const UNDERPROMO_DIRS: ReadonlyArray<[number, number]> = [
  [1, -1],
  [1, 0],
  [1, 1],
];

function indexOfDelta(
  list: ReadonlyArray<[number, number]>,
  dr: number,
  dc: number
): number {
  for (let i = 0; i < list.length; i++) {
    if (list[i][0] === dr && list[i][1] === dc) return i;
  }
  return -1;
}

function squareToRankFile(square: string): { rank: number; file: number } {
  return {
    file: square.charCodeAt(0) - 97,
    rank: square.charCodeAt(1) - 49,
  };
}

export function moveToIndex(
  from: string,
  to: string,
  piece: PieceSymbol,
  promotion: PieceSymbol | undefined,
  turn: Color
): number {
  const flip = turn === "b";

  const f = squareToRankFile(from);
  const t = squareToRankFile(to);

  let fromRank = f.rank;
  const fromFile = f.file;
  let toRank = t.rank;
  const toFile = t.file;

  if (flip) {
    fromRank = 7 - fromRank;
    toRank = 7 - toRank;
  }

  const dr = toRank - fromRank;
  const dc = toFile - fromFile;

  const sourceSquareIdx = fromRank * 8 + fromFile;

  if (promotion !== undefined && promotion !== "q") {
    const pieceIdx = UNDERPROMO_PIECES.indexOf(promotion);
    const dirIdx = indexOfDelta(UNDERPROMO_DIRS, dr, dc);
    const plane = 64 + pieceIdx * 3 + dirIdx;
    return sourceSquareIdx * 73 + plane;
  }

  if (piece === "n") {
    const knightIdx = indexOfDelta(KNIGHT_DELTAS, dr, dc);
    const plane = 56 + knightIdx;
    return sourceSquareIdx * 73 + plane;
  }

  const distance = Math.max(Math.abs(dr), Math.abs(dc));
  const unitDr = dr !== 0 ? Math.trunc(dr / distance) : 0;
  const unitDc = dc !== 0 ? Math.trunc(dc / distance) : 0;
  const dirIdx = indexOfDelta(DIRECTIONS, unitDr, unitDc);
  const plane = dirIdx * 7 + (distance - 1);
  return sourceSquareIdx * 73 + plane;
}

function rankFileToSquare(file: number, rank: number): string {
  return String.fromCharCode(97 + file) + String.fromCharCode(49 + rank);
}

/** Inverse of moveToIndex. Returns null if the index does not land on the board. */
export function indexToMove(index: number, turn: Color): string | null {
  const flip = turn === "b";

  const sourceSquareIdx = Math.floor(index / 73);
  const plane = index % 73;

  const fromRank = Math.floor(sourceSquareIdx / 8);
  const fromFile = sourceSquareIdx % 8;

  let promotion: PieceSymbol | undefined;
  let toRank: number;
  let toFile: number;

  if (plane >= 64) {
    const off = plane - 64;
    const pieceIdx = Math.floor(off / 3);
    const dirIdx = off % 3;
    promotion = UNDERPROMO_PIECES[pieceIdx];
    const [dr, dc] = UNDERPROMO_DIRS[dirIdx];
    toRank = fromRank + dr;
    toFile = fromFile + dc;
  } else if (plane >= 56) {
    const [dr, dc] = KNIGHT_DELTAS[plane - 56];
    toRank = fromRank + dr;
    toFile = fromFile + dc;
  } else {
    const dirIdx = Math.floor(plane / 7);
    const distance = (plane % 7) + 1;
    const [unitDr, unitDc] = DIRECTIONS[dirIdx];
    toRank = fromRank + unitDr * distance;
    toFile = fromFile + unitDc * distance;
  }

  if (toRank < 0 || toRank > 7 || toFile < 0 || toFile > 7) return null;

  const actualFromRank = flip ? 7 - fromRank : fromRank;
  const actualToRank = flip ? 7 - toRank : toRank;

  const fromSq = rankFileToSquare(fromFile, actualFromRank);
  const toSq = rankFileToSquare(toFile, actualToRank);

  // We can't know piece type without the board, so callers should match
  // against legal moves; this helper is primarily exercised via legalMoveEntries.
  let suffix = "";
  if (promotion !== undefined) suffix = promotion;
  return fromSq + toSq + suffix;
}

export interface LegalMoveEntry {
  index: number;
  uci: string;
  san: string;
}

export function legalMoveEntries(chess: Chess): LegalMoveEntry[] {
  const turn = chess.turn();
  const verbose = chess.moves({ verbose: true });
  return verbose.map((m) => {
    const idx = moveToIndex(m.from, m.to, m.piece, m.promotion, turn);
    const uci = m.from + m.to + (m.promotion ?? "");
    return { index: idx, uci, san: m.san };
  });
}

export function legalMoveIndices(chess: Chess): number[] {
  return legalMoveEntries(chess).map((e) => e.index);
}
