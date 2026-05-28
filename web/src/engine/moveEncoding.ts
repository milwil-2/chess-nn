// =====================================================================
// Move encoding — TypeScript port of
//   models/v3_vast/chess_nn/move_encoding.py :: move_to_index / index_to_move
//
// AlphaZero encoding: 73 move planes * 64 source squares = 4672.
//   planes 0-55:  queen-style moves (8 dirs * 7 distances; also pawns)
//   planes 56-63: knight moves (8 L-shapes)
//   planes 64-72: underpromotions (3 dirs * 3 pieces; queen-promo reuses 0-55)
// Coordinates are taken from the CURRENT player's perspective (board flipped
// vertically when black is to move).
// =====================================================================
import { Chess } from "chess.js";
import type { Color, PieceSymbol } from "chess.js";

export const POLICY_SIZE = 4672;

// (dr, dc) per single step. Order MUST match the Python DIRECTIONS list.
const DIRECTIONS: ReadonlyArray<[number, number]> = [
  [1, 0], // N
  [1, 1], // NE
  [0, 1], // E
  [-1, 1], // SE
  [-1, 0], // S
  [-1, -1], // SW
  [0, -1], // W
  [1, -1], // NW
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

// Underpromotion target pieces, in order: knight, bishop, rook.
const UNDERPROMO_PIECES: ReadonlyArray<PieceSymbol> = ["n", "b", "r"];
// Underpromo directions (dr, dc) from current player's POV: capture-left,
// straight, capture-right (pawn always moves "up" => dr = +1).
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
    file: square.charCodeAt(0) - 97, // 'a' -> 0
    rank: square.charCodeAt(1) - 49, // '1' -> 0
  };
}

/**
 * Convert a move to a policy index (0-4671).
 *
 * @param from       source square, e.g. "e2"
 * @param to         destination square, e.g. "e4"
 * @param piece      moving piece type (chess.js PieceSymbol)
 * @param promotion  promotion piece type or undefined
 * @param turn       side to move ('w' | 'b') — controls the perspective flip
 */
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

  const sourceSquareIdx = fromRank * 8 + fromFile; // 0-63

  // --- Underpromotion? (anything but queen) ---
  if (promotion !== undefined && promotion !== "q") {
    const pieceIdx = UNDERPROMO_PIECES.indexOf(promotion);
    const dirIdx = indexOfDelta(UNDERPROMO_DIRS, dr, dc);
    const plane = 64 + pieceIdx * 3 + dirIdx;
    return sourceSquareIdx * 73 + plane;
  }

  // --- Knight move? ---
  if (piece === "n") {
    const knightIdx = indexOfDelta(KNIGHT_DELTAS, dr, dc);
    const plane = 56 + knightIdx;
    return sourceSquareIdx * 73 + plane;
  }

  // --- Queen-style move (includes pawns + queen promotions) ---
  const distance = Math.max(Math.abs(dr), Math.abs(dc));
  const unitDr = dr !== 0 ? Math.trunc(dr / distance) : 0;
  const unitDc = dc !== 0 ? Math.trunc(dc / distance) : 0;
  const dirIdx = indexOfDelta(DIRECTIONS, unitDr, unitDc);
  const plane = dirIdx * 7 + (distance - 1); // 0-55
  return sourceSquareIdx * 73 + plane;
}

function rankFileToSquare(file: number, rank: number): string {
  return String.fromCharCode(97 + file) + String.fromCharCode(49 + rank);
}

/**
 * Inverse of moveToIndex. Returns a UCI string, or null if the index does not
 * land on the board. (Promotion suffix appended for pawns reaching the back
 * rank; queen by default, underpromotion piece otherwise.)
 */
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

  // Queen promotion default: pawn reaching the (perspective-flipped) back rank
  // lands on rank 8 for white / rank 1 for black. We can't know the piece type
  // here without the board, so callers should match against legal moves; this
  // helper is primarily exercised via the legal-index map in policy.ts.
  let suffix = "";
  if (promotion !== undefined) suffix = promotion;
  return fromSq + toSq + suffix;
}

/**
 * Map every legal move in the position to its policy index.
 * Returns parallel info needed for building suggestions.
 */
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

/** Policy indices of all legal moves (mirrors get_legal_move_indices). */
export function legalMoveIndices(chess: Chess): number[] {
  return legalMoveEntries(chess).map((e) => e.index);
}
