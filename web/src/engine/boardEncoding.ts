// Port of models/v3_vast/chess_nn/board_encoding.py :: boards_to_tensor.
// Layout: planes[plane * 64 + row * 8 + col], rank 0 = rank 1, file 0 = a.
import { Chess } from "chess.js";
import type { Color, PieceSymbol } from "chess.js";

export const HISTORY_LENGTH = 8;
export const INPUT_PLANES = 105;
const PLANE_AREA = 64;
export const INPUT_SIZE = INPUT_PLANES * PLANE_AREA;

const PIECE_BASE: Record<PieceSymbol, number> = {
  p: 0,
  n: 1,
  b: 2,
  r: 3,
  q: 4,
  k: 5,
};

interface Frame {
  pieces: { rank: number; file: number; color: Color; type: PieceSymbol }[];
  turn: Color;
  whiteKingside: boolean;
  whiteQueenside: boolean;
  blackKingside: boolean;
  blackQueenside: boolean;
  ep: { rank: number; file: number } | null;
  halfmove: number;
  repKey: string;
}

/**
 * `epOverride` lets the caller force the en-passant target square, which is
 * required to match python-chess: it records `board.ep_square` after ANY
 * double pawn push, whereas chess.js (and the stored fixture FENs) only emit
 * the ep field in the FEN when an en-passant capture is actually legal.
 */
export function parseFen(
  fen: string,
  epOverride?: { rank: number; file: number } | null
): Frame {
  const parts = fen.trim().split(/\s+/);
  const placement = parts[0];
  const turn = (parts[1] as Color) ?? "w";
  const castling = parts[2] ?? "-";
  const epField = parts[3] ?? "-";
  const halfmove = parts[4] !== undefined ? parseInt(parts[4], 10) : 0;

  const pieces: Frame["pieces"] = [];
  const rows = placement.split("/");
  for (let i = 0; i < rows.length; i++) {
    const rank = 7 - i;
    let file = 0;
    for (const ch of rows[i]) {
      if (ch >= "1" && ch <= "8") {
        file += ch.charCodeAt(0) - 48;
      } else {
        const lower = ch.toLowerCase() as PieceSymbol;
        const color: Color = ch === lower ? "b" : "w";
        pieces.push({ rank, file, color, type: lower });
        file += 1;
      }
    }
  }

  let ep: { rank: number; file: number } | null = null;
  if (epOverride !== undefined) {
    ep = epOverride;
  } else if (epField !== "-") {
    const file = epField.charCodeAt(0) - 97;
    const rank = epField.charCodeAt(1) - 49;
    ep = { rank, file };
  }

  // python-chess repetition compares board_fen() + turn + castling_rights + ep_square.
  const epKey = ep ? `${ep.file},${ep.rank}` : "-";

  return {
    pieces,
    turn,
    whiteKingside: castling.includes("K"),
    whiteQueenside: castling.includes("Q"),
    blackKingside: castling.includes("k"),
    blackQueenside: castling.includes("q"),
    ep,
    halfmove: Number.isFinite(halfmove) ? halfmove : 0,
    repKey: `${placement}|${turn}|${castling}|${epKey}`,
  };
}

function epFromMove(
  fromFile: number,
  fromRank: number,
  toFile: number,
  toRank: number,
  pieceType: string
): { rank: number; file: number } | null {
  if (pieceType !== "p") return null;
  if (fromFile !== toFile) return null;
  if (Math.abs(toRank - fromRank) !== 2) return null;
  return { file: fromFile, rank: (fromRank + toRank) / 2 };
}

interface BoardState {
  fen: string;
  /** ep square per python-chess semantics (undefined => trust FEN field). */
  ep: { rank: number; file: number } | null | undefined;
}

/**
 * states = [startFen, after move1, ..., current]; the returned `boards` is
 * reverse(states).slice(0, 8) so boards[0] = current. Each replayed state
 * carries an explicit ep square because python-chess sets ep_square on every
 * double pawn push, unlike chess.js / the stored FENs.
 */
export function buildBoards(startFen: string, movesUci: string[]): BoardState[] {
  const game = new Chess(startFen);
  const states: BoardState[] = [{ fen: game.fen(), ep: undefined }];
  for (const uci of movesUci) {
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promotion = uci.length > 4 ? uci[4] : undefined;
    const mv = game.move({ from, to, promotion });
    const ff = from.charCodeAt(0) - 97;
    const fr = from.charCodeAt(1) - 49;
    const tf = to.charCodeAt(0) - 97;
    const tr = to.charCodeAt(1) - 49;
    const ep = epFromMove(ff, fr, tf, tr, mv.piece);
    states.push({ fen: game.fen(), ep });
  }
  states.reverse();
  return states.slice(0, HISTORY_LENGTH);
}

export function encodeGame(startFen: string, movesUci: string[]): Float32Array {
  const states = buildBoards(startFen, movesUci);
  const frames = states.map((s) => parseFen(s.fen, s.ep));
  return encodeFrames(frames);
}

export function encodeFrames(frames: Frame[]): Float32Array {
  const planes = new Float32Array(INPUT_SIZE);
  const current = frames[0];
  const flip = current.turn === "b";

  const set = (plane: number, row: number, col: number, value: number) => {
    planes[plane * PLANE_AREA + row * 8 + col] = value;
  };
  const fill = (plane: number, value: number) => {
    const base = plane * PLANE_AREA;
    for (let i = 0; i < PLANE_AREA; i++) planes[base + i] = value;
  };

  const nFrames = Math.min(frames.length, HISTORY_LENGTH);
  for (let frameIdx = 0; frameIdx < nFrames; frameIdx++) {
    const base = frameIdx * 12;
    for (const piece of frames[frameIdx].pieces) {
      let row = piece.rank;
      const col = piece.file;
      if (flip) row = 7 - row;

      let planeIdx = PIECE_BASE[piece.type] + (piece.color === "w" ? 0 : 6);
      if (flip) planeIdx = planeIdx < 6 ? planeIdx + 6 : planeIdx - 6;

      set(base + planeIdx, row, col, 1.0);
    }
  }

  const meta = HISTORY_LENGTH * 12;

  if (!flip) {
    fill(meta + 0, current.whiteKingside ? 1 : 0);
    fill(meta + 1, current.whiteQueenside ? 1 : 0);
    fill(meta + 2, current.blackKingside ? 1 : 0);
    fill(meta + 3, current.blackQueenside ? 1 : 0);
  } else {
    fill(meta + 0, current.blackKingside ? 1 : 0);
    fill(meta + 1, current.blackQueenside ? 1 : 0);
    fill(meta + 2, current.whiteKingside ? 1 : 0);
    fill(meta + 3, current.whiteQueenside ? 1 : 0);
  }

  if (current.ep) {
    let epRow = current.ep.rank;
    const epCol = current.ep.file;
    if (flip) epRow = 7 - epRow;
    set(meta + 4, epRow, epCol, 1.0);
  }

  if (current.turn === "w") fill(meta + 5, 1.0);

  fill(meta + 6, Math.min(current.halfmove / 100.0, 1.0));

  let repCount = 0;
  for (let i = 1; i < frames.length; i++) {
    if (frames[i].repKey === current.repKey) repCount++;
  }
  if (repCount >= 1) fill(meta + 7, 1.0);
  if (repCount >= 2) fill(meta + 8, 1.0);

  return planes;
}

export function planeSums(tensor: Float32Array): number[] {
  const sums = new Array<number>(INPUT_PLANES).fill(0);
  for (let plane = 0; plane < INPUT_PLANES; plane++) {
    let s = 0;
    const base = plane * PLANE_AREA;
    for (let i = 0; i < PLANE_AREA; i++) s += tensor[base + i];
    sums[plane] = s;
  }
  return sums;
}
