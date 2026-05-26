"""
Stockfish annotation pass — enrich existing .npz chunks with Stockfish best-move labels.

For each chunk_NNNN.npz it writes a companion chunk_NNNN_sf.npz containing:
  sf_policy : int64 array, shape (N,)
               move index for Stockfish's best move, or -1 if not annotated

Only SF_ANNOTATE_FRACTION of positions per chunk are annotated (random sample) to
keep wall time manageable.  The training loop adds an auxiliary cross-entropy loss
term for annotated positions (weight SF_LOSS_WEIGHT in config).

Usage:
  python3 data/stockfish_annotate.py                  # annotate all chunks
  python3 data/stockfish_annotate.py --chunk 42       # annotate one chunk
  python3 data/stockfish_annotate.py --stockfish /path/to/stockfish
"""

import argparse
import glob
import io
import os
import sys
import random

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, SF_ANNOTATE_FRACTION, POLICY_OUTPUT_SIZE
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.move_encoding import move_to_index


def _find_stockfish() -> str:
    """Try common install paths; raise if not found."""
    candidates = [
        "stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    import shutil
    for c in candidates:
        if shutil.which(c):
            return c
    raise FileNotFoundError(
        "stockfish not found. Install it:\n"
        "  apt-get install stockfish          # Debian/Ubuntu\n"
        "  brew install stockfish             # macOS\n"
        "or pass --stockfish /path/to/binary"
    )


def annotate_chunk(chunk_path: str, engine: chess.engine.SimpleEngine,
                   depth: int = 12, fraction: float = SF_ANNOTATE_FRACTION) -> str:
    """
    Load one chunk, annotate a random fraction of positions with Stockfish,
    write companion _sf.npz. Returns the output path.
    """
    out_path = chunk_path.replace(".npz", "_sf.npz")
    if os.path.exists(out_path):
        return out_path  # already done

    with np.load(chunk_path) as f:
        boards_arr = f["boards"]       # (N, 105, 8, 8) float32
        N = len(boards_arr)

    sf_policy = np.full(N, -1, dtype=np.int64)

    # Pick the positions to annotate
    n_annotate = max(1, int(N * fraction))
    indices = sorted(random.sample(range(N), n_annotate))

    for i in tqdm(indices, desc=os.path.basename(chunk_path), unit="pos", leave=False):
        # Reconstruct the board from the tensor's first plane slice.
        # The board tensor encodes piece positions; the simplest reconstruction is
        # to replay the game — but we don't have move history in the chunk.
        # Instead we use FEN reconstruction from the piece planes directly.
        board = _tensor_to_board(boards_arr[i])
        if board is None:
            continue
        try:
            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = result.get("pv", [None])[0]
            if best_move and best_move in board.legal_moves:
                sf_policy[i] = move_to_index(best_move, board)
        except Exception:
            pass

    np.savez_compressed(out_path, sf_policy=sf_policy)
    n_annotated = int((sf_policy != -1).sum())
    tqdm.write(f"  {os.path.basename(chunk_path)} → {n_annotated}/{N} positions annotated")
    return out_path


def _tensor_to_board(tensor: np.ndarray) -> chess.Board | None:
    """
    Reconstruct a chess.Board from the first frame (most-recent position) of a
    boards_to_tensor output.  Planes 0-5 = white pieces (P,N,B,R,Q,K),
    planes 6-11 = black pieces.  Plane 96 = side to move (1 = white).

    This is a best-effort reconstruction; castling rights and en-passant are lost.
    """
    try:
        board = chess.Board(fen=None)  # empty board, no castling, no ep
        board.clear()

        PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING]

        for pt_idx, pt in enumerate(PIECE_TYPES):
            plane_w = tensor[pt_idx]        # white pieces
            plane_b = tensor[pt_idx + 6]    # black pieces
            for rank in range(8):
                for file in range(8):
                    sq = chess.square(file, rank)
                    if plane_w[rank, file] > 0.5:
                        board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
                    elif plane_b[rank, file] > 0.5:
                        board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))

        # Side to move is stored in plane 96 (the 9th meta plane, index = 8*12 + 0 = 96)
        side_plane = tensor[96] if tensor.shape[0] > 96 else None
        if side_plane is not None and side_plane.mean() < 0.5:
            board.turn = chess.BLACK
        else:
            board.turn = chess.WHITE

        # Validate: must have both kings
        if (board.king(chess.WHITE) is None or board.king(chess.BLACK) is None):
            return None

        return board
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Annotate chunk .npz files with Stockfish best moves.")
    parser.add_argument("--stockfish", default=None, help="Path to stockfish binary")
    parser.add_argument("--chunk", type=int, default=None, help="Annotate only this chunk index")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth (default 12)")
    parser.add_argument("--fraction", type=float, default=SF_ANNOTATE_FRACTION,
                        help=f"Fraction of positions to annotate (default {SF_ANNOTATE_FRACTION})")
    args = parser.parse_args()

    sf_path = args.stockfish or _find_stockfish()
    print(f"Using Stockfish: {sf_path}")

    if args.chunk is not None:
        chunks = [os.path.join(PROCESSED_DATA_DIR, f"chunk_{args.chunk:04d}.npz")]
    else:
        chunks = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz")))
        # Skip companion files
        chunks = [c for c in chunks if not c.endswith("_sf.npz")]

    if not chunks:
        print(f"No chunks found in {PROCESSED_DATA_DIR}")
        sys.exit(1)

    print(f"Annotating {len(chunks)} chunk(s) at depth {args.depth}, fraction {args.fraction:.0%}")

    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        for chunk_path in tqdm(chunks, desc="chunks", unit="chunk"):
            annotate_chunk(chunk_path, engine, depth=args.depth, fraction=args.fraction)

    print("Done.")


if __name__ == "__main__":
    main()
