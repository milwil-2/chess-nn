"""
Board Encoding — converting a chess position into a tensor the neural network can read.

A chess board has 64 squares and 12 piece types (6 per color). We represent the board
as a stack of 18 binary "planes", each an 8x8 grid. Think of it like 18 transparent
sheets of paper, each one highlighting where a specific piece type sits.

Why tensors? Neural networks only understand numbers — this encoding translates
the game state into a 3D grid of 0s and 1s that a CNN can process.
"""

import numpy as np
import chess

# Number of historical board states to encode (current + 7 prior).
# Each frame contributes 12 piece planes; 9 meta planes cover current-position
# state (castling, en passant, side to move, 50-move clock, repetition x2).
# Total: 8×12 + 9 = 105 planes.
HISTORY_LENGTH = 8

# Map each piece type to its plane index (0-5 for white, 6-11 for black)
# chess.PAWN=1, chess.KNIGHT=2, ..., chess.KING=6
PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess position to an (18, 8, 8) numpy array.

    The 18 planes are:
      0-5:  White pieces  (pawn, knight, bishop, rook, queen, king)
      6-11: Black pieces  (same order)
      12:   White can castle kingside  (all 1s or all 0s)
      13:   White can castle queenside
      14:   Black can castle kingside
      15:   Black can castle queenside
      16:   En passant target square  (1 on that square, 0 elsewhere)
      17:   Side to move  (all 1s = white to move, all 0s = black to move)

    Crucially, we always encode from the CURRENT PLAYER's perspective:
    if it's Black's turn, we flip the board so Black is always "at the bottom."
    This means the network only needs to learn one perspective, halving complexity.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK  # Flip board if it's Black's turn

    # --- Planes 0-11: Piece positions ---
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)  # 0 = rank 1 (bottom), 7 = rank 8 (top)
        col = chess.square_file(square)  # 0 = a-file, 7 = h-file

        if flip:
            row = 7 - row  # Flip vertically so current player is always "at bottom"

        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]

        # When flipped, also swap which planes represent "my pieces" vs "opponent's pieces"
        if flip:
            # Black's pieces (planes 6-11) become "my pieces" (planes 0-5) and vice versa
            if plane_idx < 6:
                plane_idx += 6
            else:
                plane_idx -= 6

        planes[plane_idx, row, col] = 1.0

    # --- Planes 12-15: Castling rights ---
    # These are binary flags — entire plane is 1 if right exists, 0 if not
    if not flip:
        planes[12] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[13] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[14] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[15] = float(board.has_queenside_castling_rights(chess.BLACK))
    else:
        # From Black's perspective, Black's castling rights are "my" rights (planes 12-13)
        planes[12] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[13] = float(board.has_queenside_castling_rights(chess.BLACK))
        planes[14] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[15] = float(board.has_queenside_castling_rights(chess.WHITE))

    # --- Plane 16: En passant target square ---
    if board.ep_square is not None:
        ep_row = chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        if flip:
            ep_row = 7 - ep_row
        planes[16, ep_row, ep_col] = 1.0

    # --- Plane 17: Side to move ---
    # All 1s if white to move (from white's perspective), all 0s if black
    # Since we always flip to current player's view, this is always white-to-move = 1
    if board.turn == chess.WHITE:
        planes[17] = 1.0

    return planes


def boards_to_tensor(boards: list) -> np.ndarray:
    """
    Encode up to HISTORY_LENGTH board states into a (105, 8, 8) tensor.

    boards[0] must be the CURRENT position; boards[1] is one move ago, etc.
    Older frames are zero-padded when fewer than HISTORY_LENGTH boards are given.

    All frames are encoded from the CURRENT player's perspective (boards[0].turn)
    so the network never has to learn "I am White" vs "I am Black" separately.

    Plane layout:
      0–11:   current board (12 piece planes)
      12–23:  1 move ago
      ...
      84–95:  7 moves ago
      96–99:  castling rights (current board only)
      100:    en passant target square (current board only)
      101:    side to move (current board only)
      102:    50-move clock, normalized to [0, 1]
      103:    repetition flag — 1.0 if current position appeared once in history
      104:    repetition flag — 1.0 if current position appeared twice in history
    """
    total_planes = HISTORY_LENGTH * 12 + 9
    planes = np.zeros((total_planes, 8, 8), dtype=np.float32)

    current = boards[0]
    flip = current.turn == chess.BLACK

    # --- Planes 0-95: piece positions across up to 8 history frames ---
    for frame_idx, board in enumerate(boards[:HISTORY_LENGTH]):
        base = frame_idx * 12
        for square, piece in board.piece_map().items():
            row = chess.square_rank(square)
            col = chess.square_file(square)
            if flip:
                row = 7 - row

            plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            if flip:
                plane_idx = plane_idx + 6 if plane_idx < 6 else plane_idx - 6

            planes[base + plane_idx, row, col] = 1.0

    # --- Planes 96-101: meta planes from current board only ---
    meta = HISTORY_LENGTH * 12  # = 96
    if not flip:
        planes[meta + 0] = float(current.has_kingside_castling_rights(chess.WHITE))
        planes[meta + 1] = float(current.has_queenside_castling_rights(chess.WHITE))
        planes[meta + 2] = float(current.has_kingside_castling_rights(chess.BLACK))
        planes[meta + 3] = float(current.has_queenside_castling_rights(chess.BLACK))
    else:
        planes[meta + 0] = float(current.has_kingside_castling_rights(chess.BLACK))
        planes[meta + 1] = float(current.has_queenside_castling_rights(chess.BLACK))
        planes[meta + 2] = float(current.has_kingside_castling_rights(chess.WHITE))
        planes[meta + 3] = float(current.has_queenside_castling_rights(chess.WHITE))

    if current.ep_square is not None:
        ep_row = chess.square_rank(current.ep_square)
        ep_col = chess.square_file(current.ep_square)
        if flip:
            ep_row = 7 - ep_row
        planes[meta + 4, ep_row, ep_col] = 1.0

    if current.turn == chess.WHITE:
        planes[meta + 5] = 1.0

    # --- Planes 102-104: 50-move clock + repetition ---
    # Plane 102: half-move clock normalized. 100 half-moves = 50 full moves = draw claim.
    planes[meta + 6] = min(current.halfmove_clock / 100.0, 1.0)

    # Planes 103-104: count how many times the current position appeared in history.
    # Compare board_fen (piece layout), turn, castling rights, and ep_square — these
    # are exactly the four factors that define a position for threefold repetition.
    rep_count = sum(
        1 for prior in boards[1:]
        if (prior.board_fen() == current.board_fen()
            and prior.turn == current.turn
            and prior.castling_rights == current.castling_rights
            and prior.ep_square == current.ep_square)
    )
    if rep_count >= 1:
        planes[meta + 7] = 1.0
    if rep_count >= 2:
        planes[meta + 8] = 1.0

    return planes
