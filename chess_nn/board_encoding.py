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
