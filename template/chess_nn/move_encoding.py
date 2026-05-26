"""
Move Encoding — converting chess moves into numbers the network can predict.

The policy head outputs a probability for each of 4672 possible "move slots."
We need a consistent mapping: every legal move maps to exactly one index,
and every index maps back to at most one legal move.

We use AlphaZero's encoding: 73 move planes × 64 source squares = 4672.
The 73 planes cover every possible move geometry a piece can make.
"""

import chess
import numpy as np

# --- Move plane layout (73 planes per source square) ---
#
# Planes 0-55: "Queen-style" moves (also covers pawn pushes/captures)
#   8 directions × 7 distances = 56 planes
#   Directions: N, NE, E, SE, S, SW, W, NW (from current player's perspective)
#   Distance: 1-7 squares
#
# Planes 56-63: Knight moves (8 possible L-shapes)
#
# Planes 64-72: Underpromotions (promote to N, B, or R — not Q)
#   3 directions (capture-left, forward, capture-right) × 3 pieces = 9 planes
#   Queen promotions reuse the normal queen-move planes (distance=1, forward/diagonal)

DIRECTIONS = [
    (1, 0),   # N  — rank+1
    (1, 1),   # NE — rank+1, file+1
    (0, 1),   # E  — file+1
    (-1, 1),  # SE — rank-1, file+1
    (-1, 0),  # S  — rank-1
    (-1, -1), # SW — rank-1, file-1
    (0, -1),  # W  — file-1
    (1, -1),  # NW — rank+1, file-1
]

KNIGHT_DELTAS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
# Underpromo directions in (dr, dc) from current player's perspective (board always flipped to player's POV)
# Pawn always moves "up" (dr=+1). Three options: capture-left, straight, capture-right
UNDERPROMO_DIRS = [(1, -1), (1, 0), (1, 1)]  # capture-left, straight, capture-right


def _queen_plane(dr: int, dc: int) -> int:
    """Get plane index for a queen-style move with delta (dr, dc) per step."""
    dir_idx = DIRECTIONS.index((dr, dc))
    return dir_idx  # Plane 0-7 for distance 1; we multiply by distance offset below


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Convert a chess move to a policy index (0-4671).

    Args:
        move: The move to encode
        board: Current board position (needed for perspective flip and context)

    Returns:
        Integer index in [0, 4671]
    """
    flip = board.turn == chess.BLACK

    from_sq = move.from_square
    to_sq = move.to_square

    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    # Flip coordinates for Black's perspective
    if flip:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    dr = to_rank - from_rank
    dc = to_file - from_file

    source_square_idx = from_rank * 8 + from_file  # 0-63

    # --- Underpromotion? ---
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        # Find which underpromo direction
        dir_idx = UNDERPROMO_DIRS.index((dr, dc))
        plane = 64 + piece_idx * 3 + dir_idx
        return source_square_idx * 73 + plane

    # --- Knight move? ---
    if board.piece_at(from_sq) is not None and board.piece_at(from_sq).piece_type == chess.KNIGHT:
        knight_idx = KNIGHT_DELTAS.index((dr, dc))
        plane = 56 + knight_idx
        return source_square_idx * 73 + plane

    # --- Queen-style move (includes pawns) ---
    distance = max(abs(dr), abs(dc))
    unit_dr = dr // distance if dr != 0 else 0
    unit_dc = dc // distance if dc != 0 else 0
    dir_idx = DIRECTIONS.index((unit_dr, unit_dc))
    plane = dir_idx * 7 + (distance - 1)  # 0-55
    return source_square_idx * 73 + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert a policy index back to a chess.Move.

    This is the reverse of move_to_index. Used when the network outputs
    a policy distribution and we need to pick the actual move to play.

    Returns chess.Move.null() if the index doesn't map to a legal move.
    """
    flip = board.turn == chess.BLACK

    source_square_idx = index // 73
    plane = index % 73

    from_rank = source_square_idx // 8
    from_file = source_square_idx % 8

    # Undo the perspective flip
    actual_from_rank = (7 - from_rank) if flip else from_rank
    from_sq = chess.square(from_file, actual_from_rank)

    promotion = None

    if plane >= 64:
        # Underpromotion
        plane_offset = plane - 64
        piece_idx = plane_offset // 3
        dir_idx = plane_offset % 3
        promotion = UNDERPROMO_PIECES[piece_idx]
        dr, dc = UNDERPROMO_DIRS[dir_idx]
        to_rank = from_rank + dr
        to_file = from_file + dc
    elif plane >= 56:
        # Knight move
        knight_idx = plane - 56
        dr, dc = KNIGHT_DELTAS[knight_idx]
        to_rank = from_rank + dr
        to_file = from_file + dc
    else:
        # Queen-style move
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        unit_dr, unit_dc = DIRECTIONS[dir_idx]
        to_rank = from_rank + unit_dr * distance
        to_file = from_file + unit_dc * distance

    if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
        return chess.Move.null()

    actual_to_rank = (7 - to_rank) if flip else to_rank
    to_sq = chess.square(to_file, actual_to_rank)

    # Queen promotion: if pawn reaches back rank and no underpromotion specified
    piece = board.piece_at(from_sq)
    if piece is not None and piece.piece_type == chess.PAWN and promotion is None:
        back_rank = 7 if board.turn == chess.WHITE else 0
        if chess.square_rank(to_sq) == back_rank:
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_indices(board: chess.Board) -> list[int]:
    """Return the policy indices of all legal moves in the current position."""
    return [move_to_index(move, board) for move in board.legal_moves]


def policy_to_moves(policy: np.ndarray, board: chess.Board, top_k: int = 10) -> list[tuple[chess.Move, float]]:
    """
    Given a raw policy array (4672 values), return the top-k legal moves
    with their probabilities (after masking illegal moves and softmax).

    Used by the visualization app to draw move arrows.
    """
    import torch
    legal_indices = get_legal_move_indices(board)
    if not legal_indices:
        return []

    # Mask: set illegal moves to -inf so they get ~0 probability
    masked = np.full(4672, -1e9, dtype=np.float32)
    for idx in legal_indices:
        masked[idx] = policy[idx]

    # Softmax
    exp = np.exp(masked - masked[legal_indices].max())
    exp_sum = exp[legal_indices].sum()
    probs = np.zeros(4672, dtype=np.float32)
    for idx in legal_indices:
        probs[idx] = exp[idx] / exp_sum

    # Sort by probability
    sorted_indices = sorted(legal_indices, key=lambda i: probs[i], reverse=True)
    results = []
    for idx in sorted_indices[:top_k]:
        move = index_to_move(idx, board)
        results.append((move, float(probs[idx])))
    return results
