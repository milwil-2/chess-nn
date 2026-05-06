"""
Rule-based chess tactics and opening detector.

Detects: forks, absolute pins, hanging pieces, skewers, and common openings.
All detection is pure python-chess logic — no neural network involved.
Each result carries the squares to highlight and a human-readable description.
"""

import chess
from dataclasses import dataclass

PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100,
}
PIECE_NAME = {
    chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
    chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King",
}

# RGB colours for each tactic type (used for board highlights + sidebar dots)
TACTIC_COLORS = {
    "Fork":          (255, 140,  0),   # orange
    "Pin":           (180,  50, 255),  # purple
    "Skewer":        (255,  50, 180),  # pink
    "Hanging Piece": (255,  60,  60),  # red
    "Opening":       (80,  180, 255),  # blue
}


@dataclass
class Tactic:
    name: str          # short label shown in sidebar
    squares: list      # chess.Square list to highlight on the board
    description: str   # one-line explanation
    color: tuple       # RGB


# ── Opening book ────────────────────────────────────────────────────────────
# Ordered longest-first so the most specific match wins.
_OPENINGS = [
    ("Nimzo-Indian Defence",    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4"]),
    ("King's Indian Defence",   ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7"]),
    ("Grünfeld Defence",        ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5"]),
    ("Queen's Gambit Declined", ["d2d4","d7d5","c2c4","e7e6"]),
    ("Queen's Gambit Accepted", ["d2d4","d7d5","c2c4","d5c4"]),
    ("Slav Defence",            ["d2d4","d7d5","c2c4","c7c6"]),
    ("Italian Game",            ["e2e4","e7e5","g1f3","b8c6","f1c4"]),
    ("Ruy Lopez",               ["e2e4","e7e5","g1f3","b8c6","f1b5"]),
    ("Scotch Game",             ["e2e4","e7e5","g1f3","b8c6","d2d4"]),
    ("King's Gambit",           ["e2e4","e7e5","f2f4"]),
    ("Vienna Game",             ["e2e4","e7e5","b1c3"]),
    ("Sicilian Defence",        ["e2e4","c7c5"]),
    ("French Defence",          ["e2e4","e7e6"]),
    ("Caro-Kann Defence",       ["e2e4","c7c6"]),
    ("Pirc Defence",            ["e2e4","d7d6"]),
    ("Alekhine's Defence",      ["e2e4","g8f6"]),
    ("Queen's Gambit",          ["d2d4","d7d5","c2c4"]),
    ("London System",           ["d2d4","d7d5","g1f3","g8f6","c1f4"]),
    ("Dutch Defence",           ["d2d4","f7f5"]),
    ("English Opening",         ["c2c4"]),
    ("Réti Opening",            ["g1f3","d7d5"]),
    ("King's Pawn Opening",     ["e2e4"]),
    ("Queen's Pawn Opening",    ["d2d4"]),
]


def identify_opening(board: chess.Board) -> Tactic | None:
    moves = [m.uci() for m in board.move_stack]
    if not moves:
        return None
    for name, seq in sorted(_OPENINGS, key=lambda x: len(x[1]), reverse=True):
        if moves[:len(seq)] == seq:
            return Tactic(
                name="Opening",
                squares=[],
                description=name,
                color=TACTIC_COLORS["Opening"],
            )
    return None


# ── Tactical detectors ───────────────────────────────────────────────────────

def find_forks(board: chess.Board) -> list[Tactic]:
    """One piece attacks two or more opponent pieces of equal or greater value."""
    results = []
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != color or piece.piece_type == chess.KING:
                continue
            targets = [
                tsq for tsq in board.attacks(sq)
                if (t := board.piece_at(tsq)) and t.color == opp
                and PIECE_VALUE[t.piece_type] >= PIECE_VALUE[piece.piece_type]
            ]
            if len(targets) >= 2:
                side = "White" if color == chess.WHITE else "Black"
                pname = PIECE_NAME[piece.piece_type]
                results.append(Tactic(
                    name="Fork",
                    squares=[sq] + targets,
                    description=(
                        f"{side} {pname} on {chess.square_name(sq)} "
                        f"forks {len(targets)} pieces"
                    ),
                    color=TACTIC_COLORS["Fork"],
                ))
    return results


def find_pins(board: chess.Board) -> list[Tactic]:
    """Piece pinned to its own king (absolute pin)."""
    results = []
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != color or piece.piece_type == chess.KING:
                continue
            if not board.is_pinned(color, sq):
                continue
            # Walk the pin ray to find the attacker
            pin_ray = board.pin(color, sq)
            attacker_sq = next(
                (s for s in pin_ray
                 if s != king_sq and s != sq
                 and (p := board.piece_at(s)) and p.color != color),
                None,
            )
            side = "White" if color == chess.WHITE else "Black"
            pname = PIECE_NAME[piece.piece_type]
            sq_list = [sq, king_sq] + ([attacker_sq] if attacker_sq else [])
            results.append(Tactic(
                name="Pin",
                squares=sq_list,
                description=f"{side} {pname} on {chess.square_name(sq)} is pinned to the king",
                color=TACTIC_COLORS["Pin"],
            ))
    return results


def find_skewers(board: chess.Board) -> list[Tactic]:
    """High-value piece attacked; moving it exposes a lower-value piece behind it."""
    results = []
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != color:
                continue
            if not board.is_attacked_by(opp, sq):
                continue
            # Check each attacker: is there a less-valuable piece behind the target?
            for att_sq in board.attackers(opp, sq):
                attacker = board.piece_at(att_sq)
                if attacker is None:
                    continue
                if attacker.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                    continue
                # Ray from attacker through target square
                diff_file = chess.square_file(sq) - chess.square_file(att_sq)
                diff_rank = chess.square_rank(sq) - chess.square_rank(att_sq)
                steps = max(abs(diff_file), abs(diff_rank))
                if steps == 0:
                    continue
                df = diff_file // steps
                dr = diff_rank // steps
                # Walk beyond the target
                f, r = chess.square_file(sq) + df, chess.square_rank(sq) + dr
                while 0 <= f <= 7 and 0 <= r <= 7:
                    behind_sq = chess.square(f, r)
                    behind = board.piece_at(behind_sq)
                    if behind:
                        if (behind.color == color and
                                PIECE_VALUE[behind.piece_type] < PIECE_VALUE[piece.piece_type]):
                            side = "White" if color == chess.WHITE else "Black"
                            pname = PIECE_NAME[piece.piece_type]
                            results.append(Tactic(
                                name="Skewer",
                                squares=[att_sq, sq, behind_sq],
                                description=(
                                    f"{side} {pname} on {chess.square_name(sq)} "
                                    f"is skewered — {chess.square_name(behind_sq)} exposed"
                                ),
                                color=TACTIC_COLORS["Skewer"],
                            ))
                        break
                    f += df
                    r += dr
    return results


def find_hanging(board: chess.Board) -> list[Tactic]:
    """Undefended pieces under attack (free captures)."""
    results = []
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != color or piece.piece_type == chess.KING:
                continue
            if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
                side = "White" if color == chess.WHITE else "Black"
                pname = PIECE_NAME[piece.piece_type]
                results.append(Tactic(
                    name="Hanging Piece",
                    squares=[sq],
                    description=f"{side} {pname} on {chess.square_name(sq)} is undefended",
                    color=TACTIC_COLORS["Hanging Piece"],
                ))
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def detect_tactics(board: chess.Board) -> list[Tactic]:
    """Run all detectors. Returns list of Tactic objects, deduplicated by square set."""
    tactics: list[Tactic] = []
    tactics.extend(find_forks(board))
    tactics.extend(find_pins(board))
    tactics.extend(find_skewers(board))
    tactics.extend(find_hanging(board))
    opening = identify_opening(board)
    if opening:
        tactics.append(opening)
    return tactics
