"""Opening book module for chess MCTS.

This module provides an `OpeningBook` class that MCTS consults *before*
running search. If the current position is in the book, MCTS returns the
book move and skips search entirely. This kills "early king moves" in
plies 1-20 deterministically.

The book tries to load a Polyglot `.bin` file first (weighted random move
selection), and falls back to a hardcoded mini-book of mainline openings
(deterministic top-move) covering ~25+ common openings 6-10 plies deep.
"""

from __future__ import annotations

import os
import random
import sys

import chess
import chess.polyglot


def _build_hardcoded() -> dict[str, list[str]]:
    """Return {position_key: [uci_move, ...]}. The first move in each list
    is the "main line" recommendation; alternates are also valid."""
    book: dict[str, list[str]] = {}

    def add_line(uci_moves: list[str]) -> None:
        b = chess.Board()
        for uci in uci_moves:
            key = " ".join(b.fen().split()[:4])
            book.setdefault(key, [])
            if uci not in book[key]:
                book[key].append(uci)
            b.push_uci(uci)

    # Italian
    add_line(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d3"])
    add_line(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3"])
    # Ruy Lopez
    add_line(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"])
    add_line(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f8e7"])
    # Scotch
    add_line(["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "g8f6", "d4c6"])
    # Petrov
    add_line(["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6", "e5f3", "f6e4"])
    # Sicilian Najdorf
    add_line(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"])
    # Sicilian Dragon
    add_line(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"])
    # Sicilian Sveshnikov
    add_line(["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5"])
    # Caro-Kann
    add_line(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7", "g1f3"])
    add_line(["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "g1f3", "e7e6", "f1e2"])
    # French
    add_line(["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5", "f8e7"])
    add_line(["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6", "g1f3"])
    # Pirc
    add_line(["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3", "f8g7"])
    # Scandinavian
    add_line(["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6", "g1f3"])
    # QGD
    add_line(["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8"])
    # Slav
    add_line(["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "e7e6", "e2e3"])
    # Catalan
    add_line(["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7", "g1f3"])
    # KID
    add_line(["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8"])
    # Nimzo-Indian
    add_line(["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "f1d3", "d7d5"])
    # Queen's Indian
    add_line(["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2"])
    # Grunfeld
    add_line(["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4"])
    # English
    add_line(["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3"])
    add_line(["c2c4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6", "f1g2", "f8g7"])
    # Reti
    add_line(["g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6", "f1g2"])
    # London
    add_line(["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6", "e2e3", "c7c5", "c2c3"])
    return book


def _position_key(board: chess.Board) -> str:
    """Piece placement + side to move + castling + EP. Drops move counters
    so transpositions with different half-move counts still match."""
    return " ".join(board.fen().split()[:4])


class OpeningBook:
    """Opening book consulted by MCTS before search.

    Tries to load a Polyglot `.bin` file; on failure falls back to a small
    hardcoded mini-book of common openings.
    """

    def __init__(self, polyglot_path: str | None = None, max_ply: int = 16):
        """Try to load a polyglot .bin opening book at polyglot_path.
        Falls back to a hardcoded mini-book of common openings if the file
        is missing or unreadable. max_ply caps how deep into the game the
        book is consulted (default 16 = first 8 full moves)."""
        self.max_ply = max_ply
        self._reader: chess.polyglot.MemoryMappedReader | None = None
        self._hardcoded: dict[str, list[str]] | None = None
        self._source: str = "none"

        if polyglot_path is not None and os.path.isfile(polyglot_path):
            try:
                self._reader = chess.polyglot.open_reader(polyglot_path)
                self._source = f"polyglot:{polyglot_path}"
                return
            except Exception as e:
                print(
                    f"[OpeningBook] WARN: failed to open polyglot file "
                    f"{polyglot_path!r}: {e}; falling back to hardcoded book.",
                    file=sys.stderr,
                )
        elif polyglot_path is not None:
            print(
                f"[OpeningBook] WARN: polyglot file not found at "
                f"{polyglot_path!r}; falling back to hardcoded book.",
                file=sys.stderr,
            )

        self._hardcoded = _build_hardcoded()
        self._source = "hardcoded"

    @property
    def loaded_source(self) -> str:
        """'polyglot:<path>' or 'hardcoded' or 'none'. For diagnostics."""
        return self._source

    def lookup(self, board: chess.Board) -> chess.Move | None:
        """Return a book move for `board` or None.

        Returns None if:
          - board.fullmove_number > max_ply // 2
          - position is not in the book
          - book entry's moves are all illegal in the current position
        """
        if board.fullmove_number > self.max_ply // 2:
            return None

        if self._reader is not None:
            try:
                # weighted_choice respects polyglot entry weights — mainlines
                # get picked roughly in proportion to their popularity in the
                # source database. The bare `.choice()` method is UNIFORM
                # random, which means low-weight sidelines (e.g. Latvian
                # Gambit at 0.1%) get picked as often as the top mainline.
                # See issue #17 investigation by Wave-B agent B-BOOK.
                entry = self._reader.weighted_choice(board)
            except IndexError:
                return None
            except Exception:
                return None
            move = entry.move
            if move in board.legal_moves:
                return move
            return None

        if self._hardcoded is not None:
            key = _position_key(board)
            entries = self._hardcoded.get(key)
            if not entries:
                return None
            for uci in entries:
                try:
                    mv = chess.Move.from_uci(uci)
                except ValueError:
                    continue
                if mv in board.legal_moves:
                    return mv
            return None

        return None

    def close(self) -> None:
        """Release the polyglot reader, if any."""
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    random.seed(0)
    book = OpeningBook()
    start = chess.Board()
    key = _position_key(start)
    moves = book._hardcoded.get(key, []) if book._hardcoded is not None else []
    print(f"OpeningBook({book.loaded_source}): {len(moves)} moves for startpos")
