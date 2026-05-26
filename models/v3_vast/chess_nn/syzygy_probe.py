"""
Syzygy endgame tablebase probe for MCTS.

For positions with <= 6 pieces total, Syzygy gives the *true* win/draw/loss
result under optimal play — no learning needed. We use this to override the
network's value head (which has a measured side-to-move bias) and to filter
MCTS root priors down to moves that don't throw away a winning tablebase
outcome.

Tables must be downloaded separately (Syzygy 3-4-5 piece tables are ~1 GB).
Standard mirrors:
  https://syzygy-tables.info/
  http://tablebase.sesse.net/syzygy/3-4-5/

Point SYZYGY_PATH (config.py) at the directory containing the .rtbw / .rtbz
files. If the path is missing or empty, SyzygyTable.is_available is False
and every probe returns None — MCTS then falls back to network evaluation.

This is inference-only — never wired into self-play. Self-play sees raw
network values so training learns to evaluate endgames itself.
"""

import os
import sys

import chess
import chess.syzygy


_TB_MAX_PIECES = 6  # current Syzygy releases cover up to 7-man; we keep 6 for safety


class SyzygyTable:
    """Wraps `chess.syzygy.Tablebase`. Opens on init, closes on `close()` or
    GC. All probe methods return None when the table is unavailable, when the
    position has too many pieces, or when probing raises (corrupt file etc.)."""

    def __init__(self, path: str | None = None, max_pieces: int = _TB_MAX_PIECES):
        self.path = path
        self.max_pieces = max_pieces
        self._tb: chess.syzygy.Tablebase | None = None

        if not path:
            return
        if not os.path.isdir(path):
            print(f"[syzygy] tablebase dir not found: {path} — disabling probe", file=sys.stderr)
            return
        try:
            self._tb = chess.syzygy.open_tablebase(path)
            n = len(os.listdir(path))
            print(f"[syzygy] loaded {n} files from {path}", file=sys.stderr)
        except Exception as e:
            print(f"[syzygy] failed to open tablebase at {path}: {e}", file=sys.stderr)
            self._tb = None

    @property
    def is_available(self) -> bool:
        return self._tb is not None

    def applies(self, board: chess.Board) -> bool:
        """True iff this position is small enough for a probe to make sense.
        Castling rights disable Syzygy (they aren't in the table encoding)."""
        if self._tb is None:
            return False
        if board.castling_rights:
            return False
        return chess.popcount(board.occupied) <= self.max_pieces

    def probe_wdl(self, board: chess.Board) -> int | None:
        """Return WDL from the side-to-move's POV:
            +2 = win, +1 = cursed win (50-move rule),
             0 = draw,
            -1 = blessed loss, -2 = loss.
        Returns None if the table can't answer."""
        if not self.applies(board):
            return None
        try:
            return self._tb.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError, IndexError, OSError):
            return None
        except Exception:
            return None

    def value_scalar(self, board: chess.Board) -> float | None:
        """WDL → scalar in [-1, 1] (current-player POV), to match
        `wdl_to_scalar()`'s contract. Cursed/blessed treated as draws because
        the network's value head doesn't distinguish them either."""
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        if wdl >= 2:
            return 1.0
        if wdl <= -2:
            return -1.0
        return 0.0

    def best_moves(self, board: chess.Board) -> set[chess.Move] | None:
        """Return the set of legal moves that preserve the best achievable
        WDL outcome under optimal play. Returns None if the table doesn't
        cover this position. Kept for backwards compatibility — prefer
        `best_progress_moves` which adds DTZ tie-breaking.

        Note: probes EVERY legal move's resulting WDL (negated, since after
        the move it's the opponent's turn). With <=6 men this is cheap (~30
        probes × ~1 ms = under 50 ms)."""
        if not self.applies(board):
            return None
        scores: dict[chess.Move, int] = {}
        for move in board.legal_moves:
            scratch = board.copy(stack=False)
            scratch.push(move)
            try:
                if scratch.castling_rights:
                    return None
                wdl = self._tb.probe_wdl(scratch)
            except Exception:
                return None
            scores[move] = -wdl
        if not scores:
            return None
        best = max(scores.values())
        return {m for m, s in scores.items() if s == best}

    def best_progress_moves(self, board: chess.Board) -> set[chess.Move] | None:
        """Return the set of legal moves that preserve the best WDL **AND**
        make fastest progress per DTZ (distance-to-zero, i.e. plies until
        the next pawn move or capture).

        For a winning side, return moves with the lowest DTZ (force progress).
        For a losing side, return moves with the highest DTZ (best defense).
        For drawn positions, DTZ doesn't matter — return all draw-preserving
        moves.

        Returns None if the table doesn't cover the position. This is the
        method MCTS should use to mask root priors — `best_moves` alone
        returns too many "equally winning" moves and the engine can't pick
        progress-makers from waiting moves.

        Sign convention: python-chess `probe_dtz(board)` returns DTZ from
        the current-player POV — positive when winning (plies-to-zero with
        win preserved), negative when losing (plies-to-zero before losing).
        After pushing our move it's the OPPONENT's turn, so we negate."""
        if not self.applies(board):
            return None
        wdls: dict[chess.Move, int] = {}
        dtzs: dict[chess.Move, int] = {}
        for move in board.legal_moves:
            scratch = board.copy(stack=False)
            scratch.push(move)
            if scratch.castling_rights:
                # extremely unlikely (would require castling-back) — skip.
                continue
            try:
                wdl_after = self._tb.probe_wdl(scratch)
                dtz_after = self._tb.probe_dtz(scratch)
            except Exception:
                # Some 4-piece tables (e.g. KBNvK on sesse.net mirror) trigger
                # MissingTableError in python-chess. Skip this move rather
                # than aborting the whole override — partial coverage is
                # better than none.
                continue
            wdls[move] = -wdl_after
            # DTZ from mover's POV: negate the opponent-POV DTZ.
            # |dtz| measures distance to next zeroing move regardless of sign;
            # for our tie-breaking we want minimum |dtz| for winning,
            # maximum |dtz| for losing.
            dtzs[move] = -dtz_after
        if not wdls:
            return None
        best_wdl = max(wdls.values())
        wdl_preserving = [m for m, s in wdls.items() if s == best_wdl]
        if best_wdl >= 1:
            # Winning side — minimize |DTZ| to force progress.
            # |dtz| can be 0 for moves that immediately reach 50-move boundary;
            # those moves are typically pawn pushes or captures, which we want.
            best_dtz = min(abs(dtzs[m]) for m in wdl_preserving)
            return {m for m in wdl_preserving if abs(dtzs[m]) == best_dtz}
        if best_wdl <= -1:
            # Losing side — maximize |DTZ| to delay the loss as long as possible.
            best_dtz = max(abs(dtzs[m]) for m in wdl_preserving)
            return {m for m in wdl_preserving if abs(dtzs[m]) == best_dtz}
        # Drawn — any WDL-preserving move is fine.
        return set(wdl_preserving)

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass
            self._tb = None

    def __del__(self):
        self.close()
