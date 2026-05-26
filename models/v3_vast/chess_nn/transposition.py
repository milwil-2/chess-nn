"""
Cross-game transposition cache for MCTS.

Stores `position_key → {move_uci: visit_count}` across MCTS searches. On a
fresh search at a known position, we mix the cached visit distribution into
the network's raw policy priors so MCTS gets a head start instead of
re-searching the same opening every game.

Inference-only. Self-play passes `tcache=None` so training sees the raw
network policy without contamination from past inference runs.

Persistence: optional. Call `save()` to flush to disk; `load()` is called
automatically on init if the path exists. Format is JSON (compact, human
readable, easy to inspect or hand-edit if needed).
"""

import json
import os
import sys
from collections import OrderedDict
from typing import Dict

import chess


def position_key(board: chess.Board) -> str:
    """Stable transposition key — piece placement + STM + castling + EP.
    Drops the move counters so we collapse half-move-count transpositions."""
    return " ".join(board.fen().split()[:4])


class TranspositionCache:
    """LRU cache mapping position_key → {uci_move: visit_count}.
    Max size caps memory; oldest positions are evicted on overflow."""

    def __init__(self, path: str | None = None, max_size: int = 50_000,
                 min_visits_to_record: int = 20):
        self.path = path
        self.max_size = max_size
        self.min_visits_to_record = min_visits_to_record
        self._data: "OrderedDict[str, Dict[str, int]]" = OrderedDict()

        if path and os.path.exists(path):
            self.load(path)

    # ---------- read path ----------

    def lookup(self, board: chess.Board) -> Dict[chess.Move, int] | None:
        """Return cached {Move: visits} for this position, or None on miss.
        Move objects are reconstructed from UCI; entries whose move is no
        longer legal (rule changes? shouldn't happen) are silently dropped."""
        key = position_key(board)
        entry = self._data.get(key)
        if entry is None:
            return None
        # Move recently-used entry to the end (LRU touch).
        self._data.move_to_end(key)
        out: Dict[chess.Move, int] = {}
        for uci, n in entry.items():
            try:
                m = chess.Move.from_uci(uci)
            except Exception:
                continue
            if m in board.legal_moves:
                out[m] = int(n)
        return out or None

    # ---------- write path ----------

    def record(self, board: chess.Board, visits: Dict[chess.Move, int]) -> None:
        """Store an MCTS root's visit distribution. Skips noisy / shallow
        searches by requiring at least `min_visits_to_record` total visits."""
        total = sum(visits.values())
        if total < self.min_visits_to_record:
            return
        key = position_key(board)
        entry = {m.uci(): int(n) for m, n in visits.items() if n > 0}
        if not entry:
            return
        # If we already have an entry, blend by addition (running running total
        # across all searches that touched this position).
        existing = self._data.get(key)
        if existing is not None:
            for uci, n in entry.items():
                existing[uci] = existing.get(uci, 0) + n
            self._data.move_to_end(key)
        else:
            self._data[key] = entry
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)  # evict LRU

    # ---------- persistence ----------

    def save(self, path: str | None = None) -> None:
        target = path or self.path
        if not target:
            return
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = target + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"entries": list(self._data.items())}, fh)
            os.replace(tmp, target)
        except Exception as e:
            print(f"[tcache] save failed: {e}", file=sys.stderr)

    def load(self, path: str | None = None) -> None:
        target = path or self.path
        if not target or not os.path.exists(target):
            return
        try:
            with open(target) as fh:
                blob = json.load(fh)
            self._data = OrderedDict()
            for key, entry in blob.get("entries", []):
                if isinstance(entry, dict):
                    self._data[key] = {str(k): int(v) for k, v in entry.items()}
            print(f"[tcache] loaded {len(self._data)} positions from {target}", file=sys.stderr)
        except Exception as e:
            print(f"[tcache] load failed: {e}", file=sys.stderr)
            self._data = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)
