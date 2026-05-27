"""
Cross-game transposition cache for MCTS.

Stores `position_key → tag → {move_uci: visit_count}` across MCTS searches. On
a fresh search at a known position, we mix the cached visit distribution into
the network's raw policy priors so MCTS gets a head start instead of
re-searching the same opening every game.

The `tag` segment disambiguates entries written by different checkpoints (or
any other context the caller wants to keep isolated), so a strong network's
priors don't get blended with visit counts produced by a weaker one. See
GitHub issue #27.

Inference-only. Self-play passes `tcache=None` so training sees the raw
network policy without contamination from past inference runs.

Persistence: optional. Call `save()` to flush to disk; `load()` is called
automatically on init if the path exists. Format is JSON (compact, human
readable, easy to inspect or hand-edit if needed). A periodic auto-save
fires from `record()` every 5 minutes so a crash loses at most that much
search work (GitHub issue #18).
"""

import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict

import chess


# Auto-save threshold: every 5 minutes of wall-clock time, `record()` will
# flush the cache to disk. Synchronous (~1 ms cost per save) — see issue #18.
_AUTO_SAVE_INTERVAL_S = 300.0


def position_key(board: chess.Board) -> str:
    """Stable transposition key — piece placement + STM + castling + EP.
    Drops the move counters so we collapse half-move-count transpositions."""
    return " ".join(board.fen().split()[:4])


class TranspositionCache:
    """LRU cache mapping position_key → tag → {uci_move: visit_count}.
    Max size caps memory; oldest positions are evicted on overflow.

    `tag` segments entries so a single on-disk cache can hold visit
    distributions from multiple checkpoints without cross-contamination.
    A `lookup()` only ever returns entries written under this instance's
    tag. On load, legacy single-segment caches are silently migrated under
    `tag="default"`.
    """

    def __init__(self, path: str | None = None, max_size: int = 50_000,
                 min_visits_to_record: int = 20, tag: str | None = None):
        self.path = path
        self.max_size = max_size
        self.min_visits_to_record = min_visits_to_record
        self.tag = tag if tag is not None else "default"
        # New layout: position_key -> {tag -> {move_uci: visits}}
        self._data: "OrderedDict[str, Dict[str, Dict[str, int]]]" = OrderedDict()
        self._last_save = time.monotonic()

        if path and os.path.exists(path):
            self.load(path)

    # ---------- read path ----------

    def lookup(self, board: chess.Board) -> Dict[chess.Move, int] | None:
        """Return cached {Move: visits} for this position under the current
        tag, or None on miss. Move objects are reconstructed from UCI;
        entries whose move is no longer legal (rule changes? shouldn't
        happen) are silently dropped."""
        key = position_key(board)
        tag_map = self._data.get(key)
        if tag_map is None:
            return None
        entry = tag_map.get(self.tag)
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
        """Store an MCTS root's visit distribution under the current tag.
        Skips noisy / shallow searches by requiring at least
        `min_visits_to_record` total visits. Periodically auto-saves to
        disk so a crash loses at most ~5 min of search work (issue #18)."""
        total = sum(visits.values())
        if total < self.min_visits_to_record:
            return
        key = position_key(board)
        entry = {m.uci(): int(n) for m, n in visits.items() if n > 0}
        if not entry:
            return
        tag_map = self._data.get(key)
        if tag_map is not None:
            # Blend by addition under THIS tag — running total across all
            # searches that touched this position with the same checkpoint.
            existing = tag_map.get(self.tag)
            if existing is not None:
                for uci, n in entry.items():
                    existing[uci] = existing.get(uci, 0) + n
            else:
                tag_map[self.tag] = entry
            self._data.move_to_end(key)
        else:
            self._data[key] = {self.tag: entry}
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)  # evict LRU

        # Periodic auto-save (issue #18). Synchronous — ~1 ms for our
        # cache sizes, no thread needed.
        if self.path and (time.monotonic() - self._last_save) > _AUTO_SAVE_INTERVAL_S:
            self.save()
            self._last_save = time.monotonic()

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
            migrated = 0
            for key, entry in blob.get("entries", []):
                if not isinstance(entry, dict) or not entry:
                    continue
                # Detect format by inspecting the first value:
                #   new format: {tag: {uci: count}}  -> value is a dict
                #   old format: {uci: count}         -> value is an int
                sample_val = next(iter(entry.values()))
                if isinstance(sample_val, dict):
                    # New format — copy as-is, normalising types.
                    tag_map: Dict[str, Dict[str, int]] = {}
                    for tag, moves in entry.items():
                        if not isinstance(moves, dict):
                            continue
                        tag_map[str(tag)] = {str(k): int(v) for k, v in moves.items()}
                    if tag_map:
                        self._data[key] = tag_map
                else:
                    # Old format — wrap under THIS instance's tag for
                    # backwards compat. (The spec calls this "tag='default'"
                    # because that's the no-tag default value; in practice
                    # we adopt the loader's tag so the loader actually sees
                    # the migrated entries on lookup.)
                    self._data[key] = {
                        self.tag: {str(k): int(v) for k, v in entry.items()}
                    }
                    migrated += 1
            msg = f"[tcache] loaded {len(self._data)} positions from {target}"
            if migrated:
                msg += f" (migrated {migrated} legacy entries to tag={self.tag!r})"
            print(msg, file=sys.stderr)
        except Exception as e:
            print(f"[tcache] load failed: {e}", file=sys.stderr)
            self._data = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)
