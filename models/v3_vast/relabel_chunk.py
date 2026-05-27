"""One-shot: read chunk_0000.npz with CP-POV value labels and write
chunk_0000_whitepov.npz with W-POV labels (GH #6, Phase 4a pilot).

The board tensor's plane 101 is the side-to-move plane (1.0 = white to move,
0.0 = black to move) — it records the ABSOLUTE turn even though the rest of
the encoding is mirrored to current-player perspective. We use it to gate
the value flip:
  * stm=white  → CP-POV == W-POV → unchanged
  * stm=black  → CP-POV != W-POV → swap 0 (mover-wins) ↔ 2 (mover-loses);
                                    1 (draw) is symmetric

The original chunk is left untouched. Wave-F deliberately writes the
re-labeled chunk to a NEW filename so the pilot is apples-to-apples
against the CP-POV baseline.
"""

import os
import sys

import numpy as np

# Run from the v3_vast dir so the default relative paths resolve to
# data/processed/. Override SRC/DST via env vars if needed.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SRC = os.environ.get(
    "RELABEL_SRC",
    os.path.join(HERE, "data", "processed", "chunk_0000.npz"),
)
DST = os.environ.get(
    "RELABEL_DST",
    os.path.join(HERE, "data", "processed", "chunk_0000_whitepov.npz"),
)


def main() -> None:
    if not os.path.exists(SRC):
        print(f"ERROR: source chunk not found: {SRC}", file=sys.stderr)
        print(
            "       (Expected the Vast-encoded v3_vast chunk. Sync it from the "
            "Vast.ai instance or run data/download_data.py locally first.)",
            file=sys.stderr,
        )
        sys.exit(1)

    d = np.load(SRC)
    print(f"Loaded {SRC}  keys={list(d.keys())}")

    boards = d["boards"]  # shape (N, 105, 8, 8) float32
    values = d["values"]  # shape (N,) int64 in {0,1,2}, CP-POV

    if boards.ndim != 4 or boards.shape[1] != 105:
        print(
            f"ERROR: expected boards shape (N, 105, 8, 8); got {boards.shape}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Plane 101 is the side-to-move plane (constant across all 64 squares).
    # 1.0 = white to move, 0.0 = black to move.
    stm = boards[:, 101, 0, 0]
    black_to_move = stm < 0.5
    new_values = values.copy()
    # Flip 0 ↔ 2 for positions where black was to move; leave draws (1) alone.
    flip_mask = black_to_move & (new_values != 1)
    new_values[flip_mask] = 2 - new_values[flip_mask]

    n = len(values)
    print(f"positions: {n}")
    print(f"  black-to-move (flipped): {int(flip_mask.sum())}")
    print(f"  white-to-move:           {int((~black_to_move).sum())}")
    cp_dist = np.bincount(values, minlength=3).tolist()
    wpov_dist = np.bincount(new_values, minlength=3).tolist()
    print(f"  CP-POV class distribution: W={cp_dist[0]} D={cp_dist[1]} L={cp_dist[2]}")
    print(f"  W-POV class distribution:  W={wpov_dist[0]} D={wpov_dist[1]} L={wpov_dist[2]}")

    # Preserve every other key (policies, legal_masks, tactical, sf_*, etc.)
    save = {k: d[k] for k in d.keys()}
    save["values"] = new_values
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    np.savez_compressed(DST, **save)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
