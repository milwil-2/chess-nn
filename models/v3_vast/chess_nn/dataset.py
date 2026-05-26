"""
Dataset: convert PGN games into tensors the model can train on.

For each game, we walk through every position and create a training example:
  - Input:  board tensor (105, 8, 8)
  - Policy target: the move that was actually played (as an index 0-4671)
  - Value target:  WDL class from current player's perspective
                   0 = win, 1 = draw, 2 = loss

We save in chunks of CHUNK_SIZE positions as .npz files to keep memory usage low.
"""

import os
import sys
import bisect
from collections import deque
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TRAIN_SPLIT, VAL_SPLIT, BATCH_SIZE, POLICY_OUTPUT_SIZE, DATA_WORKERS
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.move_encoding import move_to_index, get_legal_move_indices

CHUNK_SIZE = 20_000


def result_to_class(result: str, turn: bool) -> int:
    """Convert PGN result to WDL class from current player's perspective.
    0 = win, 1 = draw, 2 = loss.
    """
    if result == "1-0":
        return 0 if turn == chess.WHITE else 2
    elif result == "0-1":
        return 2 if turn == chess.WHITE else 0
    else:
        return 1


def process_games(pgn_path: str, output_dir: str = None) -> list[str]:
    """
    Read a PGN file, convert every position to training examples, save as .npz chunks.
    Returns list of paths to the saved chunk files.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    boards, policies, values, masks = [], [], [], []
    chunk_idx = 0
    chunk_paths = []
    total_positions = 0

    print(f"Processing games from: {pgn_path}")

    games_seen = 0
    games_kept = 0

    with open(pgn_path) as pgn_file:
        pbar = tqdm(desc="Processing", unit="pos", dynamic_ncols=True)

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            games_seen += 1
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            board = game.board()
            history: deque = deque(maxlen=8)  # boards[0] = most recent
            for move in game.mainline_moves():
                history.appendleft(board.copy(stack=False))
                board_tensor = boards_to_tensor(list(history))
                move_idx = move_to_index(move, board)
                value = result_to_class(result, board.turn)

                legal_mask = np.zeros(POLICY_OUTPUT_SIZE, dtype=bool)
                for idx in get_legal_move_indices(board):
                    legal_mask[idx] = True

                boards.append(board_tensor)
                policies.append(move_idx)
                values.append(value)
                masks.append(legal_mask)

                board.push(move)
                total_positions += 1
                pbar.update(1)
                pbar.set_postfix(games=games_kept, chunk=chunk_idx)

                if len(boards) >= CHUNK_SIZE:
                    path = _save_chunk(boards, policies, values, masks, output_dir, chunk_idx)
                    chunk_paths.append(path)
                    chunk_idx += 1
                    boards, policies, values, masks = [], [], [], []

            games_kept += 1

        pbar.close()

    if boards:
        path = _save_chunk(boards, policies, values, masks, output_dir, chunk_idx)
        chunk_paths.append(path)

    print(f"Total positions: {total_positions:,} across {len(chunk_paths)} chunks")
    return chunk_paths


def _save_chunk(boards, policies, values, masks, output_dir, chunk_idx):
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards=np.array(boards, dtype=np.float32),
        policies=np.array(policies, dtype=np.int64),
        values=np.array(values, dtype=np.int64),
        legal_masks=np.packbits(np.array(masks, dtype=bool), axis=1),
    )
    print(f"\nSaved chunk {chunk_idx} ({len(boards):,} positions) → {path}")
    return path


class ChessDataset(Dataset):
    """
    PyTorch Dataset that loads from .npz chunk files — one chunk at a time.

    Instead of loading all positions into RAM at once (which can use 5+ GB),
    this scans chunk sizes upfront then loads one chunk (~450 MB) on demand.
    Each DataLoader worker maintains its own single-chunk cache, so peak RAM
    is roughly: (num_workers + 1) × chunk_size × ~4.5 KB per position.
    """

    def __init__(self, chunk_paths: list[str]):
        self.chunk_paths = chunk_paths
        # Companion Stockfish annotation files (chunk_NNNN_sf.npz)
        self.sf_paths = [p.replace(".npz", "_sf.npz") for p in chunk_paths]

        # Scan sizes and tactical flags — no full arrays loaded yet
        self.chunk_sizes = []
        self.chunk_tactical: list[np.ndarray | None] = []
        for path in chunk_paths:
            with np.load(path) as f:
                self.chunk_sizes.append(len(f["boards"]))
                self.chunk_tactical.append(
                    f["tactical"].astype(np.uint8) if "tactical" in f else None
                )

        # Prefix sums so we can bisect-search chunk_idx for any global idx
        self.offsets = [0]
        for sz in self.chunk_sizes:
            self.offsets.append(self.offsets[-1] + sz)

        print(f"Dataset: {self.offsets[-1]:,} positions across {len(chunk_paths)} chunks (lazy)")

        # Single-chunk cache per worker process (main + SF companion)
        self._cached_chunk_idx = None
        self._cached_data = None
        self._cached_sf = None

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        # Find which chunk owns this index
        chunk_idx = bisect.bisect_right(self.offsets, idx) - 1
        chunk_idx = max(0, min(chunk_idx, len(self.chunk_paths) - 1))
        local_idx = idx - self.offsets[chunk_idx]

        # Load chunk into RAM if it's not already cached
        if chunk_idx != self._cached_chunk_idx:
            self._cached_data = dict(np.load(self.chunk_paths[chunk_idx]))
            sf_path = self.sf_paths[chunk_idx]
            self._cached_sf = dict(np.load(sf_path)) if os.path.exists(sf_path) else None
            self._cached_chunk_idx = chunk_idx

        data = self._cached_data
        mask = np.unpackbits(data["legal_masks"][local_idx])[:POLICY_OUTPUT_SIZE].astype(bool)
        sf_pol = int(self._cached_sf["sf_policy"][local_idx]) if self._cached_sf is not None else -1
        return (
            torch.from_numpy(data["boards"][local_idx].copy()),
            torch.tensor(int(data["policies"][local_idx]), dtype=torch.long),
            torch.tensor(int(data["values"][local_idx]), dtype=torch.long),
            torch.from_numpy(mask),
            torch.tensor(sf_pol, dtype=torch.long),
        )


TACTICAL_OVERSAMPLE = 3  # tactical positions sampled 3× more often during training


class ChunkBatchSampler(Sampler):
    """
    Yield batches where every index belongs to the same .npz chunk.

    With shuffle=False (validation), items are ordered sequentially within chunks.
    With shuffle=True (training), chunk order and within-chunk order are randomised
    each epoch. If the dataset has tactical flags, tactical positions are oversampled
    TACTICAL_OVERSAMPLE× via weighted replacement sampling.
    """

    def __init__(self, subset, batch_size: int, shuffle: bool = True):
        from collections import defaultdict
        dataset = subset.dataset
        chunk_groups: dict[int, list[int]] = defaultdict(list)
        for local_idx, gi in enumerate(subset.indices):
            ci = bisect.bisect_right(dataset.offsets, gi) - 1
            chunk_groups[ci].append(local_idx)

        self._groups = list(chunk_groups.values())
        self._batch_size = batch_size
        self._shuffle = shuffle

        # Build per-group weights for tactical oversampling (training only)
        self._group_weights: list[list[float] | None] = []
        chunk_ids = sorted(chunk_groups.keys())
        for ci in chunk_ids:
            tac = dataset.chunk_tactical[ci] if ci < len(dataset.chunk_tactical) else None
            if tac is None or not shuffle:
                self._group_weights.append(None)
            else:
                # local_idx → within-chunk position index
                local_indices = chunk_groups[ci]
                base_offset = dataset.offsets[ci]
                weights = [
                    float(TACTICAL_OVERSAMPLE) if tac[gi - base_offset] else 1.0
                    for gi in (subset.indices[li] for li in local_indices)
                ]
                self._group_weights.append(weights)

    def __iter__(self):
        import random
        groups = [g[:] for g in self._groups]
        if self._shuffle:
            random.shuffle(groups)
            paired = list(zip(groups, self._group_weights))
            random.shuffle(paired)
            result_groups = []
            for g, w in paired:
                if w is not None:
                    # Weighted sampling with replacement — same length as original group
                    sampled = random.choices(g, weights=w, k=len(g))
                else:
                    sampled = random.sample(g, len(g))
                result_groups.append(sampled)
            groups = result_groups
        for group in groups:
            for i in range(0, len(group), self._batch_size):
                batch = group[i:i + self._batch_size]
                if batch:
                    yield batch

    def __len__(self) -> int:
        return sum(
            (len(g) + self._batch_size - 1) // self._batch_size
            for g in self._groups
        )


def make_dataloaders(chunk_paths: list[str]):
    """Split data into train/val/test DataLoaders."""
    dataset = ChessDataset(chunk_paths)
    n = len(dataset)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_sampler = ChunkBatchSampler(train_set, BATCH_SIZE, shuffle=True)
    val_sampler   = ChunkBatchSampler(val_set,   BATCH_SIZE, shuffle=False)
    test_sampler  = ChunkBatchSampler(test_set,  BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=DATA_WORKERS,
                              pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_set,   batch_sampler=val_sampler,   num_workers=DATA_WORKERS,
                              pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_set,  batch_sampler=test_sampler,  num_workers=DATA_WORKERS,
                              pin_memory=True, persistent_workers=True, prefetch_factor=2)

    print(f"Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import glob
    from config import RAW_DATA_DIR

    pgn_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.pgn")))
    if not pgn_files:
        print(f"No PGN files found in {RAW_DATA_DIR}. Run data/download_data.py first.")
        sys.exit(1)

    for pgn_path in pgn_files:
        process_games(pgn_path)
