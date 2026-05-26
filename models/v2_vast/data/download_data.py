"""
Combined streaming pipeline: Lichess → filter → encode → .npz chunks.

Replaces the separate download_data.py + dataset.py preprocessing steps.
Three parallel processes (one per month) each:
  1. Stream .zst directly from Lichess (no full archive stored)
  2. Read game headers only for low-rated games (chess.pgn SKIP) — avoids
     parsing moves for the ~85% of games that fail the 2000+ Elo filter
  3. Encode qualifying positions to board tensors + policy/value targets
  4. Write chunk_{NNNN}.npz when the position buffer fills

Result: ~2-3× faster than sequential download+dataset.py, no intermediate
PGN files, no double-parse.

Resume: each month writes a .{label}.done marker on completion.
To force-restart a month: delete its marker and its chunk_{NNNN}.npz files.
"""

import glob
import io
import os
import sys
import time
from collections import deque
from multiprocessing import Process

import chess
import chess.pgn
import numpy as np
import requests
import zstandard as zstd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_MOVES, MIN_RATING, POLICY_OUTPUT_SIZE, PROCESSED_DATA_DIR
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.move_encoding import get_legal_move_indices, move_to_index

def _url(ym: str) -> tuple[str, str]:
    return ym, f"https://database.lichess.org/standard/lichess_db_standard_rated_{ym}.pgn.zst"

# 12 months across 6 years (2019-2024, bi-annual) — 12 workers keeps per-stream
# bandwidth high while still covering full era/style diversity for training.
LICHESS_MONTHS = [
    _url("2024-10"), _url("2024-04"),
    _url("2023-10"), _url("2023-04"),
    _url("2022-10"), _url("2022-04"),
    _url("2021-10"), _url("2021-04"),
    _url("2020-10"), _url("2020-04"),
    _url("2019-10"), _url("2019-04"),
]
TARGET_GAMES    = 1_000_000
GAMES_PER_MONTH = TARGET_GAMES // len(LICHESS_MONTHS)   # ~41,667 per month
CHUNK_SIZE      = 20_000
MAX_RETRIES     = 5
SLOTS_PER_MONTH = 500    # pre-assigned chunk index range per month (83k games × 40 pos / 20k = ~166 chunks max)


def _wdl(result: str, turn: bool) -> int:
    if result == "1-0":
        return 0 if turn == chess.WHITE else 2
    if result == "0-1":
        return 2 if turn == chess.WHITE else 0
    return 1


class _Visitor(chess.pgn.BaseVisitor):
    """
    Fast single-pass visitor.

    For low-rated games: returns chess.pgn.SKIP from visit_header so the
    parser never touches the move text — only headers are read.
    For qualifying games: encodes every position as a (tensor, policy, value,
    legal_mask) tuple ready to write to .npz.
    """

    def begin_game(self):
        self._skip   = False
        self._result = None
        self._hist   = deque(maxlen=8)
        self._pos    = []

    def visit_header(self, tagname, tagvalue):
        if self._skip:
            return chess.pgn.SKIP
        if tagname == "Result":
            if tagvalue not in ("1-0", "0-1", "1/2-1/2"):
                self._skip = True
                return chess.pgn.SKIP
            self._result = tagvalue
        elif tagname in ("WhiteElo", "BlackElo"):
            try:
                if int(tagvalue or "0") < MIN_RATING:
                    self._skip = True
                    return chess.pgn.SKIP
            except ValueError:
                self._skip = True
                return chess.pgn.SKIP

    def visit_move(self, board, move):
        self._hist.appendleft(board.copy(stack=False))
        tensor   = boards_to_tensor(list(self._hist))
        move_idx = move_to_index(move, board)
        mask     = np.zeros(POLICY_OUTPUT_SIZE, dtype=bool)
        for idx in get_legal_move_indices(board):
            mask[idx] = True
        self._pos.append((tensor, move_idx, board.turn, mask))

    def result(self):
        if self._skip or not self._result or len(self._pos) < MIN_MOVES:
            return []
        return [
            (t, m, _wdl(self._result, turn), mask)
            for t, m, turn, mask in self._pos
        ]


def _save_chunk(boards, policies, values, masks, out_dir, idx):
    path = os.path.join(out_dir, f"chunk_{idx:04d}.npz")
    np.savez_compressed(
        path,
        boards      = np.array(boards,   dtype=np.float32),
        policies    = np.array(policies, dtype=np.int64),
        values      = np.array(values,   dtype=np.int64),
        legal_masks = np.packbits(np.array(masks, dtype=bool), axis=1),
    )


def _worker(label: str, url: str, out_dir: str, target: int,
            chunk_offset: int, tqdm_pos: int) -> None:
    done_marker = os.path.join(out_dir, f".{label}.done")
    if os.path.exists(done_marker):
        tqdm.write(f"[{label}] already complete — skipping")
        return

    # Clear partial chunks from any previous crashed run for this month
    for f in glob.glob(os.path.join(out_dir, "chunk_*.npz")):
        idx = int(os.path.basename(f)[6:10])
        if chunk_offset <= idx < chunk_offset + SLOTS_PER_MONTH:
            os.remove(f)

    os.makedirs(out_dir, exist_ok=True)
    pbar = tqdm(total=target, desc=label, unit="game",
                position=tqdm_pos, dynamic_ncols=True, leave=True)

    bufs  = [[], [], [], []]  # boards, policies, values, masks
    games = 0
    local = 0  # chunks written this month

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp   = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            stream = io.TextIOWrapper(
                zstd.ZstdDecompressor().stream_reader(resp.raw),
                encoding="utf-8", errors="replace",
            )
            while games < target:
                positions = chess.pgn.read_game(stream, Visitor=_Visitor)
                if positions is None:  # end of file
                    break
                if not positions:      # filtered out
                    continue
                for tensor, move_idx, value_cls, mask in positions:
                    bufs[0].append(tensor)
                    bufs[1].append(move_idx)
                    bufs[2].append(value_cls)
                    bufs[3].append(mask)
                    if len(bufs[0]) >= CHUNK_SIZE:
                        _save_chunk(*bufs, out_dir, chunk_offset + local)
                        local += 1
                        bufs = [[], [], [], []]
                games += 1
                pbar.update(1)
            break  # success

        except Exception as exc:
            pbar.write(f"[{label}] attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                wait = 15 * (2 ** (attempt - 1))
                pbar.write(f"  retrying in {wait}s…")
                time.sleep(wait)

    if bufs[0]:
        _save_chunk(*bufs, out_dir, chunk_offset + local)
        local += 1

    pbar.close()
    open(done_marker, "w").close()
    tqdm.write(f"[{label}] done — {games:,} games → {local} chunks")


def stream_and_process() -> list[str]:
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Target: {TARGET_GAMES:,} games ({GAMES_PER_MONTH:,}/month × {len(LICHESS_MONTHS)} months)")
    print(f"Filter: ≥{MIN_RATING} Elo, ≥{MIN_MOVES} moves — SKIP visitor skips move parsing for filtered games")
    print(f"Output: {PROCESSED_DATA_DIR}  ({CHUNK_SIZE:,} positions/chunk)\n")

    procs = [
        Process(
            target=_worker,
            args=(label, url, PROCESSED_DATA_DIR, GAMES_PER_MONTH,
                  i * SLOTS_PER_MONTH, i),
        )
        for i, (label, url) in enumerate(LICHESS_MONTHS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    chunks = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz")))
    print(f"\nAll done. {len(chunks)} chunks ready for training.")
    return chunks


if __name__ == "__main__":
    stream_and_process()
