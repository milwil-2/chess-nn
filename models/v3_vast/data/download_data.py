"""
v3 pipeline: pgn-extract (C) filter + parallel encode worker pool.

Stage 1 — 12 filter processes (one per Lichess month):
  curl URL | zstdcat | pgn-extract [elo + moves criteria] /dev/stdin
  → qualifying PGN text → game_queue

Stage 2 — N_ENCODE_WORKERS processes:
  game_queue → chess.pgn.read_game → boards_to_tensor/move_to_index → .npz chunks

Resume: existing chunk files are kept by default.  Pass --fresh to wipe them.
The chunk index base is set above the highest existing chunk so new files never
collide with previously written ones.

Stall detection: game_queue.put() uses a timeout (QUEUE_PUT_TIMEOUT seconds).
If encode workers fall so far behind that the queue stays full for longer than
that, the filter worker raises RuntimeError instead of blocking forever.
"""

import argparse
import glob
import io
import multiprocessing
import multiprocessing.queues
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from multiprocessing import Process

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MIN_MOVES,
    MIN_RATING,
    N_ENCODE_WORKERS,
    POLICY_OUTPUT_SIZE,
    PROCESSED_DATA_DIR,
)
from chess_nn.board_encoding import boards_to_tensor
from chess_nn.move_encoding import get_legal_move_indices, move_to_index
from chess_nn.tactics import find_hanging, find_forks


def _url(ym: str) -> tuple[str, str]:
    return ym, f"https://database.lichess.org/standard/lichess_db_standard_rated_{ym}.pgn.zst"


# 12 months across 6 years (same coverage as v2_vast)
LICHESS_MONTHS = [
    _url("2024-10"), _url("2024-04"),
    _url("2023-10"), _url("2023-04"),
    _url("2022-10"), _url("2022-04"),
    _url("2021-10"), _url("2021-04"),
    _url("2020-10"), _url("2020-04"),
    _url("2019-10"), _url("2019-04"),
]

TARGET_GAMES      = 1_000_000
GAMES_PER_MONTH   = TARGET_GAMES // len(LICHESS_MONTHS)   # ~83k per month
CHUNK_SIZE        = 20_000
SLOTS_PER_WORKER  = 500   # chunk index range per encode worker; 500 × 20k = 10M positions max each
MAX_RETRIES       = 5
QUEUE_PUT_TIMEOUT = 120   # seconds: raise instead of blocking if queue stays full this long


def _wdl(result: str, turn: bool) -> int:
    if result == "1-0":
        return 0 if turn == chess.WHITE else 2
    if result == "0-1":
        return 2 if turn == chess.WHITE else 0
    return 1


def _write_criteria_file(min_elo: int) -> str:
    """Write a pgn-extract -t tag criteria file using relational operators.

    pgn-extract supports >=, <=, >, <, =, <> for Elo tags (numeric comparison).
    Character-class glob patterns like [5-9] are NOT supported for Elo — only
    for FEN position matching. Relational operators are the correct approach.
    """
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".pgn-criteria.txt", delete=False)
    f.write(f'WhiteElo >= "{min_elo}"\n')
    f.write(f'BlackElo >= "{min_elo}"\n')
    f.close()
    return f.name


def _iter_game_texts(text_io, max_games: int):
    """Yield complete PGN game strings from a text stream, up to max_games.

    Splits on [Event ...] header lines — the canonical PGN game boundary marker.
    Avoids running chess.pgn.read_game in the filter worker so Python parsing
    cost stays zero for rejected games.
    """
    buf: list[str] = []
    games = 0

    for line in text_io:
        if line.startswith("[Event ") and buf:
            game_text = "".join(buf).strip()
            if game_text:
                yield game_text
                games += 1
                if games >= max_games:
                    return
            buf = [line]
        else:
            buf.append(line)

    if buf and games < max_games:
        game_text = "".join(buf).strip()
        if game_text:
            yield game_text


def encode_game(game: chess.pgn.Game) -> list[tuple]:
    """Encode all positions from a qualifying game into (tensor, policy, value, mask, tactical) tuples."""
    result = game.headers.get("Result", "*")
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return []

    # Safety net: pgn-extract glob patterns may pass games with non-standard
    # ELO tags (e.g. "?" or blank). Catch them here at negligible cost since
    # pgn-extract already filtered out the vast majority.
    try:
        white_elo = int(game.headers.get("WhiteElo", "0") or "0")
        black_elo = int(game.headers.get("BlackElo", "0") or "0")
    except ValueError:
        return []
    if white_elo < MIN_RATING or black_elo < MIN_RATING:
        return []

    hist = deque(maxlen=8)
    positions = []
    board = game.board()

    for move in game.mainline_moves():
        hist.appendleft(board.copy(stack=False))
        tensor   = boards_to_tensor(list(hist))
        move_idx = move_to_index(move, board)
        mask     = np.zeros(POLICY_OUTPUT_SIZE, dtype=bool)
        for idx in get_legal_move_indices(board):
            mask[idx] = True
        is_tactical = bool(find_hanging(board) or find_forks(board))
        positions.append((tensor, move_idx, _wdl(result, board.turn), mask, is_tactical))
        board.push(move)

    if len(positions) < MIN_MOVES:
        return []
    return positions


def _save_chunk(boards, policies, values, masks, tacticals, out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"chunk_{idx:04d}.npz")
    np.savez_compressed(
        path,
        boards      = np.array(boards,     dtype=np.float32),
        policies    = np.array(policies,   dtype=np.int64),
        values      = np.array(values,     dtype=np.int64),
        legal_masks = np.packbits(np.array(masks, dtype=bool), axis=1),
        tactical    = np.array(tacticals,  dtype=np.uint8),
    )


def _filter_worker(label: str, url: str, game_queue: multiprocessing.Queue,
                   tqdm_pos: int) -> None:
    """Download one Lichess month, filter via pgn-extract, put game PGN text into game_queue."""
    criteria_file = _write_criteria_file(MIN_RATING)
    # -pl counts half-moves (plies), matching Python's ply-based MIN_MOVES threshold.
    # Ubuntu 19.04 package uses old-style -pl flag; --minply only exists in 25.x+.
    extract_flags = f"-t {criteria_file} -pl{MIN_MOVES}"

    # /dev/stdin lets pgn-extract read from the pipe on Linux
    cmd = (
        f"curl -s --retry 5 '{url}' "
        f"| zstdcat "
        f"| pgn-extract {extract_flags} /dev/stdin"
    )

    pbar = tqdm(total=GAMES_PER_MONTH, desc=f"[{label}] filter",
                unit="game", position=tqdm_pos, dynamic_ncols=True, leave=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            text_io = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
            for game_text in _iter_game_texts(text_io, GAMES_PER_MONTH):
                try:
                    game_queue.put(game_text, timeout=QUEUE_PUT_TIMEOUT)
                except queue.Full:
                    raise RuntimeError(
                        f"[{label}] queue full for {QUEUE_PUT_TIMEOUT}s — "
                        f"encode workers are overloaded (N_ENCODE_WORKERS too low?)"
                    )
                pbar.update(1)
            proc.wait()
            break
        except Exception as exc:
            pbar.write(f"[{label}] attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                wait = 15 * (2 ** (attempt - 1))
                pbar.write(f"  retrying in {wait}s…")
                time.sleep(wait)

    pbar.close()
    if criteria_file and os.path.exists(criteria_file):
        os.unlink(criteria_file)


def _encode_worker(worker_id: int, game_queue: multiprocessing.Queue,
                   out_dir: str, chunk_start: int, tqdm_pos: int) -> None:
    """Consume game PGN strings from game_queue, encode positions, write .npz chunks."""
    pbar = tqdm(desc=f"encode-{worker_id:02d}", unit="game",
                position=tqdm_pos, dynamic_ncols=True, leave=True)

    bufs        = [[], [], [], [], []]   # boards, policies, values, masks, tacticals
    local_chunk = 0

    while True:
        item = game_queue.get()
        if item is None:
            break
        game = chess.pgn.read_game(io.StringIO(item))
        if game is None:
            continue
        positions = encode_game(game)
        for tensor, move_idx, value_cls, mask, is_tactical in positions:
            bufs[0].append(tensor)
            bufs[1].append(move_idx)
            bufs[2].append(value_cls)
            bufs[3].append(mask)
            bufs[4].append(is_tactical)
            if len(bufs[0]) >= CHUNK_SIZE:
                _save_chunk(*bufs, out_dir, chunk_start + local_chunk)
                local_chunk += 1
                bufs = [[], [], [], [], []]
        pbar.update(1)

    if bufs[0]:
        _save_chunk(*bufs, out_dir, chunk_start + local_chunk)
        local_chunk += 1

    pbar.close()
    tqdm.write(f"[encode-{worker_id:02d}] done — {local_chunk} chunks written")


def stream_and_process(fresh: bool = False) -> list[str]:
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz")))

    if fresh:
        for f in existing:
            os.remove(f)
        existing = []
        chunk_id_base = 0
        print("--fresh: cleared existing chunks.")
    elif existing:
        max_id = max(
            int(os.path.basename(f).replace("chunk_", "").replace(".npz", ""))
            for f in existing
        )
        # Start new chunk IDs one full SLOTS_PER_WORKER block above the highest existing one
        chunk_id_base = ((max_id // SLOTS_PER_WORKER) + 1) * SLOTS_PER_WORKER
        print(f"Resuming: {len(existing)} existing chunks found, new IDs start at {chunk_id_base}.")
    else:
        chunk_id_base = 0

    n_filter = len(LICHESS_MONTHS)
    print(f"Target: {TARGET_GAMES:,} games ({GAMES_PER_MONTH:,}/month × {n_filter} months)")
    print(f"Filter: ≥{MIN_RATING} Elo, ≥{MIN_MOVES} moves — pgn-extract C pipeline")
    print(f"Encode: {N_ENCODE_WORKERS} workers  Output: {PROCESSED_DATA_DIR}  ({CHUNK_SIZE:,} pos/chunk)\n")

    game_queue = multiprocessing.Queue(maxsize=2000)

    # tqdm positions: encode workers 0..N-1, filter workers N..N+11
    encode_procs = [
        Process(target=_encode_worker,
                args=(i, game_queue, PROCESSED_DATA_DIR, chunk_id_base + i * SLOTS_PER_WORKER, i))
        for i in range(N_ENCODE_WORKERS)
    ]
    filter_procs = [
        Process(target=_filter_worker,
                args=(label, url, game_queue, N_ENCODE_WORKERS + i))
        for i, (label, url) in enumerate(LICHESS_MONTHS)
    ]

    for p in encode_procs + filter_procs:
        p.start()

    # Background thread: once all filter workers finish, send one shutdown
    # sentinel (None) per encode worker so each gets exactly one stop signal.
    failed_filters: list[str] = []

    def _wait_and_shutdown():
        for p in filter_procs:
            p.join()
            if p.exitcode != 0:
                failed_filters.append(p.name)
                tqdm.write(f"[sentinel] WARNING: {p.name} exited with code {p.exitcode}")
        for _ in encode_procs:
            game_queue.put(None)

    sentinel_thread = threading.Thread(target=_wait_and_shutdown, daemon=True)
    sentinel_thread.start()

    for p in encode_procs:
        p.join()
    sentinel_thread.join()

    chunks = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "chunk_*.npz")))

    if failed_filters:
        raise RuntimeError(
            f"{len(failed_filters)} filter worker(s) failed: {failed_filters}. "
            f"Only {len(chunks)} chunks written. Check logs above for details."
        )

    print(f"\nAll done. {len(chunks)} chunks ready for training.")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and encode Lichess games.")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete all existing processed chunks before starting (default: resume)."
    )
    args = parser.parse_args()
    stream_and_process(fresh=args.fresh)
