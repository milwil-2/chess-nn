"""
Download and filter games from the Lichess open database.

Lichess publishes every rated game ever played, freely available at:
  https://database.lichess.org/

Files are compressed with zstandard (.zst). We stream-decompress on the fly
so we never need to store the full uncompressed file (they're huge — GB range).

We filter for quality: both players 2000+ rated, standard rules, 10+ moves.
Target: ~100,000 games → ~3-5 million board positions for training.
"""

import os
import sys
import requests
import zstandard as zstd
import chess.pgn
import io
from tqdm import tqdm

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, MIN_RATING, MIN_MOVES

# Lichess database URL — January 2024 standard rated games
# Change the date suffix to get a different month
LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "filtered_games.pgn")
TARGET_GAMES = 20_000


def _clear_old_data():
    import shutil
    from config import PROCESSED_DATA_DIR

    cleared = []

    # Raw PGN
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        cleared.append(OUTPUT_FILE)

    # Processed .npz chunks
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
        cleared.append(PROCESSED_DATA_DIR)

    if cleared:
        print("Cleared old data:")
        for p in cleared:
            print(f"  {p}")
        print()


def download_and_filter():
    _clear_old_data()
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"Downloading from: {LICHESS_URL}")
    print(f"Target: {TARGET_GAMES:,} games with both players rated {MIN_RATING}+")
    print("This streams the file — no need to download the full archive first.\n")

    response = requests.get(LICHESS_URL, stream=True)
    response.raise_for_status()

    # zstd streaming decompressor
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)  # type: ignore[arg-type]
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")

    games_written = 0
    games_seen = 0

    with open(OUTPUT_FILE, "w") as out_file:
        pbar = tqdm(total=TARGET_GAMES, desc="Games collected", unit="game")

        while games_written < TARGET_GAMES:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                print("\nReached end of database file.")
                break

            games_seen += 1

            # Filter: ratings
            white_elo = int(game.headers.get("WhiteElo", "0") or "0")
            black_elo = int(game.headers.get("BlackElo", "0") or "0")
            if white_elo < MIN_RATING or black_elo < MIN_RATING:
                continue

            # Filter: game must have a result (not abandoned)
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            # Filter: enough moves
            moves = list(game.mainline_moves())
            if len(moves) < MIN_MOVES:
                continue

            # Write to output PGN
            print(game, file=out_file, end="\n\n")
            games_written += 1
            pbar.update(1)

        pbar.close()

    print(f"\nDone. Seen {games_seen:,} games, kept {games_written:,}.")
    print(f"Saved to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    download_and_filter()
