"""
UCI adapter for the chess-nn MCTS engine.

Speaks the full UCI protocol so the engine can be driven by python-chess's
SimpleEngine (and any other UCI-conforming GUI / harness).

Wire-up: `run.py engine` calls `run(model, sims, fast)` here instead of the
old stub protocol.

Known limitation: `stop` is a no-op — the underlying MCTS loop is synchronous
and single-threaded, so we can't preempt it. Current search finishes; bestmove
is emitted on its own schedule. Time controls are still honored via
`MCTS.search_time_budget(...)`.
"""

import sys
import collections
from typing import Optional

import chess

from chess_nn.board_encoding import HISTORY_LENGTH


def _emit(line: str) -> None:
    print(line, flush=True)


def _parse_position(tokens: list[str]) -> tuple[chess.Board, collections.deque]:
    """Parse a `position ...` command's tokens (without the leading 'position').

    Returns (board, history_deque) where history_deque[0] is the current board
    and earlier entries are prior plies (most-recent-first).
    """
    history: collections.deque = collections.deque(maxlen=HISTORY_LENGTH)

    if not tokens:
        board = chess.Board()
        history.appendleft(board.copy(stack=False))
        return board, history

    idx = 0
    if tokens[0] == "startpos":
        board = chess.Board()
        idx = 1
    elif tokens[0] == "fen":
        # FEN is 6 tokens: piece-placement side castling ep halfmove fullmove
        fen_parts = tokens[1:7]
        board = chess.Board(" ".join(fen_parts))
        idx = 7
    else:
        # Unknown — default to startpos.
        board = chess.Board()

    history.appendleft(board.copy(stack=False))

    if idx < len(tokens) and tokens[idx] == "moves":
        for uci in tokens[idx + 1:]:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move in board.legal_moves:
                board.push(move)
                history.appendleft(board.copy(stack=False))

    return board, history


def _compute_time_budget_ms(tokens: list[str], board: chess.Board) -> Optional[int]:
    """Parse a `go ...` command and return the time budget in milliseconds.

    Returns None if no time-related token was found (caller should use a
    simulation-count fallback in that case).
    """
    if not tokens:
        return None

    args: dict[str, int] = {}
    flags: set[str] = set()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("wtime", "btime", "winc", "binc", "movetime", "depth",
                   "nodes", "movestogo"):
            if i + 1 < len(tokens):
                try:
                    args[tok] = int(tokens[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
        elif tok in ("infinite", "ponder"):
            flags.add(tok)
        i += 1

    if "infinite" in flags:
        return 60_000

    if "movetime" in args:
        return max(50, args["movetime"] - 50)

    side = board.turn
    has_time = ("wtime" in args) if side == chess.WHITE else ("btime" in args)
    if has_time:
        if side == chess.WHITE:
            time_left = args.get("wtime", 0)
            inc = args.get("winc", 0)
        else:
            time_left = args.get("btime", 0)
            inc = args.get("binc", 0)

        move_num = board.fullmove_number
        divisor = max(40 - move_num, 10)
        budget = int(time_left / divisor + inc // 2)
        # 30% safety: never spend more than 40% of remaining time on one move.
        cap = int(time_left * 0.4)
        budget = min(budget, cap)
        return max(50, budget)

    if "depth" in args or "nodes" in args:
        return None  # caller falls back to fixed sim budget

    return None


def _fallback_sim_count(tokens: list[str], default_sims: int) -> int:
    """Parse depth/nodes for fixed-sim fallback."""
    sims = default_sims
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "nodes" and i + 1 < len(tokens):
            try:
                sims = min(int(tokens[i + 1]), 2000)
            except ValueError:
                pass
            i += 2
            continue
        if tok == "depth" and i + 1 < len(tokens):
            try:
                sims = min(int(tokens[i + 1]) * 50, 2000)
            except ValueError:
                pass
            i += 2
            continue
        i += 1
    return sims


def run(model, default_sims: int = 200, fast: bool = False) -> None:
    """Main UCI loop. Reads stdin line-by-line until 'quit'."""
    # Configure helpers exactly the way run.py's old cmd_engine did so MCTS
    # picks up the opening book, Syzygy tablebase, and persistent cache.
    book = None
    tablebase = None
    tcache = None
    mcts = None
    syzygy_path_default = ""
    book_path_default = ""

    if not fast:
        from chess_nn.mcts import MCTS
        from chess_nn.opening_book import OpeningBook
        from chess_nn.syzygy_probe import SyzygyTable
        from chess_nn.transposition import TranspositionCache
        from config import OPENING_BOOK_PATH, SYZYGY_PATH, MCTS_CACHE_PATH

        syzygy_path_default = SYZYGY_PATH
        book_path_default = OPENING_BOOK_PATH

        book = OpeningBook(polyglot_path=OPENING_BOOK_PATH)
        tablebase = SyzygyTable(path=SYZYGY_PATH)
        tcache = TranspositionCache(path=MCTS_CACHE_PATH)
        mcts = MCTS(model, num_simulations=default_sims,
                    book=book, tablebase=tablebase, tcache=tcache)
    else:
        from chess_nn.evaluate import select_move
        _select_move_fast = select_move

    board = chess.Board()
    history: collections.deque = collections.deque(maxlen=HISTORY_LENGTH)
    history.appendleft(board.copy(stack=False))

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            _emit("id name Chess-NN v3_vast")
            _emit("id author milwil-2")
            _emit(f"option name SyzygyPath type string default {syzygy_path_default}")
            _emit(f"option name OpeningBookPath type string default {book_path_default}")
            _emit("option name Hash type spin default 64 min 1 max 4096")
            _emit("uciok")

        elif cmd == "isready":
            _emit("readyok")

        elif cmd == "setoption":
            # Format: setoption name <NAME> value <VALUE>
            # We accept and silently ignore most options. SyzygyPath /
            # OpeningBookPath rebuild the helper if changed.
            try:
                name_idx = tokens.index("name")
                value_idx = tokens.index("value")
                opt_name = " ".join(tokens[name_idx + 1:value_idx])
                opt_value = " ".join(tokens[value_idx + 1:])
            except ValueError:
                continue
            if mcts is None:
                continue
            if opt_name == "SyzygyPath" and opt_value:
                try:
                    from chess_nn.syzygy_probe import SyzygyTable
                    mcts.tablebase = SyzygyTable(path=opt_value)
                except Exception as exc:
                    print(f"info string syzygy reload failed: {exc}", file=sys.stderr)
            elif opt_name == "OpeningBookPath" and opt_value:
                try:
                    from chess_nn.opening_book import OpeningBook
                    mcts.book = OpeningBook(polyglot_path=opt_value)
                except Exception as exc:
                    print(f"info string book reload failed: {exc}", file=sys.stderr)
            # Hash etc. — no-op.

        elif cmd == "ucinewgame":
            if mcts is not None:
                mcts.reset()
            board = chess.Board()
            history.clear()
            history.appendleft(board.copy(stack=False))

        elif cmd == "position":
            new_board, new_history = _parse_position(tokens[1:])
            # If the new board diverges from the previous (different start, jump,
            # or it's the opening), reset MCTS subtree reuse. Subtree reuse needs
            # the new board to be exactly one push past where we left off.
            if mcts is not None:
                # Cheap heuristic: if the new history is short or the move stack
                # doesn't extend the previous one, drop the subtree.
                if len(new_board.move_stack) <= 1:
                    mcts.reset()
            board = new_board
            history = new_history

        elif cmd == "go":
            board_history_list = list(history)
            if fast:
                move = _select_move_fast(model, board, temperature=0)
            else:
                budget_ms = _compute_time_budget_ms(tokens[1:], board)
                if budget_ms is not None:
                    move = mcts.search_time_budget(
                        board, board_history=board_history_list,
                        time_ms=budget_ms, temperature=0.0,
                        enable_blunder_filter=True,
                    )
                else:
                    sims = _fallback_sim_count(tokens[1:], default_sims)
                    saved = mcts.num_simulations
                    mcts.num_simulations = sims
                    try:
                        move = mcts.search(
                            board, board_history=board_history_list,
                            temperature=0.0, enable_blunder_filter=True,
                        )
                    finally:
                        mcts.num_simulations = saved
            _emit(f"bestmove {move.uci()}")

        elif cmd == "stop":
            # No-op: MCTS is synchronous; the in-flight search (if any) has
            # already completed by the time we see this line. The next `go`
            # will produce a new bestmove. Documented limitation.
            continue

        elif cmd == "quit":
            if mcts is not None and mcts.tcache is not None:
                try:
                    mcts.tcache.save()
                except Exception:
                    pass
            break

        # Unknown commands are ignored per UCI spec.
