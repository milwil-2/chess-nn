"""
Monte Carlo Tree Search (MCTS) for chess.

The tree is made of Nodes. Each Node represents one board position.
We grow the tree by repeatedly running 4 steps:
  1. Select   — walk down using UCB formula until we hit an unvisited node
  2. Expand   — ask the neural network what it thinks of this new position
  3. Backup   — send the value back up to all ancestors
  4. (repeat N times, then pick the most-visited move)
"""

import math
import time
import chess
import torch
import torch.nn.functional as F
import numpy as np

from chess_nn.board_encoding import boards_to_tensor
from chess_nn.move_encoding import move_to_index, index_to_move, get_legal_move_indices
from chess_nn.model import wdl_to_scalar

# Exploration constant: higher = more exploration, lower = more exploitation.
# AlphaZero uses ~1.4. Think of it as "how curious is the searcher?"
C_PUCT = 1.4


class Node:
    """
    One node in the search tree = one chess position.

    Each node tracks:
      - How many times we've visited it (N)
      - The total value accumulated from all visits (W)
      - The average value Q = W / N
      - The prior probability P from the neural network (how likely this move looked)
      - Its children (one per legal move)
    """

    def __init__(self, prior: float = 0.0):
        self.N = 0        # Visit count
        self.W = 0.0      # Total value (sum of backpropagated values)
        self.P = prior    # Prior probability from network policy head

        # Dict of {chess.Move: Node} — populated lazily when this node is expanded
        self.children: dict[chess.Move, "Node"] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        """Average value. 0 if never visited (optimistic for unexplored nodes)."""
        return self.W / self.N if self.N > 0 else 0.0

    def ucb_score(self, parent_visits: int) -> float:
        """
        Upper Confidence Bound formula:
          Q  = exploitation: average reward seen from this node
          U  = exploration bonus: high when P is large or visits are low

        The sqrt term ensures every node gets visited eventually,
        but nodes with high prior P or high Q get visited more.
        """
        U = C_PUCT * self.P * math.sqrt(parent_visits) / (1 + self.N)
        return self.Q + U

    def best_child(self) -> tuple[chess.Move, "Node"]:
        """Pick the child with the highest UCB score."""
        return max(self.children.items(), key=lambda kv: kv[1].ucb_score(self.N))

    def most_visited_child(self) -> tuple[chess.Move, "Node"]:
        """After search is done, pick the move with the most visits (most reliable)."""
        return max(self.children.items(), key=lambda kv: kv[1].N)

    def visit_distribution(self, temperature: float = 1.0) -> dict[chess.Move, float]:
        """
        Turn visit counts into a probability distribution over moves.

        temperature=1: proportional to visits (used early in game — more variety)
        temperature→0: nearly deterministic, picks the most-visited move
        (temperature is applied as visit_count^(1/temp) before normalising)
        """
        if not self.children:
            return {}
        visits = {m: n.N for m, n in self.children.items()}
        if temperature == 0:
            best = max(visits, key=visits.get)
            return {m: (1.0 if m == best else 0.0) for m in visits}

        # Apply temperature
        powered = {m: v ** (1.0 / temperature) for m, v in visits.items()}
        total = sum(powered.values())
        if total <= 0:
            # I13: search may have been stopped before any sims completed, so
            # every child has N=0. Return a uniform distribution rather than
            # dividing by zero — the caller (search()) will pick a move.
            n = len(powered)
            return {m: 1.0 / n for m in powered}
        return {m: v / total for m, v in powered.items()}


class MCTS:
    """
    The search engine. Owns the root node and runs simulations.

    Usage:
        mcts = MCTS(model, num_simulations=400)
        move = mcts.search(board)
    """

    def __init__(self, model, num_simulations: int = 200,
                 book=None, tablebase=None, tcache=None):
        """
        book      : Optional[OpeningBook]      — consulted before search; on hit, MCTS is skipped.
        tablebase : Optional[SyzygyTable]      — overrides value head + masks priors to optimal moves for <=6-piece positions.
        tcache    : Optional[TranspositionCache] — cross-game visit-count cache; seeds priors on repeated positions.
        All three are inference-only — self-play should pass them as None.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.book = book
        self.tablebase = tablebase
        self.tcache = tcache
        # Subtree of the position after our last chosen move. Used to carry over
        # search work across consecutive search() calls — see _try_reuse_root().
        self._next_root: "Node | None" = None
        # I13: cooperative cancellation flag for UCI `stop`. Checked once per
        # simulation; set via stop() from another thread (or before search()).
        # Reset at the end of every search so subsequent calls start clean.
        self._stop_requested: bool = False

    def reset(self) -> None:
        """Drop any carried-over search tree. Call on game reset or arbitrary
        position jumps where the carried subtree no longer matches the board."""
        self._next_root = None

    def stop(self) -> None:
        """I13: request the currently-running search to abort at the next
        simulation boundary. Safe to call from another thread — assignment to
        a Python bool is atomic. The flag is cleared automatically when the
        search returns, so the next search() / search_time_budget() call is
        not pre-cancelled."""
        self._stop_requested = True

    def _check_stop(self) -> bool:
        """Test hook for I13. Returns True if a stop has been requested."""
        return self._stop_requested

    def _book_move(self, board: chess.Board) -> "chess.Move | None":
        """Consult opening book. Returns the book move (and resets subtree
        reuse, since we won't have a tree under it). Returns None on miss."""
        if self.book is None:
            return None
        m = self.book.lookup(board)
        if m is None:
            return None
        # No MCTS ran → no subtree to carry over.
        self._next_root = None
        return m

    def _seed_priors_from_cache(self, board: chess.Board, root: Node,
                                cache_weight: float = 0.5) -> None:
        """If the transposition cache has prior visit counts for this position,
        blend them into the root's priors. Cached visits become a secondary
        prior signal alongside the network's policy output. No-op on miss."""
        if self.tcache is None:
            return
        cached = self.tcache.lookup(board)
        if not cached:
            return
        total = sum(cached.values())
        if total <= 0:
            return
        cw = max(0.0, min(1.0, cache_weight))
        for move, child in root.children.items():
            if move in cached:
                frac = cached[move] / total
                child.P = (1 - cw) * child.P + cw * frac
        # Renormalize so the priors still sum to ~1.
        s = sum(c.P for c in root.children.values())
        if s > 0:
            for c in root.children.values():
                c.P = c.P / s

    def _record_to_cache(self, board: chess.Board, root: Node) -> None:
        if self.tcache is None:
            return
        self.tcache.record(board, {m: c.N for m, c in root.children.items()})

    def _root_temperature(self, board: chess.Board) -> float:
        """Compute the ply-aware root softmax temperature (I8 + I2).

        Linearly interpolates between ROOT_POLICY_TEMPERATURE (opening)
        and ROOT_POLICY_TEMPERATURE_END (after ROOT_POLICY_TEMPERATURE_ANNEAL_PLY
        plies). Returns 1.0 (no-op) if config import fails."""
        try:
            from config import ROOT_POLICY_TEMPERATURE  # type: ignore
            T_start = ROOT_POLICY_TEMPERATURE
        except Exception:
            return 1.0
        try:
            from config import ROOT_POLICY_TEMPERATURE_END  # type: ignore
            T_end = ROOT_POLICY_TEMPERATURE_END
        except Exception:
            T_end = T_start
        try:
            from config import ROOT_POLICY_TEMPERATURE_ANNEAL_PLY  # type: ignore
            anneal_ply = ROOT_POLICY_TEMPERATURE_ANNEAL_PLY
        except Exception:
            anneal_ply = 40
        if anneal_ply <= 0 or T_start == T_end:
            return T_start
        # fullmove_number is 1-indexed; ply ≈ 2*(fullmove-1) + (1 if Black to move else 0)
        ply = 2 * (board.fullmove_number - 1) + (0 if board.turn else 1)
        alpha = min(1.0, ply / float(anneal_ply))
        return T_start + (T_end - T_start) * alpha

    def _retemperature_root(self, root: Node, board: chess.Board) -> None:
        """Re-apply the ply-aware root temperature to an already-expanded root's
        child priors. Used when subtree reuse hands us a root whose priors were
        computed deeper in the tree (without root smoothing). Inverts to
        logits, divides by T, re-softmaxes, so we don't double-smooth."""
        T = self._root_temperature(board)
        if T == 1.0 or not root.children:
            return
        children = list(root.children.values())
        priors = np.array([c.P for c in children], dtype=np.float64)
        priors = np.clip(priors, 1e-12, None)
        logits = np.log(priors) / T
        logits -= logits.max()
        new_p = np.exp(logits)
        new_p /= new_p.sum()
        for c, p in zip(children, new_p):
            c.P = float(p)

    def _filter_root_blunders(self, board: chess.Board, root: Node) -> None:
        """B2: remove root children whose resulting position hangs material
        above a pawn, unless the candidate move itself is a capture or check.
        Inference-only — self-play passes enable_blunder_filter=False so
        training data still includes blunders the network needs to learn from.
        Falls back to original children if filtering would empty root."""
        from chess_nn.tactics import find_hanging, PIECE_VALUE

        if not root.children:
            return
        survivors: dict[chess.Move, "Node"] = {}
        for move, child in root.children.items():
            is_forcing = board.is_capture(move) or board.gives_check(move)
            if is_forcing:
                survivors[move] = child
                continue
            scratch = board.copy(stack=False)
            scratch.push(move)
            # find_hanging skips kings internally; check pieces of the side
            # that just moved (now under attack and undefended).
            moved_color = board.turn
            hangs_big = False
            for tac in find_hanging(scratch):
                sq = tac.squares[0]
                piece = scratch.piece_at(sq)
                if piece is None or piece.color != moved_color:
                    continue
                if PIECE_VALUE[piece.piece_type] > 1:  # > pawn
                    hangs_big = True
                    break
            if not hangs_big:
                survivors[move] = child
        if not survivors:
            import sys
            print("[mcts] blunder filter would empty root; falling back to unfiltered moves",
                  file=sys.stderr)
            return
        root.children = survivors

    def _try_reuse_root(self, board: chess.Board) -> "Node | None":
        """If the previous search stored a subtree rooted at the position after
        our move, descend into the child corresponding to the opponent's most
        recent move and reuse it. Returns None if no reusable subtree exists."""
        if self._next_root is None or not board.move_stack:
            return None
        opponent_move = board.peek()
        candidate = self._next_root.children.get(opponent_move)
        if candidate is None or not candidate.is_expanded:
            return None
        return candidate

    def search(self, board: chess.Board, board_history: list = None,
               temperature: float = 1.0, add_noise: bool = False,
               enable_blunder_filter: bool = False) -> chess.Move:
        """
        Run `num_simulations` simulations from `board`, return the chosen move.

        board_history: list of chess.Board where [0] is `board` and [1..] are prior
        positions (most recent first).  Padded with zeros if fewer than 8 are given.
        add_noise=True mixes Dirichlet noise into root priors (used during self-play).
        enable_blunder_filter=True drops obvious-hanging-piece moves at the root
        before search (inference only; self-play stays False so training data
        still contains the blunders the network must learn to avoid).
        """
        # Opening book short-circuits MCTS for known positions.
        book_move = self._book_move(board)
        if book_move is not None:
            return book_move

        if board_history is None:
            board_history = [board]

        # Order at root: expand (with B1 temperature) → re-temperature if reused
        # → cache seed → blunder filter (B2) → Dirichlet noise → simulate.
        root = self._try_reuse_root(board)
        fresh_root = root is None
        if fresh_root:
            root = Node(prior=1.0)
            self._expand(root, board_history, is_root=True)
        else:
            self._retemperature_root(root, board)
        if fresh_root:
            self._seed_priors_from_cache(board, root)
        if enable_blunder_filter:
            self._filter_root_blunders(board, root)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            # I13: cooperative stop check between simulations.
            if self._stop_requested:
                break
            node = root
            # stack=False: skip copying move history — MCTS never undoes moves, saves ~80× _BoardState allocs per sim
            scratch_board = board.copy(stack=False)
            scratch_history = list(board_history)
            path = [node]

            # --- Step 1: Selection ---
            while node.is_expanded and not scratch_board.is_game_over():
                move, node = node.best_child()
                scratch_board.push(move)
                scratch_history = [scratch_board.copy(stack=False)] + scratch_history[:7]
                path.append(node)

            # --- Step 2: Expansion + Evaluation ---
            if scratch_board.is_game_over():
                result = scratch_board.result()
                value = self._terminal_value(result, scratch_board.turn)
            else:
                value = self._expand(node, scratch_history)

            # --- Step 3: Backup ---
            for i, visited_node in enumerate(reversed(path)):
                visited_node.N += 1
                visited_node.W += value if i % 2 == 0 else -value

        self._record_to_cache(board, root)

        dist = root.visit_distribution(temperature=temperature)
        # I13: clear the stop flag so the next search() starts unaborted.
        self._stop_requested = False
        if not dist:
            legal = list(board.legal_moves)
            return legal[0] if legal else chess.Move.null()
        moves = list(dist.keys())
        probs = [dist[m] for m in moves]
        chosen = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen]
        # Keep the subtree under the move we just chose. Next search() call will
        # descend one more ply (opponent's response) before reusing it.
        self._next_root = root.children.get(chosen_move)
        return chosen_move

    def search_time_budget(self, board: chess.Board, board_history: list = None,
                           time_ms: float = 1000.0, temperature: float = 0.0,
                           enable_blunder_filter: bool = False) -> chess.Move:
        """
        Run simulations until `time_ms` wall-clock elapses, then pick the
        chosen move (most-visited at temperature 0, sampled otherwise).

        Same code path as search() but the iteration count is bounded by wall
        time rather than self.num_simulations. Subtree reuse / book / Syzygy /
        cache / B1 / B2 all behave identically.
        """
        # Opening book short-circuits MCTS for known positions.
        book_move = self._book_move(board)
        if book_move is not None:
            return book_move

        if board_history is None:
            board_history = [board]

        root = self._try_reuse_root(board)
        fresh_root = root is None
        if fresh_root:
            root = Node(prior=1.0)
            self._expand(root, board_history, is_root=True)
        else:
            self._retemperature_root(root, board)
        if fresh_root:
            self._seed_priors_from_cache(board, root)
        if enable_blunder_filter:
            self._filter_root_blunders(board, root)

        start = time.monotonic()
        deadline_s = time_ms / 1000.0
        # Run at least one simulation so we always have visit counts.
        sims_done = 0
        while True:
            node = root
            scratch_board = board.copy(stack=False)
            scratch_history = list(board_history)
            path = [node]

            while node.is_expanded and not scratch_board.is_game_over():
                move, node = node.best_child()
                scratch_board.push(move)
                scratch_history = [scratch_board.copy(stack=False)] + scratch_history[:7]
                path.append(node)

            if scratch_board.is_game_over():
                result = scratch_board.result()
                value = self._terminal_value(result, scratch_board.turn)
            else:
                value = self._expand(node, scratch_history)

            for i, visited_node in enumerate(reversed(path)):
                visited_node.N += 1
                visited_node.W += value if i % 2 == 0 else -value

            sims_done += 1
            # I13: cooperative stop check between simulations.
            if self._stop_requested:
                break
            if (time.monotonic() - start) >= deadline_s:
                break

        self._record_to_cache(board, root)

        # I13: clear the stop flag so the next search starts unaborted.
        self._stop_requested = False
        dist = root.visit_distribution(temperature=temperature)
        if not dist:
            # Degenerate: no legal children. Return any legal move if possible.
            legal = list(board.legal_moves)
            return legal[0] if legal else chess.Move.null()
        moves = list(dist.keys())
        probs = [dist[m] for m in moves]
        if temperature == 0:
            chosen_move = max(zip(moves, probs), key=lambda mp: mp[1])[0]
        else:
            chosen = np.random.choice(len(moves), p=probs)
            chosen_move = moves[chosen]
        self._next_root = root.children.get(chosen_move)
        return chosen_move

    def get_policy(self, board: chess.Board, board_history: list = None,
                   temperature: float = 1.0, add_noise: bool = True,
                   enable_blunder_filter: bool = False) -> dict:
        """
        Run search and return the full visit distribution (used for training targets).
        This is the 'improved policy' that MCTS produces — better than the raw network output.
        add_noise=True by default during self-play to prevent repetitive draws.

        Self-play wires `book=None, tablebase=None, tcache=None,
        enable_blunder_filter=False` so this path is unaffected by the
        inference-only helpers; training keeps seeing the raw network signal.
        """
        if board_history is None:
            board_history = [board]

        # Same ordering as search() — get_policy always builds a fresh root, so
        # no reuse / re-temperature branch needed.
        root = Node(prior=1.0)
        self._expand(root, board_history, is_root=True)
        self._seed_priors_from_cache(board, root)
        if enable_blunder_filter:
            self._filter_root_blunders(board, root)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            # I13: cooperative stop check between simulations.
            if self._stop_requested:
                break
            node = root
            scratch_board = board.copy(stack=False)
            scratch_history = list(board_history)
            path = [node]

            while node.is_expanded and not scratch_board.is_game_over():
                move, node = node.best_child()
                scratch_board.push(move)
                scratch_history = [scratch_board.copy(stack=False)] + scratch_history[:7]
                path.append(node)

            if scratch_board.is_game_over():
                value = self._terminal_value(scratch_board.result(), scratch_board.turn)
            else:
                value = self._expand(node, scratch_history)

            for i, visited_node in enumerate(reversed(path)):
                visited_node.N += 1
                visited_node.W += value if i % 2 == 0 else -value

        self._record_to_cache(board, root)
        # I13: clear the stop flag so the next search starts unaborted.
        self._stop_requested = False
        return root.visit_distribution(temperature=temperature)

    def _add_dirichlet_noise(self, root: Node,
                             alpha: float = 0.3, epsilon: float = 0.35,
                             shape_floor: float | None = None) -> None:
        """Mix Dirichlet noise into root priors — AlphaZero's exploration trick.
        Without this, self-play games follow the same lines every time → repetition draws.

        Shaped variant (KataGo): when shape_floor is given, zero out noise on
        children whose prior is below mean(P) * shape_floor. This concentrates
        exploration on plausible moves and stops self-play from polluting
        training data with low-prior junk (e.g. early king moves)."""
        children = list(root.children.values())
        if not children:
            return
        if shape_floor is None:
            try:
                from config import DIRICHLET_SHAPE_FLOOR  # type: ignore
                shape_floor = DIRICHLET_SHAPE_FLOOR
            except Exception:
                shape_floor = 0.0

        noise = np.random.dirichlet([alpha] * len(children))

        if shape_floor > 0.0:
            priors = np.array([c.P for c in children], dtype=np.float64)
            threshold = priors.mean() * shape_floor
            mask = (priors >= threshold).astype(np.float64)
            shaped = noise * mask
            total = shaped.sum()
            if total > 0:
                noise = shaped / total
            # else: every prior is below threshold (degenerate) — fall back to flat noise

        for child, n in zip(children, noise):
            child.P = (1 - epsilon) * child.P + epsilon * n

    def _expand(self, node: Node, board_history: list, is_root: bool = False) -> float:
        """
        Run the neural network on this position.
        Populate the node's children with prior probabilities from the policy head.
        Returns the value estimate from the value head.

        Syzygy override: if a tablebase is loaded and the position is
        small enough (<= 6 pieces, no castling rights), we replace the
        network's value with the true WDL and restrict the priors to the
        set of moves that preserve the best achievable outcome. This
        bypasses the network's measured side-to-move bias for endgames.

        Root softmax temperature (B1): when is_root=True, divide policy logits
        by ROOT_POLICY_TEMPERATURE before softmax to flatten the priors.
        This makes MCTS willing to search non-top-1 moves and reduces the
        "38% out-of-top-8 picks" symptom.
        """
        board = board_history[0]
        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(boards_to_tensor(board_history)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)

        # Get legal move indices and mask out illegal moves
        legal_indices = get_legal_move_indices(board)
        policy = policy_logits.squeeze(0).cpu().numpy()

        # Softmax over legal moves only (illegal = -inf)
        legal_logits = np.array([policy[i] for i in legal_indices])
        legal_logits -= legal_logits.max()  # Numerical stability
        if is_root:
            T = self._root_temperature(board)
            if T != 1.0:
                legal_logits = legal_logits / T
        priors = np.exp(legal_logits)
        priors /= priors.sum()

        # Create a child node for each legal move with its prior probability
        for idx, move_idx in enumerate(legal_indices):
            move = index_to_move(move_idx, board)
            if move in board.legal_moves:
                node.children[move] = Node(prior=float(priors[idx]))

        node.is_expanded = True

        if self.tablebase is not None:
            tb_value = self.tablebase.value_scalar(board)
            if tb_value is not None:
                # I7: use DTZ tie-breaking so won endgames actually convert
                # (instead of shuffling among all WDL-preserving moves until
                # the 75-move rule). `best_progress_moves` picks min-|DTZ| for
                # winning side / max-|DTZ| for losing side / all-equal for draws.
                best = self.tablebase.best_progress_moves(board)
                if best:
                    kept = {m: c for m, c in node.children.items() if m in best}
                    if kept:
                        s = sum(c.P for c in kept.values()) or 1.0
                        for c in kept.values():
                            c.P = c.P / s
                        node.children = kept
                # Syzygy's value_scalar is already CP-POV (it negates internally
                # for the losing side), so no white-POV flip needed here.
                return tb_value

        v_scalar = wdl_to_scalar(value.cpu())  # WDL logits → scalar ∈ [-1, 1]
        # P2 pilot (GH #6): when the network is trained on W-POV value labels,
        # the value head outputs white-POV. MCTS backup assumes CP-POV (it
        # alternates the sign every ply). Flip back to CP-POV here so the
        # tree-walk logic is unchanged.
        try:
            from config import WHITE_POV_VALUE as _WPOV  # type: ignore
        except ImportError:
            _WPOV = False
        if _WPOV and board.turn == chess.BLACK:
            v_scalar = -v_scalar
        return v_scalar

    def _terminal_value(self, result: str, turn: bool) -> float:
        """Convert PGN result string to a value from the current player's view."""
        if result == "1-0":
            return 1.0 if turn == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if turn == chess.WHITE else 1.0
        return 0.0  # Draw
