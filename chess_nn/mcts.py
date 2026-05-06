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
import chess
import torch
import numpy as np

from chess_nn.board_encoding import board_to_tensor
from chess_nn.move_encoding import move_to_index, index_to_move, get_legal_move_indices

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
        return {m: v / total for m, v in powered.items()}


class MCTS:
    """
    The search engine. Owns the root node and runs simulations.

    Usage:
        mcts = MCTS(model, num_simulations=400)
        move = mcts.search(board)
    """

    def __init__(self, model, num_simulations: int = 200):
        self.model = model
        self.num_simulations = num_simulations

    def search(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """
        Run `num_simulations` simulations from `board`, return the chosen move.
        """
        root = Node(prior=1.0)
        self._expand(root, board)  # Expand root immediately so it has children

        for _ in range(self.num_simulations):
            node = root
            # stack=False: skip copying move history — MCTS never undoes moves, saves ~80× _BoardState allocs per sim
            scratch_board = board.copy(stack=False)
            path = [node]                 # Track nodes visited this simulation for backup

            # --- Step 1: Selection ---
            # Walk down the tree, always picking the highest-UCB child,
            # until we reach a node that hasn't been expanded yet.
            while node.is_expanded and not scratch_board.is_game_over():
                move, node = node.best_child()
                scratch_board.push(move)
                path.append(node)

            # --- Step 2: Expansion + Evaluation ---
            if scratch_board.is_game_over():
                # Terminal node: use the actual game result as the value
                result = scratch_board.result()
                value = self._terminal_value(result, scratch_board.turn)
            else:
                # Ask the network: what's the value here, and which moves look promising?
                value = self._expand(node, scratch_board)

            # --- Step 3: Backup ---
            # Send the value back up the path.
            # Flip sign at each level: what's good for the current player
            # is bad for the previous player (they're opponents).
            for i, visited_node in enumerate(reversed(path)):
                visited_node.N += 1
                # Flip value at each level to account for alternating players
                visited_node.W += value if i % 2 == 0 else -value

        # After all simulations: pick the move from the root's visit distribution
        dist = root.visit_distribution(temperature=temperature)
        moves = list(dist.keys())
        probs = [dist[m] for m in moves]
        chosen = np.random.choice(len(moves), p=probs)
        return moves[chosen]

    def get_policy(self, board: chess.Board, temperature: float = 1.0) -> dict[chess.Move, float]:
        """
        Run search and return the full visit distribution (used for training targets).
        This is the 'improved policy' that MCTS produces — better than the raw network output.
        """
        root = Node(prior=1.0)
        self._expand(root, board)

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy(stack=False)
            path = [node]

            while node.is_expanded and not scratch_board.is_game_over():
                move, node = node.best_child()
                scratch_board.push(move)
                path.append(node)

            if scratch_board.is_game_over():
                value = self._terminal_value(scratch_board.result(), scratch_board.turn)
            else:
                value = self._expand(node, scratch_board)

            for i, visited_node in enumerate(reversed(path)):
                visited_node.N += 1
                visited_node.W += value if i % 2 == 0 else -value

        return root.visit_distribution(temperature=temperature)

    def _expand(self, node: Node, board: chess.Board) -> float:
        """
        Run the neural network on this position.
        Populate the node's children with prior probabilities from the policy head.
        Returns the value estimate from the value head.
        """
        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)

        # Get legal move indices and mask out illegal moves
        legal_indices = get_legal_move_indices(board)
        policy = policy_logits.squeeze(0).cpu().numpy()

        # Softmax over legal moves only (illegal = -inf)
        legal_logits = np.array([policy[i] for i in legal_indices])
        legal_logits -= legal_logits.max()  # Numerical stability
        priors = np.exp(legal_logits)
        priors /= priors.sum()

        # Create a child node for each legal move with its prior probability
        for idx, move_idx in enumerate(legal_indices):
            move = index_to_move(move_idx, board)
            if move in board.legal_moves:
                node.children[move] = Node(prior=float(priors[idx]))

        node.is_expanded = True
        return float(value.cpu().item())

    def _terminal_value(self, result: str, turn: bool) -> float:
        """Convert PGN result string to a value from the current player's view."""
        if result == "1-0":
            return 1.0 if turn == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if turn == chess.WHITE else 1.0
        return 0.0  # Draw
