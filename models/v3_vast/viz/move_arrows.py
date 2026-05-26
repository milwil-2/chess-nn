"""
Draws arrows on the board showing the model's top move candidates.
Arrow opacity and width = move probability. Green = top choice, yellow = alternatives.
"""

import pygame
import chess
import math


def draw_arrow(surface: pygame.Surface, color: tuple, start: tuple, end: tuple,
               width: int, alpha: int):
    """Draw a semi-transparent arrow from start to end."""
    if start == end:
        return

    arrow_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return

    # Shorten arrow so it doesn't overlap the piece
    shrink = 20
    sx = start[0] + dx / length * shrink
    sy = start[1] + dy / length * shrink
    ex = end[0] - dx / length * shrink
    ey = end[1] - dy / length * shrink

    rgba = (*color[:3], alpha)
    pygame.draw.line(arrow_surf, rgba, (int(sx), int(sy)), (int(ex), int(ey)), width)

    # Arrowhead
    angle = math.atan2(dy, dx)
    head_len = max(width * 2, 12)
    for side in [math.pi / 6, -math.pi / 6]:
        hx = ex - head_len * math.cos(angle - side)
        hy = ey - head_len * math.sin(angle - side)
        pygame.draw.line(arrow_surf, rgba, (int(ex), int(ey)), (int(hx), int(hy)), width)

    surface.blit(arrow_surf, (0, 0))


def draw_move_arrows(surface: pygame.Surface, move_probs: list[tuple],
                     renderer, flip: bool = False):
    """
    Draw arrows for a list of (move, probability) tuples.

    move_probs: output of policy_to_moves() — sorted highest prob first
    """
    if not move_probs:
        return

    max_prob = move_probs[0][1] if move_probs else 1.0

    for i, (move, prob) in enumerate(move_probs):
        from_rect = renderer.square_rect(move.from_square, flip)
        to_rect = renderer.square_rect(move.to_square, flip)

        start = from_rect.center
        end = to_rect.center

        # Top move = green, rest = yellow/orange
        color = (50, 200, 50) if i == 0 else (220, 180, 50)

        # Width and opacity scale with probability
        ratio = prob / max_prob
        width = max(2, int(ratio * 8))
        alpha = max(40, int(ratio * 220))

        draw_arrow(surface, color, start, end, width, alpha)
