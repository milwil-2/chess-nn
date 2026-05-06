"""
Draws a vertical evaluation bar showing who is winning.
White fills from the bottom, black from the top.
Value = +1 means current player has a winning position, -1 means losing.
"""

import pygame


def draw_value_bar(surface: pygame.Surface, value: float, x: int, y: int,
                   width: int = 20, height: int = 640, turn_is_white: bool = True):
    """
    value: model output in [-1, +1], from CURRENT player's perspective
    We convert to absolute white/black perspective for display.
    """
    # Convert to white's advantage: +1 = white winning, -1 = black winning
    white_advantage = value if turn_is_white else -value

    # Map [-1, 1] → [0, 1] for fill ratio
    fill_ratio = (white_advantage + 1) / 2.0
    fill_ratio = max(0.0, min(1.0, fill_ratio))

    # Background (black side)
    pygame.draw.rect(surface, (30, 30, 30), (x, y, width, height))

    # White fill from bottom
    white_height = int(fill_ratio * height)
    pygame.draw.rect(surface, (230, 230, 230),
                     (x, y + height - white_height, width, white_height))

    # Centre line
    pygame.draw.line(surface, (100, 100, 100),
                     (x, y + height // 2), (x + width, y + height // 2), 1)

    # Numeric label
    font = pygame.font.SysFont("helvetica", 13)
    sign = "+" if white_advantage >= 0 else ""
    label = font.render(f"{sign}{white_advantage:.2f}", True, (200, 200, 200))
    surface.blit(label, (x, y + height + 6))
