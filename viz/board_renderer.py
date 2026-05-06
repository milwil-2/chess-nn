"""
Renders the chess board and pieces using Pygame.
Uses Unicode chess symbols rendered as text — no image files needed.
"""

import pygame
import chess

LIGHT_SQ = (240, 217, 181)
DARK_SQ  = (181, 136, 99)
HIGHLIGHT_SELECTED = (20, 85, 30, 180)   # Green, semi-transparent
HIGHLIGHT_LEGAL    = (20, 85, 30, 80)
HIGHLIGHT_LAST     = (205, 210, 106, 160)

UNICODE_PIECES = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}


class BoardRenderer:
    def __init__(self, surface: pygame.Surface, sq_size: int = 80, offset_x: int = 0, offset_y: int = 0):
        self.surface = surface
        self.sq_size = sq_size
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Load a font that supports chess Unicode symbols
        font_size = int(sq_size * 0.75)
        try:
            self.piece_font = pygame.font.SysFont("AppleSymbols", font_size)
        except Exception:
            self.piece_font = pygame.font.SysFont("symbola", font_size)

        self.coord_font = pygame.font.SysFont("helvetica", 14)

        # Overlay surface for transparent highlights
        self.overlay = pygame.Surface((sq_size * 8, sq_size * 8), pygame.SRCALPHA)

    def square_rect(self, sq: int, flip: bool = False) -> pygame.Rect:
        """Get pixel rect for a chess square (optionally flipped for Black's perspective)."""
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if flip:
            rank = 7 - rank
        col = file
        row = 7 - rank  # Pygame y increases downward
        x = self.offset_x + col * self.sq_size
        y = self.offset_y + row * self.sq_size
        return pygame.Rect(x, y, self.sq_size, self.sq_size)

    def pixel_to_square(self, px: int, py: int, flip: bool = False) -> int | None:
        """Convert a pixel position to a chess square index, or None if outside the board."""
        col = (px - self.offset_x) // self.sq_size
        row = (py - self.offset_y) // self.sq_size
        if not (0 <= col <= 7 and 0 <= row <= 7):
            return None
        rank = 7 - row
        if flip:
            rank = 7 - rank
        return chess.square(col, rank)

    def draw(self, board: chess.Board, selected_sq: int | None = None,
             last_move: chess.Move | None = None, legal_targets: list[int] = None,
             flip: bool = False):
        self.overlay.fill((0, 0, 0, 0))

        for sq in chess.SQUARES:
            rect = self.square_rect(sq, flip)
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            color = LIGHT_SQ if (file + rank) % 2 == 0 else DARK_SQ
            pygame.draw.rect(self.surface, color, rect)

        # Highlight last move
        if last_move:
            for sq in [last_move.from_square, last_move.to_square]:
                rect = self.square_rect(sq, flip)
                rel = pygame.Rect(rect.x - self.offset_x, rect.y - self.offset_y, self.sq_size, self.sq_size)
                pygame.draw.rect(self.overlay, HIGHLIGHT_LAST, rel)

        # Highlight selected square
        if selected_sq is not None:
            rect = self.square_rect(selected_sq, flip)
            rel = pygame.Rect(rect.x - self.offset_x, rect.y - self.offset_y, self.sq_size, self.sq_size)
            pygame.draw.rect(self.overlay, HIGHLIGHT_SELECTED, rel)

        # Highlight legal destinations
        if legal_targets:
            for sq in legal_targets:
                rect = self.square_rect(sq, flip)
                rel = pygame.Rect(rect.x - self.offset_x, rect.y - self.offset_y, self.sq_size, self.sq_size)
                pygame.draw.rect(self.overlay, HIGHLIGHT_LEGAL, rel)
                cx = rel.centerx
                cy = rel.centery
                pygame.draw.circle(self.overlay, HIGHLIGHT_LEGAL, (cx, cy), self.sq_size // 6)

        self.surface.blit(self.overlay, (self.offset_x, self.offset_y))

        # Draw pieces
        for sq, piece in board.piece_map().items():
            rect = self.square_rect(sq, flip)
            symbol = UNICODE_PIECES[(piece.piece_type, piece.color)]
            color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
            shadow = self.piece_font.render(symbol, True, (100, 100, 100))
            text = self.piece_font.render(symbol, True, color)
            self.surface.blit(shadow, (rect.x + 5, rect.y + 5))
            self.surface.blit(text, (rect.x + 4, rect.y + 4))

        # Coordinate labels
        for i in range(8):
            # Rank numbers (1-8) on left edge
            rank = i if flip else 7 - i
            label = self.coord_font.render(str(rank + 1), True, (80, 80, 80))
            self.surface.blit(label, (self.offset_x - 18, self.offset_y + i * self.sq_size + 4))
            # File letters (a-h) on bottom edge
            file_char = chr(ord('a') + (7 - i if flip else i))
            label = self.coord_font.render(file_char, True, (80, 80, 80))
            self.surface.blit(label, (self.offset_x + i * self.sq_size + self.sq_size // 2 - 4,
                                       self.offset_y + 8 * self.sq_size + 4))
