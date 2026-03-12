"""Basic geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def lerp(self, other: Point, t: float) -> Point:
        return Point(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)


@dataclass
class Size:
    width: float
    height: float


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def top_center(self) -> Point:
        return Point(self.x + self.width / 2, self.y)

    @property
    def bottom_center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height)

    @property
    def left_center(self) -> Point:
        return Point(self.x, self.y + self.height / 2)

    @property
    def right_center(self) -> Point:
        return Point(self.x + self.width, self.y + self.height / 2)

    def inset(self, padding: float) -> Rect:
        return Rect(
            self.x + padding,
            self.y + padding,
            self.width - 2 * padding,
            self.height - 2 * padding,
        )

    def subdivide_vertical(self, ratios: list[float], gap: float = 0) -> list[Rect]:
        """Split into vertical sections according to ratios."""
        total = sum(ratios)
        total_gap = gap * (len(ratios) - 1)
        available = self.height - total_gap
        rects = []
        y = self.y
        for r in ratios:
            h = available * (r / total)
            rects.append(Rect(self.x, y, self.width, h))
            y += h + gap
        return rects

    def subdivide_horizontal(self, ratios: list[float], gap: float = 0) -> list[Rect]:
        """Split into horizontal sections according to ratios."""
        total = sum(ratios)
        total_gap = gap * (len(ratios) - 1)
        available = self.width - total_gap
        rects = []
        x = self.x
        for r in ratios:
            w = available * (r / total)
            rects.append(Rect(x, self.y, w, self.height))
            x += w + gap
        return rects
