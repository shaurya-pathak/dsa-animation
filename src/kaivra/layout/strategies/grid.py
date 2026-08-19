"""Grid layout — arrange objects in a columns x rows grid."""

from __future__ import annotations

import math

from kaivra.dsl.schema import LayoutSpec, ObjectSpec
from kaivra.layout.strategies._sizing import estimate_object_size
from kaivra.themes.base import ThemeSpec
from kaivra.utils.geometry import Rect


class GridStrategy:
    def compute(
        self,
        objects: list[ObjectSpec],
        layout: LayoutSpec,
        bounds: Rect,
        theme: ThemeSpec,
    ) -> dict[str, Rect]:
        gap = theme.resolve_gap(layout.gap if isinstance(layout.gap, str) else str(layout.gap))
        # Preserve the existing row-major default for ordinary grid children,
        # but let an authored child pin either coordinate (and span cells).
        # This is especially useful for two visual states that deliberately
        # share one cell during a causal `replace` animation.
        automatic_cols = layout.columns or max(1, min(len(objects), 4))
        placements: list[tuple[ObjectSpec, int, int, int, int]] = []
        required_cols = automatic_cols
        required_rows = max(1, math.ceil(len(objects) / automatic_cols))
        for index, obj in enumerate(objects):
            default_row, default_col = divmod(index, automatic_cols)
            grid = obj.grid
            row = grid.row if grid and grid.row is not None else default_row + 1
            col = grid.col if grid and grid.col is not None else default_col + 1
            span = grid.span if grid and grid.span is not None else 1
            row_span = grid.row_span if grid and grid.row_span is not None else 1
            # Invalid authored values retain the historical forgiving behavior
            # of the top-level grid resolver: clamp them to the nearest valid
            # placement rather than failing at render time.
            row = max(1, row)
            col = max(1, col)
            span = max(1, span)
            row_span = max(1, row_span)
            placements.append((obj, row, col, row_span, span))
            required_cols = max(required_cols, col + span - 1)
            required_rows = max(required_rows, row + row_span - 1)

        cols = layout.columns or required_cols
        rows = layout.rows or required_rows

        cell_w = (bounds.width - gap * (cols - 1)) / cols
        cell_h = (bounds.height - gap * (rows - 1)) / rows

        results: dict[str, Rect] = {}
        for obj, requested_row, requested_col, requested_row_span, requested_span in placements:
            row = min(rows, requested_row)
            col = min(cols, requested_col)
            span = min(cols - col + 1, requested_span)
            row_span = min(rows - row + 1, requested_row_span)
            region_w = cell_w * span + gap * (span - 1)
            region_h = cell_h * row_span + gap * (row_span - 1)
            size = estimate_object_size(obj, theme)
            # Center object within its occupied region. Identical explicit
            # rows/columns therefore resolve to exactly identical centers.
            cx = bounds.x + (col - 1) * (cell_w + gap) + region_w / 2
            cy = bounds.y + (row - 1) * (cell_h + gap) + region_h / 2
            w = min(size.width, region_w)
            h = min(size.height, region_h)
            obj_id = obj.id or f"obj_{id(obj)}"
            results[obj_id] = Rect(cx - w / 2, cy - h / 2, w, h)

        return results
