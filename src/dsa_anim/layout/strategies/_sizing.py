"""Object size estimation without a rendering context.

Uses heuristics based on content length, font size, and theme settings.
When we have a Cairo context available, we can use actual text measurement.
"""

from __future__ import annotations

from dsa_anim.dsl.schema import ObjectSpec, ObjectType
from dsa_anim.themes.base import ThemeSpec
from dsa_anim.utils.geometry import Size


def estimate_object_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    """Estimate the rendered size of an object."""
    match obj.type:
        case ObjectType.TEXT:
            return _text_size(obj, theme)
        case ObjectType.BOX:
            return _box_size(obj, theme)
        case ObjectType.TOKEN:
            return _token_size(obj, theme)
        case ObjectType.CONNECTOR:
            return Size(0, 0)  # connectors don't occupy layout space
        case ObjectType.GROUP:
            return _group_size(obj, theme)
        case ObjectType.MATRIX:
            return _matrix_size(obj, theme)
        case ObjectType.ATTENTION_MAP:
            return _attention_map_size(obj, theme)
        case ObjectType.PROBABILITY_BAR:
            return _probability_bar_size(obj, theme)
        case ObjectType.CIRCLE:
            return Size(theme.box_min_height, theme.box_min_height)
        case ObjectType.CALLOUT:
            text = obj.content or ""
            width = min(300, max(len(text) * 9, 150))
            lines = max(1, len(text) // 35 + 1)
            height = lines * 22 + 30
            return Size(width, height)
        case _:
            return Size(theme.box_min_width, theme.box_min_height)


def _text_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    style = theme.resolve_style(obj.style)
    font_size = style.get("font_size", theme.font_size_body)
    text = obj.content or ""
    # Rough estimate: ~0.6 * font_size per character width
    char_width = font_size * 0.55
    width = max(len(text) * char_width, theme.box_min_width)
    height = font_size * 1.4
    return Size(width, height)


def _box_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    text = obj.content or ""
    char_width = theme.font_size_body * 0.55
    text_width = len(text) * char_width
    width = max(text_width + theme.box_padding * 2, theme.box_min_width)
    height = max(theme.font_size_body * 1.4 + theme.box_padding * 2, theme.box_min_height)
    return Size(width, height)


def _token_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    text = obj.content or ""
    char_width = theme.font_size_body * 0.55
    text_width = len(text) * char_width
    width = text_width + theme.token_padding * 2 + 8  # extra for badge
    height = theme.font_size_body * 1.4 + theme.token_padding * 2
    return Size(max(width, 50), height)


def _group_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    """Estimate group size as sum of children."""
    if not obj.children:
        return Size(theme.box_min_width, theme.box_min_height)
    child_sizes = [estimate_object_size(c, theme) for c in obj.children]
    # Default: horizontal flow
    total_w = sum(s.width for s in child_sizes) + theme.gap_small * (len(child_sizes) - 1)
    max_h = max(s.height for s in child_sizes)
    return Size(total_w, max_h + theme.box_padding * 2)


def _matrix_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    rows = obj.rows or 4
    cols = obj.cols or 4
    cell_size = 40
    label_width = 80 if obj.labels else 0
    width = cols * cell_size + label_width + theme.box_padding * 2
    height = rows * cell_size + theme.box_padding * 2
    return Size(width, height)


def _attention_map_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    n = len(obj.tokens) if obj.tokens else 4
    # Tokens arranged in a circle/arc, need space
    size = max(n * 80, 400)
    return Size(size, size * 0.7)


def _probability_bar_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    n = len(obj.items) if obj.items else 4
    width = 500
    height = n * 40 + theme.box_padding * 2
    return Size(width, height)
