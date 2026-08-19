"""Object size estimation without a rendering context.

Uses heuristics based on content length, font size, and theme settings.
When we have a Cairo context available, we can use actual text measurement.
"""

from __future__ import annotations

from kaivra.dsl.schema import LayoutSpec, LayoutType, ObjectSpec, ObjectType
from kaivra.themes.base import ThemeSpec
from kaivra.utils.geometry import Size
from kaivra.utils.typography import is_metric_style, metric_sign_slot_width, split_metric_sign

# A conservative Inter/Arial-equivalent width for the all-caps labels that
# dominate teaching diagrams. The old 0.62 estimate sized the shell from a
# narrower lowercase average, so labels such as ``BORROW SUPPORT`` could draw
# beyond the token even though semantic layout believed they fit.
_ESTIMATED_GLYPH_WIDTH = 0.72


def _variant_multiplier(obj: ObjectSpec) -> float:
    """Match the font scale resolved by the scene-graph builder.

    Boxes, tokens, text, and circles can all contain text. Their layout
    footprint must therefore use the same variant scale as the renderer-facing
    style, or compact labels are measured at 42% while rendered at 72%.
    """
    if obj.size_variant.value == "compact":
        return 0.72
    if obj.size_variant.value == "hero":
        return 1.18
    return 1.0


def _portrait_variant_multiplier(obj: ObjectSpec) -> float:
    """Keep illustration footprints independent from typography scaling."""
    if obj.size_variant.value == "compact":
        return 0.42
    if obj.size_variant.value == "hero":
        return 1.35
    return 1.0


def _circle_variant_multiplier(obj: ObjectSpec) -> float:
    """Let labeled actors fit text without enlarging unlabeled data marks."""
    if obj.content:
        return _variant_multiplier(obj)
    return _portrait_variant_multiplier(obj)


def _circle_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    variant = _circle_variant_multiplier(obj)
    diameter = theme.box_min_height * variant
    if obj.content:
        style = theme.resolve_style(obj.style)
        font_size = style.get("font_size", theme.font_size_body) * variant
        text_width = len(obj.content) * font_size * _ESTIMATED_GLYPH_WIDTH
        diameter = max(diameter, text_width + theme.box_padding * 2.5)
    return Size(diameter, diameter)


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
        case ObjectType.CIRCLE:
            return _circle_size(obj, theme)
        case ObjectType.LINEAR_METER:
            return _linear_meter_size(obj)
        case ObjectType.SIGMOID_PLOT:
            return _sigmoid_plot_size(obj)
        case ObjectType.PET_PORTRAIT | ObjectType.PET:
            # A pet is an illustration, not a text container. Keep its base
            # square footprint stable across themes. Feature labels reserve
            # real left/right gutters instead of leaking into neighbor space.
            side = 220.0 * _portrait_variant_multiplier(obj)
            if obj.show_feature_labels:
                return Size(side * 1.9, side)
            return Size(side, side)
        case ObjectType.SEMANTIC_ICON:
            # Icons carry visual meaning, so they get a genuinely legible
            # footprint instead of being treated as decorative badges.
            side = 160.0 * _portrait_variant_multiplier(obj)
            caption_height = 28.0 * _variant_multiplier(obj) if obj.content else 0.0
            return Size(side, side + caption_height)
        case ObjectType.CALLOUT:
            text = obj.content or ""
            width = min(300, max(len(text) * 9, 150))
            lines = max(1, len(text) // 35 + 1)
            height = lines * 22 + 30
            return Size(width, height)
        case _:
            return Size(theme.box_min_width, theme.box_min_height)


def _linear_meter_size(obj: ObjectSpec) -> Size:
    """Reserve a broad, label-safe footprint for a semantic teaching scale."""
    if obj.size_variant.value == "compact":
        return Size(300.0, 88.0)
    if obj.size_variant.value == "hero":
        return Size(720.0, 176.0)
    return Size(540.0, 136.0)


def _sigmoid_plot_size(obj: ObjectSpec) -> Size:
    """Reserve enough room for axes, the S-curve, guides, and readable labels."""
    if obj.size_variant.value == "compact":
        return Size(300.0, 210.0)
    if obj.size_variant.value == "hero":
        return Size(520.0, 330.0)
    return Size(420.0, 270.0)


def _text_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    style = theme.resolve_style(obj.style)
    font_size = style.get("font_size", theme.font_size_body) * _variant_multiplier(obj)
    text = obj.content or ""
    _sign, text = split_metric_sign(text, obj.style)
    # Inter's uppercase teaching labels are materially wider than the old
    # 0.55 heuristic. A slightly conservative estimate prevents adjacent
    # phrases from colliding and keeps shells around their full label.
    char_width = font_size * _ESTIMATED_GLYPH_WIDTH
    sign_slot = (
        metric_sign_slot_width(
            font_size,
            sign_scale=theme.metric_sign_scale,
            sign_gap=theme.metric_sign_gap,
        )
        if is_metric_style(obj.style)
        else 0.0
    )
    width = max(len(text) * char_width + sign_slot, theme.box_min_width)
    height = font_size * 1.4
    return Size(width, height)


def _box_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    variant = _variant_multiplier(obj)
    text = obj.content or ""
    style = theme.resolve_style(obj.style)
    font_size = style.get("font_size", theme.font_size_body) * variant
    char_width = font_size * _ESTIMATED_GLYPH_WIDTH
    text_width = len(text) * char_width
    shadow_extra = theme.shadow_offset if theme.shadow else 0
    padding = theme.box_padding * (0.75 if obj.size_variant.value == "compact" else variant)
    width = max(text_width + padding * 2, theme.box_min_width * variant) + shadow_extra
    height = max(font_size * 1.4 + padding * 2, theme.box_min_height * variant) + shadow_extra
    return Size(width, height)


def _token_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    variant = _variant_multiplier(obj)
    text = obj.content or ""
    style = theme.resolve_style(obj.style)
    font_size = style.get("font_size", theme.font_size_body) * variant
    char_width = font_size * _ESTIMATED_GLYPH_WIDTH
    text_width = len(text) * char_width
    padding = theme.token_padding * (0.75 if obj.size_variant.value == "compact" else variant)
    width = text_width + padding * 2 + 8
    height = font_size * 1.4 + padding * 2
    return Size(max(width, 50 * variant), height)


def _group_size(obj: ObjectSpec, theme: ThemeSpec) -> Size:
    """Estimate group size from its layout and children."""
    if not obj.children:
        return Size(theme.box_min_width, theme.box_min_height)
    child_sizes = [estimate_object_size(c, theme) for c in obj.children]
    child_layout = (
        obj.layout if isinstance(obj.layout, LayoutSpec) else LayoutSpec(type=LayoutType.FLOW)
    )
    gap = theme.resolve_gap(
        child_layout.gap if isinstance(child_layout.gap, str) else str(child_layout.gap)
    )

    if child_layout.type == LayoutType.STACK or (
        child_layout.type == LayoutType.FLOW and child_layout.direction == "vertical"
    ):
        total_h = sum(s.height for s in child_sizes) + gap * (len(child_sizes) - 1)
        max_w = max(s.width for s in child_sizes)
        return Size(max_w, total_h)

    if child_layout.type == LayoutType.GRID:
        cols = child_layout.columns or len(child_sizes) or 1
        rows = child_layout.rows or ((len(child_sizes) + cols - 1) // cols)
        max_w = max(s.width for s in child_sizes)
        max_h = max(s.height for s in child_sizes)
        total_w = max_w * cols + gap * (cols - 1)
        total_h = max_h * rows + gap * (rows - 1)
        return Size(total_w, total_h)

    # Default: horizontal flow (including carousel)
    total_w = sum(s.width for s in child_sizes) + gap * (len(child_sizes) - 1)
    max_h = max(s.height for s in child_sizes)
    return Size(total_w, max_h)
