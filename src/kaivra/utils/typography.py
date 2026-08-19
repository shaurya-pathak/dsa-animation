"""Typography helpers shared by layout and renderers."""

from __future__ import annotations


def is_metric_style(style: str | None) -> bool:
    """Return whether a typography role uses the metric alignment system."""
    return bool(style and style.startswith("metric"))


def split_metric_sign(content: str, style: str | None) -> tuple[str | None, str]:
    """Split a leading metric sign so it can hang beside the aligned magnitude.

    Authors should write ordinary values such as ``-0.40`` or ``−0.40``. The
    engine owns the typographic treatment; prompt authors do not need to create
    a separate sign object or alignment group.
    """
    if not is_metric_style(style) or len(content) < 2:
        return None, content
    if content[0] not in {"+", "-", "−"}:
        return None, content
    sign = "−" if content[0] == "-" else content[0]
    return sign, content[1:]


def metric_sign_slot_width(
    font_size: float,
    *,
    sign_scale: float,
    sign_gap: float,
) -> float:
    """Return a conservative reserved width for an optional metric sign.

    The slot is present for *all* metrics, not just signed values. That keeps
    magnitude columns aligned and gives a visible sign a bounded place to live
    inside its measured rectangle.
    """
    # Plus/minus glyphs in the supported sans-serif faces are comfortably
    # below 0.70em. Keeping that headroom avoids clipping when Cairo and Canvas
    # have slightly different glyph metrics.
    return sign_gap + font_size * sign_scale * 0.70


def split_equation(content: str) -> tuple[str, str] | None:
    """Return the two sides of a simple equation for equals-sign alignment."""
    if content.count("=") != 1:
        return None
    left, right = content.split("=", 1)
    if not left.strip() or not right.strip():
        return None
    return left.strip(), right.strip()


def aligned_equation_reveal(content: str, progress: float) -> tuple[str, bool, str] | None:
    """Reveal a simple equation while keeping its eventual equals anchor fixed.

    The left side grows toward a known equals position; after the equals sign
    appears, the right side grows away from it. This avoids the final-frame jump
    caused by switching a centered typewriter string into aligned equation text.
    """
    if split_equation(content) is None:
        return None

    visible_chars = int(len(content) * max(0.0, min(1.0, progress)))
    equals_index = content.index("=")
    left = content[: min(visible_chars, equals_index)].rstrip()
    show_equals = visible_chars > equals_index
    right_chars = max(0, visible_chars - equals_index - 1)
    right = content[equals_index + 1 : equals_index + 1 + right_chars].lstrip()
    return left, show_equals, right.rstrip()
