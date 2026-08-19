from __future__ import annotations

from kaivra.dsl.schema import ObjectSpec
from kaivra.layout.strategies._sizing import estimate_object_size
from kaivra.themes.editorial import EDITORIAL
from kaivra.utils.typography import (
    aligned_equation_reveal,
    metric_sign_slot_width,
    split_equation,
    split_metric_sign,
)


def test_compact_text_containers_reserve_the_renderer_font_scale() -> None:
    from kaivra.themes.registry import get_theme

    theme = get_theme("editorial")
    compact_token = ObjectSpec.model_validate(
        {
            "type": "token",
            "content": "BORROW SUPPORT",
            "size_variant": "compact",
        }
    )
    compact_circle = ObjectSpec.model_validate(
        {"type": "circle", "content": "FUND", "size_variant": "compact"}
    )

    token_size = estimate_object_size(compact_token, theme)
    circle_size = estimate_object_size(compact_circle, theme)

    # The scene-graph builder resolves compact body text to 72% of the theme
    # size. The estimated shell must use that same scale.
    compact_font_size = theme.font_size_body * 0.72
    expected_text_width = len("BORROW SUPPORT") * compact_font_size * 0.62
    assert token_size.width > expected_text_width
    assert circle_size.width > theme.box_min_height * 0.72


def test_focus_actor_styles_reserve_large_readable_shells() -> None:
    from kaivra.themes.registry import get_theme

    theme = get_theme("editorial")
    focus_fund = ObjectSpec.model_validate(
        {
            "type": "circle",
            "content": "FUND",
            "style": "focus-coral",
            "size_variant": "hero",
        }
    )
    focus_prime = ObjectSpec.model_validate(
        {
            "type": "box",
            "content": "PRIME ACCOUNT",
            "style": "focus-gold",
            "size_variant": "hero",
        }
    )

    assert estimate_object_size(focus_fund, theme).width >= 150
    assert estimate_object_size(focus_prime, theme).width >= 420


def test_metric_sign_is_owned_by_typography_system() -> None:
    assert split_metric_sign("-0.40", "metric-accent") == ("−", "0.40")
    assert split_metric_sign("−0.40", "metric") == ("−", "0.40")
    assert split_metric_sign("+0.10", "metric-warning") == ("+", "0.10")
    assert split_metric_sign("-0.40", "body") == (None, "-0.40")


def test_metric_sign_does_not_shift_magnitude_layout_width() -> None:
    positive = ObjectSpec.model_validate(
        {"type": "text", "content": "0.40", "style": "metric-accent"}
    )
    negative = ObjectSpec.model_validate(
        {"type": "text", "content": "-0.40", "style": "metric-accent"}
    )

    assert estimate_object_size(positive, EDITORIAL) == estimate_object_size(negative, EDITORIAL)
    assert (
        metric_sign_slot_width(
            EDITORIAL.resolve_style("metric-cyan")["font_size"],
            sign_scale=EDITORIAL.metric_sign_scale,
            sign_gap=EDITORIAL.metric_sign_gap,
        )
        > EDITORIAL.metric_sign_gap
    )


def test_simple_equations_expose_a_shared_visual_anchor() -> None:
    assert split_equation("× 0.50 = 0.35") == ("× 0.50", "0.35")
    assert split_equation("positive") is None
    assert split_equation("a = b = c") is None


def test_aligned_equation_reveal_keeps_the_equals_anchor_stable() -> None:
    assert aligned_equation_reveal("a = b", 0.4) == ("a", False, "")
    assert aligned_equation_reveal("a = b", 0.6) == ("a", True, "")
    assert aligned_equation_reveal("a = b", 1.0) == ("a", True, "b")
