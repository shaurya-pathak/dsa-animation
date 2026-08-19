from __future__ import annotations

import json

import pytest

from kaivra.dsl.parser import parse_string
from kaivra.dsl.schema import ObjectSpec, ObjectType, SemanticIconName
from kaivra.layout.strategies._sizing import estimate_object_size
from kaivra.render.cairo_renderer import CairoRenderer
from kaivra.render.web.exporter import build_web_preview_html
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.themes.registry import get_theme

ICON_NAMES = [
    "fund",
    "storefront",
    "warehouse",
    "cash",
    "shares",
    "borrow",
    "loan",
    "handshake",
]


def _icon_document():
    return parse_string(
        json.dumps(
            {
                "version": "1.5",
                "meta": {
                    "theme": "editorial",
                    "resolution": [1600, 900],
                    "show_subtitles": False,
                },
                "scenes": [
                    {
                        "id": "visual-vocabulary",
                        "duration": "3s",
                        "auto_visible": True,
                        "show_progress_bar": False,
                        "layout": {"type": "grid", "columns": 4, "gap": "large"},
                        "objects": [
                            {
                                "id": f"icon-{name}",
                                "type": "semantic_icon",
                                "icon_name": name,
                                "content": name.title(),
                            }
                            for name in ICON_NAMES
                        ],
                    }
                ],
            }
        ),
        format="json",
    )


def test_semantic_icon_has_a_bounded_stable_schema_vocabulary() -> None:
    schema = _icon_document().model_json_schema()

    assert "semantic_icon" in schema["$defs"]["ObjectType"]["enum"]
    assert schema["$defs"]["SemanticIconName"]["enum"] == ICON_NAMES
    assert "icon_name" in schema["$defs"]["ObjectSpec"]["properties"]
    assert list(SemanticIconName) == [SemanticIconName(name) for name in ICON_NAMES]

    with pytest.raises(ValueError, match="semantic_icon requires icon_name"):
        ObjectSpec.model_validate({"type": "semantic_icon"})
    with pytest.raises(ValueError, match="icon_name requires type='semantic_icon'"):
        ObjectSpec.model_validate({"type": "box", "icon_name": "fund"})


def test_semantic_icons_build_at_legible_sizes() -> None:
    doc = _icon_document()
    graph = build_scene_graph(doc, get_theme("editorial"))
    nodes = graph.scenes[0].node_map

    assert [nodes[f"icon-{name}"].icon_name for name in ICON_NAMES] == ICON_NAMES
    assert all(nodes[f"icon-{name}"].obj_type is ObjectType.SEMANTIC_ICON for name in ICON_NAMES)

    default = estimate_object_size(
        ObjectSpec.model_validate({"type": "semantic_icon", "icon_name": "fund"}),
        get_theme("editorial"),
    )
    hero = estimate_object_size(
        ObjectSpec.model_validate(
            {"type": "semantic_icon", "icon_name": "fund", "size_variant": "hero"}
        ),
        get_theme("editorial"),
    )
    assert (default.width, default.height) == pytest.approx((160.0, 160.0))
    assert (hero.width, hero.height) == pytest.approx((216.0, 216.0))


def test_semantic_icons_render_in_cairo_and_web_preview() -> None:
    doc = _icon_document()
    theme = get_theme("editorial")
    graph = build_scene_graph(doc, theme)
    pixels = CairoRenderer(theme).render_frame_to_bytes(graph, 0.5)

    # The flat illustrations contribute several deterministic channel colors
    # beyond the editorial background and neutral outline.
    assert len(set(pixels)) > 30

    html = build_web_preview_html(doc)
    assert '"type": "semantic_icon"' in html
    for name in ICON_NAMES:
        assert f'"iconName": "{name}"' in html
        assert f"icon === '{name}'" in html
    assert "function drawSemanticIcon(ctx, node)" in html
    assert "case 'semantic_icon': drawSemanticIcon(ctx, node); break;" in html
