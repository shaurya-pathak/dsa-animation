from __future__ import annotations

import json
import math

import pytest

from kaivra.dsl.parser import parse_string
from kaivra.dsl.schema import ObjectSpec, ObjectType
from kaivra.layout.strategies._sizing import estimate_object_size
from kaivra.render.cairo_renderer import CairoRenderer
from kaivra.render.web.exporter import build_web_preview_html
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.themes.registry import get_theme


def _sigmoid_document():
    return parse_string(
        json.dumps(
            {
                "version": "1.5",
                "meta": {
                    "theme": "editorial",
                    "resolution": [1200, 900],
                    "show_subtitles": False,
                },
                "scenes": [
                    {
                        "id": "sigmoid",
                        "duration": "3s",
                        "auto_visible": False,
                        "show_progress_bar": False,
                        "layout": "center",
                        "objects": [
                            {
                                "id": "curve",
                                "type": "sigmoid_plot",
                                "style": "metric-gold",
                                "size_variant": "hero",
                                "sigmoid_input": 0.57,
                                "sigmoid_input_label": "+0.57",
                                "sigmoid_output_label": "64%",
                                "sigmoid_caption": "SIGMOID",
                                "visible": False,
                            }
                        ],
                        "animations": [
                            {
                                "action": "draw",
                                "target": "curve",
                                "at": "0s",
                                "duration": "1.5s",
                            }
                        ],
                    }
                ],
            }
        ),
        format="json",
    )


def test_sigmoid_plot_parses_builds_and_preserves_semantic_input() -> None:
    doc = _sigmoid_document()
    schema = doc.model_json_schema()

    assert "sigmoid_plot" in schema["$defs"]["ObjectType"]["enum"]
    for field_name in (
        "sigmoid_input",
        "sigmoid_input_label",
        "sigmoid_output_label",
        "sigmoid_caption",
    ):
        assert field_name in schema["$defs"]["ObjectSpec"]["properties"]

    node = build_scene_graph(doc, get_theme("editorial")).scenes[0].node_map["curve"]
    assert node.obj_type is ObjectType.SIGMOID_PLOT
    assert node.sigmoid_input == pytest.approx(0.57)
    assert node.sigmoid_input_label == "+0.57"
    assert node.sigmoid_output_label == "64%"
    assert round(100 / (1 + math.exp(-node.sigmoid_input))) == 64


def test_sigmoid_plot_has_legible_size_presets() -> None:
    theme = get_theme("editorial")
    compact = estimate_object_size(
        ObjectSpec.model_validate({"type": "sigmoid_plot", "size_variant": "compact"}), theme
    )
    default = estimate_object_size(ObjectSpec.model_validate({"type": "sigmoid_plot"}), theme)
    hero = estimate_object_size(
        ObjectSpec.model_validate({"type": "sigmoid_plot", "size_variant": "hero"}), theme
    )

    assert (compact.width, compact.height) == pytest.approx((300.0, 210.0))
    assert (default.width, default.height) == pytest.approx((420.0, 270.0))
    assert (hero.width, hero.height) == pytest.approx((520.0, 330.0))


def test_sigmoid_plot_draws_in_cairo_and_serializes_for_web() -> None:
    doc = _sigmoid_document()
    theme = get_theme("editorial")
    graph = build_scene_graph(doc, theme)
    pixels = CairoRenderer(theme).render_frame_to_bytes(graph, 1.5)

    assert len(set(pixels)) > 20

    html = build_web_preview_html(doc)
    assert '"type": "sigmoid_plot"' in html
    assert '"sigmoidInput": 0.57' in html
    assert '"sigmoidOutputLabel": "64%"' in html
    assert "case 'sigmoid_plot': drawSigmoidPlot(ctx, node); break;" in html
    assert "function drawSigmoidPlot(ctx, node)" in html
    assert "const probability = 1 / (1 + Math.exp(-inputValue));" in html
    assert "const pointY = bottom - probability * plotHeight;" in html
