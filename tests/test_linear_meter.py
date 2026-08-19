from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kaivra.dsl.parser import parse_string
from kaivra.dsl.schema import AnimAction, ObjectSpec, ObjectType
from kaivra.layout.strategies._sizing import estimate_object_size
from kaivra.render.cairo_renderer import CairoRenderer
from kaivra.render.web.exporter import build_web_preview_html
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.scene_graph.timeline import apply_animations_at_time
from kaivra.themes.registry import get_theme


def _meter_document():
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
                        "id": "meters",
                        "duration": "3s",
                        "auto_visible": True,
                        "show_progress_bar": False,
                        "layout": {"type": "flow", "direction": "vertical", "gap": "medium"},
                        "objects": [
                            {
                                "id": "clue_strength",
                                "type": "linear_meter",
                                "style": "coral",
                                "meter_min": 0,
                                "meter_max": 10,
                                "meter_value": 7,
                                "meter_left_label": "0",
                                "meter_center_label": "5",
                                "meter_right_label": "10",
                                "meter_value_label": "7 / 10",
                                "meter_caption": "Floppy-ear clue strength",
                            },
                            {
                                "id": "internal_lean",
                                "type": "linear_meter",
                                "style": "coral",
                                "meter_min": -1,
                                "meter_max": 1,
                                "meter_value": 0.57,
                                "meter_left_label": "CAT",
                                "meter_center_label": "0 EVEN",
                                "meter_right_label": "DOG",
                                "meter_value_label": "+0.57",
                                "meter_caption": "Internal lean",
                            },
                            {
                                "id": "probability",
                                "type": "linear_meter",
                                "style": "cyan",
                                "meter_min": 0,
                                "meter_max": 100,
                                "meter_value": 64,
                                "meter_left_label": "0%",
                                "meter_center_label": "50 / 50",
                                "meter_right_label": "100%",
                                "meter_value_label": "64% DOG",
                                "meter_caption": "Readable chance",
                            },
                        ],
                    }
                ],
            }
        ),
        format="json",
    )


def test_linear_meter_parses_builds_and_preserves_semantic_values() -> None:
    doc = _meter_document()
    schema = doc.model_json_schema()
    assert "linear_meter" in schema["$defs"]["ObjectType"]["enum"]
    properties = schema["$defs"]["ObjectSpec"]["properties"]
    for field_name in (
        "meter_value",
        "meter_min",
        "meter_max",
        "meter_left_label",
        "meter_center_label",
        "meter_right_label",
        "meter_value_label",
        "meter_caption",
    ):
        assert field_name in properties

    graph = build_scene_graph(doc, get_theme("editorial"))
    clue = graph.scenes[0].node_map["clue_strength"]
    lean = graph.scenes[0].node_map["internal_lean"]
    probability = graph.scenes[0].node_map["probability"]

    assert clue.obj_type is ObjectType.LINEAR_METER
    assert (clue.meter_min, clue.meter_value, clue.meter_max) == (0, 7, 10)
    assert (lean.meter_min, lean.meter_value, lean.meter_max) == (-1, 0.57, 1)
    assert probability.meter_value_label == "64% DOG"
    assert clue.rect.width == pytest.approx(540.0)


def test_linear_meter_has_label_safe_size_presets_and_validates_its_range() -> None:
    theme = get_theme("editorial")
    compact = estimate_object_size(
        ObjectSpec.model_validate({"type": "linear_meter", "size_variant": "compact"}), theme
    )
    default = estimate_object_size(ObjectSpec.model_validate({"type": "linear_meter"}), theme)
    hero = estimate_object_size(
        ObjectSpec.model_validate({"type": "linear_meter", "size_variant": "hero"}), theme
    )

    assert (compact.width, compact.height) == pytest.approx((300.0, 88.0))
    assert (default.width, default.height) == pytest.approx((540.0, 136.0))
    assert (hero.width, hero.height) == pytest.approx((720.0, 176.0))

    with pytest.raises(ValidationError, match="meter_max"):
        ObjectSpec.model_validate({"type": "linear_meter", "meter_min": 1, "meter_max": 1})


def test_linear_meter_draws_in_cairo_and_serializes_for_web() -> None:
    doc = _meter_document()
    theme = get_theme("editorial")
    graph = build_scene_graph(doc, theme)
    pixels = CairoRenderer(theme).render_frame_to_bytes(graph, 0.5)

    # The rails, bounded fills, pointers, and labels all pass through Cairo's
    # native primitive instead of degrading to a generic text/box fallback.
    assert len(set(pixels)) > 20

    html = build_web_preview_html(doc)
    assert '"type": "linear_meter"' in html
    assert '"meterValue": 0.57' in html
    assert '"meterCenterLabel": "0 EVEN"' in html
    assert "case 'linear_meter': drawLinearMeter(ctx, node); break;" in html
    assert "function drawLinearMeter(ctx, node)" in html
    assert "const ratio = Math.max(0, Math.min(1" in html
    assert "const pointerX = trackX + trackWidth * ratio;" in html


def test_meter_to_animates_a_linear_meter_deterministically_in_both_renderers() -> None:
    doc = parse_string(
        json.dumps(
            {
                "version": "1.5",
                "meta": {"theme": "editorial", "show_subtitles": False},
                "scenes": [
                    {
                        "id": "meter_motion",
                        "duration": "3s",
                        "auto_visible": True,
                        "layout": "center",
                        "objects": [
                            {
                                "id": "confidence",
                                "type": "linear_meter",
                                "meter_min": 0,
                                "meter_max": 100,
                                "meter_value": 20,
                            }
                        ],
                        "animations": [
                            {
                                "id": "confidence_rises",
                                "action": "meter-to",
                                "target": "confidence",
                                "meter_value": 80,
                                "at": "0s",
                                "duration": "2s",
                                "easing": "linear",
                            }
                        ],
                    }
                ],
            }
        ),
        format="json",
    )
    graph = build_scene_graph(doc, get_theme("editorial"))
    scene = graph.scenes[0]
    keyframe = scene.timeline[0]

    assert keyframe.action is AnimAction.METER_TO
    assert keyframe.to_value == 80

    apply_animations_at_time(scene.node_map, scene.timeline, 1.0)
    assert scene.node_map["confidence"].meter_value == pytest.approx(50.0)

    # Seeking backwards restores the authored semantic value rather than the
    # previous sampled frame's value.
    apply_animations_at_time(scene.node_map, scene.timeline, 0.0)
    assert scene.node_map["confidence"].meter_value == pytest.approx(20.0)

    html = build_web_preview_html(doc)
    assert '"action": "meter-to"' in html
    assert '"baseMeterValue": 20.0' in html
    assert "case 'meter-to'" in html


def test_meter_to_rejects_missing_value_or_non_meter_target() -> None:
    with pytest.raises(ValueError, match="meter_value"):
        parse_string(
            '{"scenes":[{"id":"bad","duration":"1s","objects":['
            '{"id":"meter","type":"linear_meter"}],"animations":['
            '{"action":"meter-to","target":"meter"}]}]}',
            format="json",
        )

    with pytest.raises(ValueError, match="must target a linear_meter"):
        parse_string(
            '{"scenes":[{"id":"bad","duration":"1s","objects":['
            '{"id":"label","type":"text","content":"No"}],"animations":['
            '{"action":"meter-to","target":"label","meter_value":1}]}]}',
            format="json",
        )

    with pytest.raises(ValueError, match="only supported for meter-to"):
        parse_string(
            '{"scenes":[{"id":"bad","duration":"1s","objects":['
            '{"id":"meter","type":"linear_meter"}],"animations":['
            '{"action":"appear","target":"meter","meter_value":1}]}]}',
            format="json",
        )
