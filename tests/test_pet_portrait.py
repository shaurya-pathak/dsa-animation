from __future__ import annotations

import json

import pytest

from kaivra.dsl.parser import parse_string
from kaivra.dsl.schema import ObjectSpec, ObjectType, PetFeature, PetKind
from kaivra.layout.strategies._sizing import estimate_object_size
from kaivra.render.cairo_renderer import CairoRenderer
from kaivra.render.web.exporter import build_web_preview_html
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.themes.registry import get_theme


def _portrait_document(
    *,
    object_type: str = "pet_portrait",
    kind: str = "mystery",
    highlights: list[str] | None = None,
    show_feature_labels: bool = False,
):
    return parse_string(
        json.dumps(
            {
                "version": "1.5",
                "meta": {
                    "theme": "editorial",
                    "resolution": [800, 600],
                    "show_subtitles": False,
                },
                "scenes": [
                    {
                        "id": "portrait",
                        "duration": "2s",
                        "auto_visible": True,
                        "show_progress_bar": False,
                        "objects": [
                            {
                                "id": "pet",
                                "type": object_type,
                                "pet_kind": kind,
                                "content": "Dog or cat?",
                                "size_variant": "hero",
                                "pet_highlights": highlights or [],
                                "show_feature_labels": show_feature_labels,
                            }
                        ],
                    }
                ],
            }
        ),
        format="json",
    )


def test_pet_portrait_schema_builds_a_square_native_actor() -> None:
    doc = _portrait_document()
    schema = doc.model_json_schema()
    object_type_values = schema["$defs"]["ObjectType"]["enum"]

    assert "pet_portrait" in object_type_values
    assert "pet" in object_type_values
    assert "pet_kind" in schema["$defs"]["ObjectSpec"]["properties"]
    assert "pet_highlights" in schema["$defs"]["ObjectSpec"]["properties"]
    assert "show_feature_labels" in schema["$defs"]["ObjectSpec"]["properties"]

    graph = build_scene_graph(doc, get_theme("editorial"))
    node = graph.scenes[0].node_map["pet"]
    assert node.obj_type is ObjectType.PET_PORTRAIT
    assert node.pet_kind is PetKind.MYSTERY
    assert node.rect.width == node.rect.height == 297.0


def test_pet_alias_and_variant_sizing_are_supported() -> None:
    alias_doc = _portrait_document(object_type="pet", kind="cat")
    alias_node = build_scene_graph(alias_doc, get_theme("editorial")).scenes[0].node_map["pet"]
    assert alias_node.obj_type is ObjectType.PET
    assert alias_node.pet_kind is PetKind.CAT

    theme = get_theme("editorial")
    compact = estimate_object_size(
        ObjectSpec.model_validate({"type": "pet_portrait", "size_variant": "compact"}), theme
    )
    default = estimate_object_size(ObjectSpec.model_validate({"type": "pet_portrait"}), theme)
    hero = estimate_object_size(
        ObjectSpec.model_validate({"type": "pet_portrait", "size_variant": "hero"}), theme
    )
    assert (compact.width, compact.height) == pytest.approx((92.4, 92.4))
    assert (default.width, default.height) == pytest.approx((220.0, 220.0))
    assert (hero.width, hero.height) == pytest.approx((297.0, 297.0))


def test_pet_feature_highlights_are_semantic_and_reserve_label_space() -> None:
    doc = _portrait_document(
        highlights=["ears", "snout"],
        show_feature_labels=True,
    )
    node = build_scene_graph(doc, get_theme("editorial")).scenes[0].node_map["pet"]

    assert node.pet_highlights == [PetFeature.EARS, PetFeature.SNOUT]
    assert node.show_feature_labels is True
    # Labels stay inside the object's declared bounds, rather than colliding
    # with a neighboring layout object.
    assert (node.rect.width, node.rect.height) == pytest.approx((564.3, 297.0))

    with pytest.raises(ValueError, match="pet_highlights and show_feature_labels require"):
        ObjectSpec.model_validate({"type": "box", "pet_highlights": ["ears"]})


def test_pet_portrait_is_drawn_by_cairo_and_serialized_for_web() -> None:
    doc = _portrait_document(
        highlights=["ears", "snout"],
        show_feature_labels=True,
    )
    theme = get_theme("editorial")
    graph = build_scene_graph(doc, theme)
    pixels = CairoRenderer(theme).render_frame_to_bytes(graph, 0.5)

    # The portrait introduces several deterministic colors beyond the flat
    # editorial background, proving it went through the native Cairo dispatch.
    assert len(set(pixels)) > 20

    html = build_web_preview_html(doc)
    assert '"type": "pet_portrait"' in html
    assert '"petKind": "mystery"' in html
    assert '"petHighlights": ["ears", "snout"]' in html
    assert '"showFeatureLabels": true' in html
    assert '"channelCoral": "#A63D36"' in html
    assert '"channelCyan": "#267684"' in html
    assert "function drawPetPortrait(ctx, node)" in html
    assert "function floppyEarPath(direction)" in html
    assert "function catEarPath(direction)" in html
    assert "function drawFeatureLabel(text, color" in html
    assert "highlights.has('ears')" in html
    assert "highlights.has('snout')" in html
