from __future__ import annotations

import pytest

from kaivra.dsl.schema import DocumentSpec
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.themes.registry import get_theme


def _group_grid_graph(children: list[dict], layout: dict):
    doc = DocumentSpec.model_validate(
        {
            "version": "1.5",
            "meta": {"theme": "editorial", "show_subtitles": False},
            "scenes": [
                {
                    "id": "grid_group",
                    "duration": "3s",
                    "auto_visible": True,
                    "layout": "center",
                    "objects": [
                        {
                            "id": "state_grid",
                            "type": "group",
                            "layout": layout,
                            "children": children,
                        }
                    ],
                }
            ],
        }
    )
    return build_scene_graph(doc, get_theme("editorial"))


def test_group_grid_honors_explicit_cells_spans_and_same_cell_overlays() -> None:
    graph = _group_grid_graph(
        [
            {"id": "old_state", "type": "box", "content": "Old", "grid": {"row": 1, "col": 2}},
            {"id": "new_state", "type": "box", "content": "New", "grid": {"row": 1, "col": 2}},
            {
                "id": "wide_state",
                "type": "box",
                "content": "Wide",
                "grid": {"row": 2, "col": 1, "span": 2},
            },
            {
                "id": "tall_state",
                "type": "box",
                "content": "Tall",
                "grid": {"row": 1, "col": 3, "row_span": 2},
            },
        ],
        {"type": "grid", "columns": 3, "rows": 2, "gap": "small"},
    )
    scene = graph.scenes[0]
    group = scene.node_map["state_grid"].rect
    old_state = scene.node_map["old_state"].rect
    new_state = scene.node_map["new_state"].rect
    wide_state = scene.node_map["wide_state"].rect
    tall_state = scene.node_map["tall_state"].rect

    # Two authored visual states can now occupy the exact same semantic cell,
    # which gives a `replace` animation a stable causal location.
    assert (old_state.center.x, old_state.center.y) == pytest.approx(
        (new_state.center.x, new_state.center.y)
    )

    gap = get_theme("editorial").resolve_gap("small")
    cell_width = (group.width - gap * 2) / 3
    cell_height = (group.height - gap) / 2
    assert wide_state.center.x == pytest.approx(group.x + (cell_width * 2 + gap) / 2)
    assert wide_state.center.y == pytest.approx(group.y + cell_height + gap + cell_height / 2)
    assert tall_state.center.x == pytest.approx(group.x + 2 * (cell_width + gap) + cell_width / 2)
    assert tall_state.center.y == pytest.approx(group.y + (cell_height * 2 + gap) / 2)


def test_group_grid_keeps_row_major_defaults_for_children_without_grid_metadata() -> None:
    graph = _group_grid_graph(
        [
            {"id": "first", "type": "box", "content": "First"},
            {"id": "second", "type": "box", "content": "Second"},
            {"id": "third", "type": "box", "content": "Third"},
        ],
        {"type": "grid", "columns": 2, "rows": 2, "gap": "small"},
    )
    scene = graph.scenes[0]
    first = scene.node_map["first"].rect
    second = scene.node_map["second"].rect
    third = scene.node_map["third"].rect

    assert first.center.y == pytest.approx(second.center.y)
    assert first.center.x < second.center.x
    assert third.center.x == pytest.approx(first.center.x)
    assert third.center.y > first.center.y
