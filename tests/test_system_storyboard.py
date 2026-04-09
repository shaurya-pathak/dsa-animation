from __future__ import annotations

import json
from pathlib import Path

from kaivra.dsl.schema import AnimAction, DocumentSpec
from kaivra.mcp.blueprints import build_starter_document
from kaivra.mcp.resources import read_resource
from kaivra.mcp.workspace import KaivraWorkspace
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.themes.registry import get_theme


def _build_graph(doc_dict: dict) -> object:
    doc = DocumentSpec.model_validate(doc_dict)
    return build_scene_graph(doc, get_theme(doc.meta.theme))


def test_system_storyboard_pattern_supports_explicit_scene_kinds() -> None:
    doc = build_starter_document(
        title="Storyboard Demo",
        pattern="system_storyboard",
        beats=[
            {"title": "Intro", "detail": "Introduce the system.", "scene_kind": "title"},
            {
                "title": "Signals",
                "detail": "Many signals appear at once.",
                "scene_kind": "transform",
            },
            {"title": "Outcome", "detail": "A clear summary lands.", "scene_kind": "summary"},
        ],
        theme="storyboard_dark",
        audience=None,
        include_narration=True,
    )

    assert doc.meta.theme == "storyboard_dark"
    assert doc.scenes[1].template == "storyboard"
    transform_stage = next(
        obj for obj in doc.scenes[1].objects if obj.id == "storyboard_transform_stage"
    )
    dense_field = next(child for child in transform_stage.children if child.id == "transform_field")
    assert dense_field.layout is not None
    assert dense_field.layout.type.value == "grid"
    assert len(dense_field.children) == 100
    assert all(child.size_variant.value == "compact" for child in dense_field.children)


def test_actor_id_continuity_moves_actor_when_local_ids_change() -> None:
    graph = _build_graph(
        {
            "version": "1.4",
            "meta": {"theme": "storyboard_dark", "show_subtitles": False, "continuity": True},
            "scenes": [
                {
                    "id": "scene_a",
                    "duration": "3s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "slot_left",
                            "actor_id": "actor_main",
                            "type": "box",
                            "content": "Failure",
                            "grid": {"region": "stage"},
                        }
                    ],
                },
                {
                    "id": "scene_b",
                    "duration": "3s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "slot_right",
                            "actor_id": "actor_main",
                            "type": "box",
                            "content": "Failure",
                            "grid": {"region": "aside"},
                        }
                    ],
                },
            ],
        }
    )

    assert any(
        keyframe.action == AnimAction.MOVE and keyframe.target_id == "slot_right"
        for keyframe in graph.scenes[1].timeline
    )


def test_evolving_continuity_allows_moderate_copy_change() -> None:
    graph = _build_graph(
        {
            "version": "1.4",
            "meta": {"theme": "storyboard_dark", "show_subtitles": False, "continuity": True},
            "scenes": [
                {
                    "id": "scene_a",
                    "duration": "3s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "card_a",
                            "actor_id": "actor_main",
                            "type": "box",
                            "content": "Failure detected",
                            "continuity_mode": "evolving",
                            "grid": {"region": "stage"},
                        }
                    ],
                },
                {
                    "id": "scene_b",
                    "duration": "3s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "card_b",
                            "actor_id": "actor_main",
                            "type": "box",
                            "content": "Failure classification",
                            "continuity_mode": "evolving",
                            "grid": {"region": "aside"},
                        }
                    ],
                },
            ],
        }
    )

    assert any(
        keyframe.action == AnimAction.MOVE and keyframe.target_id == "card_b"
        for keyframe in graph.scenes[1].timeline
    )


def test_position_only_continuity_supports_dense_actor_copy_changes() -> None:
    graph = _build_graph(
        {
            "version": "1.4",
            "meta": {"theme": "storyboard_dark", "show_subtitles": False, "continuity": True},
            "scenes": [
                {
                    "id": "scene_a",
                    "duration": "3s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "dense_a",
                            "actor_id": "signal_001",
                            "type": "circle",
                            "content": "1",
                            "continuity_mode": "position_only",
                            "size_variant": "compact",
                            "grid": {"region": "stage"},
                        }
                    ],
                },
                {
                    "id": "scene_b",
                    "duration": "3s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "dense_b",
                            "actor_id": "signal_001",
                            "type": "circle",
                            "content": "A",
                            "continuity_mode": "position_only",
                            "size_variant": "compact",
                            "grid": {"region": "aside"},
                        }
                    ],
                },
            ],
        }
    )

    assert any(
        keyframe.action == AnimAction.MOVE and keyframe.target_id == "dense_b"
        for keyframe in graph.scenes[1].timeline
    )


def test_storyboard_stage_supports_a_dense_ten_by_ten_grid() -> None:
    dense_children = [
        {
            "id": f"node_{index:03d}",
            "actor_id": f"signal_{index:03d}",
            "type": "circle",
            "size_variant": "compact",
            "style": "error" if index < 30 else "muted",
        }
        for index in range(100)
    ]
    graph = _build_graph(
        {
            "version": "1.4",
            "meta": {"theme": "storyboard_dark", "show_subtitles": False},
            "scenes": [
                {
                    "id": "dense",
                    "duration": "4s",
                    "template": "storyboard",
                    "objects": [
                        {
                            "id": "dense_stage",
                            "type": "group",
                            "grid": {"region": "stage"},
                            "layout": {"type": "grid", "columns": 10, "rows": 10, "gap": "small"},
                            "children": dense_children,
                        }
                    ],
                }
            ],
        }
    )
    scene = graph.scenes[0]
    dense_nodes = [scene.node_map[f"node_{index:03d}"] for index in range(100)]

    assert len(dense_nodes) == 100
    assert all(node.rect.width <= 30 for node in dense_nodes)
    assert all(node.rect.height <= 30 for node in dense_nodes)
    assert all(node.rect.x >= 0 and node.rect.y >= 0 for node in dense_nodes)
    assert all(
        node.rect.right <= graph.width and node.rect.bottom <= graph.height for node in dense_nodes
    )


def test_resources_advertise_system_storyboard_and_storyboard_dark() -> None:
    pattern_catalog = read_resource("kaivra://pattern-catalog")["contents"][0]["text"]
    theme_catalog = read_resource("kaivra://theme-catalog")["contents"][0]["text"]

    assert "system_storyboard" in pattern_catalog
    assert "storyboard_dark" in theme_catalog


def test_storyboard_reference_examples_preview_and_render_png(tmp_path: Path) -> None:
    workspace = KaivraWorkspace(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    source_names = ("system_storyboard_demo.json", "qa_copilot_storyboard.json")

    for source_name in source_names:
        source_path = repo_root / "examples" / "reference" / source_name
        staged_path = tmp_path / "animations" / source_name
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

        checked = workspace.check_animation(file_path=str(staged_path))
        assert checked["valid"] is True, source_name

        previewed = workspace.preview_animation(file_path=str(staged_path))
        rendered = workspace.render_animation(file_path=str(staged_path), format="png")

        assert Path(previewed["html_path"]).exists(), source_name
        assert Path(previewed["preview_image_path"]).exists(), source_name
        assert Path(rendered["artifact_path"]).exists(), source_name
        assert Path(rendered["artifact_path"]).stat().st_size > 0, source_name


def test_qa_copilot_reference_example_covers_acceptance_storyboard_beats() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    qa_path = repo_root / "examples" / "reference" / "qa_copilot_storyboard.json"
    raw = json.loads(qa_path.read_text(encoding="utf-8"))
    scene_ids = {scene["id"] for scene in raw["scenes"]}

    assert scene_ids >= {
        "title_card",
        "scale_explosion",
        "bottleneck",
        "golden_rule",
        "agent_intro",
        "old_flow",
        "intercepted_flow",
        "inspection",
        "decision_engine",
        "action_layer",
        "aggregation",
        "bug_filing",
        "onboarding",
        "impact",
        "final_statement",
    }
