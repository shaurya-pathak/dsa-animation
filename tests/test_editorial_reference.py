from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from kaivra.mcp.story_contract import validate_story_contract_markdown

REFERENCE_PATH = Path(__file__).parents[1] / "examples/reference/forward_propagation.json"
STORY_PATH = REFERENCE_PATH.with_suffix(".story.md")


def _objects(items: list[dict]) -> list[dict]:
    result: list[dict] = []
    for item in items:
        result.append(item)
        result.extend(_objects(item.get("children", [])))
    return result


def _scene(reference: dict, scene_id: str) -> dict:
    return next(scene for scene in reference["scenes"] if scene["id"] == scene_id)


def _scene_objects(reference: dict, scene_id: str) -> dict[str, dict]:
    return {item["id"]: item for item in _objects(_scene(reference, scene_id)["objects"])}


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _contains_phrase(tokens: list[str], phrase: str) -> bool:
    phrase_tokens = _normalized_tokens(phrase)
    return any(
        tokens[index : index + len(phrase_tokens)] == phrase_tokens
        for index in range(len(tokens) - len(phrase_tokens) + 1)
    )


def _animation(scene: dict, animation_id: str) -> dict:
    return next(item for item in scene["animations"] if item["id"] == animation_id)


def _assert_meter(item: dict, *, minimum: float, value: float, maximum: float) -> None:
    assert item["type"] == "linear_meter"
    assert item["meter_min"] == minimum
    assert item["meter_value"] == value
    assert item["meter_max"] == maximum


def test_reference_starts_from_a_complete_layperson_story_contract() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    story = STORY_PATH.read_text(encoding="utf-8")

    assert reference["version"] == "1.5"
    assert reference["meta"]["audience"] == "layperson"
    assert reference["meta"]["story_contract"] == STORY_PATH.name
    assert validate_story_contract_markdown(story) == ()
    assert "The goal is not to preserve a neural-network equation" in story
    assert "Do not open with “forward propagation.”" in story
    assert "no presentation chrome" in story


def test_reference_is_six_movements_in_one_visual_world() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    assert reference["objects"] == []
    assert [scene["id"] for scene in reference["scenes"]] == [
        "one_photo",
        "learned_earlier",
        "two_clue_paths",
        "one_lean",
        "friendlier_scale",
        "predict_the_change",
    ]
    assert all(scene.get("template") is None for scene in reference["scenes"])
    assert all(scene["show_progress_bar"] is False for scene in reference["scenes"])

    all_animations = [
        animation for scene in reference["scenes"] for animation in scene["animations"]
    ]
    assert not {animation["action"] for animation in all_animations} & {
        "pulse",
        "highlight",
        "bounce",
    }
    all_content = " ".join(
        item.get("content", "")
        for scene in reference["scenes"]
        for item in _objects(scene["objects"])
    )
    assert "→" not in all_content
    assert "ONE PHOTO" not in all_content
    assert "ONE GUESS" not in all_content
    assert "TECHNICAL NAME" not in all_content
    assert reference["meta"]["title"] == "How a Model Chooses Between a Dog and a Cat"

    all_narration = " ".join(scene["narration"] for scene in reference["scenes"]).casefold()
    for self_declarative_phrase in (
        "we'll walk through",
        "let's slow this down",
        "i'm going to show you",
        "we're about to see",
    ):
        assert self_declarative_phrase not in all_narration


def test_opening_leads_with_one_human_question() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    opening = _scene(reference, "one_photo")
    objects = _scene_objects(reference, "one_photo")

    assert "dog or a cat" in opening["narration"].casefold()
    assert "forward propagation" not in opening["narration"].casefold()
    assert objects["hero_pet"]["type"] == "pet_portrait"
    assert objects["hero_pet"]["size_variant"] == "hero"
    assert objects["dog_choice"]["content"] == "DOG"
    assert objects["cat_choice"]["content"] == "CAT"
    assert "opening_question" not in objects
    assert "we'll" not in opening["narration"].casefold()
    assert "slow" not in opening["narration"].casefold()
    assert _animation(opening, "reveal_pet")["at"] < _animation(opening, "reveal_choices")["at"]


def test_saved_rules_exist_before_current_photo_math() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    scene = _scene(reference, "learned_earlier")
    objects = _scene_objects(reference, "learned_earlier")

    _assert_meter(objects["saved_ear_rule"], minimum=0, value=0.5, maximum=1)
    _assert_meter(objects["saved_snout_rule"], minimum=0, value=0.55, maximum=1)
    assert "WEIGHT" in objects["saved_ear_rule"]["meter_caption"]
    assert "WEIGHT" in objects["saved_snout_rule"]["meter_caption"]
    assert "stay fixed" in scene["narration"].casefold()


def test_pet_visual_state_is_stable_across_saved_rules_to_clue_paths_cut() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    saved_pet = _scene_objects(reference, "learned_earlier")["hero_pet"]
    clue_pet = _scene_objects(reference, "two_clue_paths")["hero_pet"]

    for field in (
        "actor_id",
        "pet_kind",
        "pet_highlights",
        "show_feature_labels",
        "size_variant",
    ):
        assert saved_pet.get(field) == clue_pet.get(field)


def test_each_clue_visibly_travels_through_its_own_rule() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    scene = _scene(reference, "two_clue_paths")
    objects = _scene_objects(reference, "two_clue_paths")

    _assert_meter(objects["ears_input"], minimum=0, value=7, maximum=10)
    _assert_meter(objects["snout_input"], minimum=0, value=4, maximum=10)
    assert objects["ear_weight_value"]["content"] == "× 0.50"
    assert objects["ear_weight_label"]["content"] == "WEIGHT"
    assert objects["snout_weight_value"]["content"] == "× 0.55"
    assert objects["snout_weight_label"]["content"] == "WEIGHT"
    assert objects["ear_push"]["content"] == "+0.35"
    assert objects["snout_push"]["content"] == "+0.22"

    for draw_id, flow_id in (
        ("draw_pet_to_ears", "flow_pet_to_ears"),
        ("draw_ears_to_rule", "flow_ears_to_rule"),
        ("draw_ear_rule_to_push", "flow_ear_rule_to_push"),
        ("draw_pet_to_snout", "flow_pet_to_snout"),
        ("draw_snout_to_rule", "flow_snout_to_rule"),
        ("draw_snout_rule_to_push", "flow_snout_rule_to_push"),
    ):
        assert _animation(scene, flow_id)["after"] == draw_id


def test_two_pushes_converge_into_one_nonpercentage_lean() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    scene = _scene(reference, "one_lean")
    objects = _scene_objects(reference, "one_lean")

    assert objects["ear_push"]["content"] == "+0.35"
    assert objects["snout_push"]["content"] == "+0.22"
    _assert_meter(objects["combined_lean"], minimum=-1, value=0.57, maximum=1)
    assert "one answer, not two" in scene["narration"].casefold()
    assert objects["ear_push_to_lean"]["to"] == "combined_lean"
    assert objects["snout_push_to_lean"]["to"] == "combined_lean"
    assert _animation(scene, "reveal_combined_lean")["after"] == "flow_snout_push_to_lean"


def test_translation_changes_the_scale_without_adding_evidence() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    scene = _scene(reference, "friendlier_scale")
    objects = _scene_objects(reference, "friendlier_scale")

    _assert_meter(objects["lean_scale"], minimum=-1, value=0.57, maximum=1)
    assert objects["sigmoid_curve"]["type"] == "sigmoid_plot"
    assert objects["sigmoid_curve"]["sigmoid_input"] == pytest.approx(0.57)
    assert objects["sigmoid_curve"]["sigmoid_input_label"] == "+0.57"
    assert objects["sigmoid_curve"]["sigmoid_output_label"] == "64%"
    assert objects["chance_value"]["content"] == "64%"
    assert "evidence has not changed" in scene["narration"].casefold()
    assert (objects["lean_to_sigmoid"]["from"], objects["lean_to_sigmoid"]["to"]) == (
        "lean_scale",
        "sigmoid_curve",
    )
    assert _animation(scene, "draw_sigmoid_curve")["cue"] == "S-shaped curve"
    assert _animation(scene, "flow_sigmoid_to_chance")["after"] == "draw_sigmoid_to_chance"


def test_counterfactual_changes_evidence_while_the_rule_stays_fixed() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    scene = _scene(reference, "predict_the_change")
    objects = _scene_objects(reference, "predict_the_change")

    assert objects["ears_strong"]["grid"] == objects["ears_weak"]["grid"]
    assert objects["lean_before"]["grid"] == objects["lean_after"]["grid"]
    assert objects["fixed_ear_weight_value"]["content"] == "× 0.50"
    assert objects["fixed_ear_weight_label"]["content"] == "WEIGHT"
    assert objects["chance_before"]["content"] == "64%"
    assert objects["chance_after"]["content"] == "58%"
    assert objects["chance_label"]["content"] == "DOG"
    assert objects["counter_lean_to_chance"]["to"] == "chance_change"
    assert objects["technical_name"]["content"].endswith("FORWARD PROPAGATION")
    _assert_meter(objects["lean_before"], minimum=-1, value=0.57, maximum=1)
    _assert_meter(objects["lean_after"], minimum=-1, value=0.32, maximum=1)

    assert (
        _animation(scene, "replace_ears")["target"],
        _animation(scene, "replace_ears")["with"],
    ) == (
        "ears_strong",
        "ears_weak",
    )
    assert (
        _animation(scene, "replace_lean")["target"],
        _animation(scene, "replace_lean")["with"],
    ) == (
        "lean_before",
        "lean_after",
    )
    assert _animation(scene, "fade_out_old_chance")["target"] == "chance_before"
    assert _animation(scene, "fade_in_new_chance")["target"] == "chance_after"
    assert _animation(scene, "fade_in_new_chance")["after"] == "fade_out_old_chance"
    assert "ear weight stays the same" in scene["narration"].casefold()


def test_math_probability_and_timing_regressions() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    assert math.isclose(0.70 * 0.50, 0.35)
    assert math.isclose(0.40 * 0.55, 0.22)
    assert math.isclose(0.35 + 0.22, 0.57)
    assert round(100 / (1 + math.exp(-0.57))) == 64
    assert round(100 / (1 + math.exp(-0.32))) == 58

    for scene in reference["scenes"]:
        narration_tokens = _normalized_tokens(scene["narration"])
        duration_seconds = float(scene["duration"].removesuffix("s"))
        # Silent renders keep an authored fallback long enough for a careful
        # read. Voice renders ignore this estimate and fit each scene to the
        # provider's measured audio plus the configured lead and hold.
        assert len(narration_tokens) / 2.5 <= duration_seconds
        for animation in scene["animations"]:
            cue = animation.get("cue")
            if cue is None:
                continue
            assert animation.get("at")
            assert _contains_phrase(narration_tokens, cue), (
                f"{scene['id']}/{animation['id']} cue {cue!r} is not spoken contiguously"
            )
