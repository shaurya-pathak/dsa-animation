from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaivra.dsl.parser import parse_file, parse_string
from kaivra.version import version_drift_warning

LEGACY_DSL_VERSION = "1.4"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "dsl_v15"


def _parse(document: dict) -> object:
    return parse_string(json.dumps(document), format="json")


@pytest.mark.parametrize(
    ("fixture_name", "expected_theme", "expected_bookends"),
    [
        ("unversioned_omitted_defaults.json", "editorial", False),
        ("v1_4_omitted_defaults.json", "whiteboard", True),
    ],
)
def test_visual_chrome_defaults_are_versioned(
    fixture_name: str,
    expected_theme: str,
    expected_bookends: bool,
) -> None:
    doc = parse_file(FIXTURE_ROOT / fixture_name)

    assert doc.meta.theme == expected_theme
    assert doc.meta.video_bookends is expected_bookends
    assert doc.meta.show_subtitles is expected_bookends
    assert all(scene.show_progress_bar is expected_bookends for scene in doc.scenes)


@pytest.mark.parametrize(
    ("fixture_name", "expected_bookends"),
    [
        ("v1_5_explicit_overrides.json", True),
        ("v1_4_explicit_overrides.json", False),
    ],
)
def test_explicit_theme_and_bookends_override_versioned_defaults(
    fixture_name: str, expected_bookends: bool
) -> None:
    doc = parse_file(FIXTURE_ROOT / fixture_name)

    assert doc.meta.theme == "material"
    assert doc.meta.video_bookends is expected_bookends


@pytest.mark.parametrize(
    ("version", "show_chrome"),
    [("1.5", True), ("1.4", False)],
)
def test_explicit_subtitles_and_progress_override_versioned_defaults(
    version: str, show_chrome: bool
) -> None:
    doc = _parse(
        {
            "version": version,
            "meta": {"show_subtitles": show_chrome},
            "scenes": [{"id": "explicit", "show_progress_bar": show_chrome}],
        }
    )

    assert doc.meta.show_subtitles is show_chrome
    assert doc.scenes[0].show_progress_bar is show_chrome


@pytest.mark.parametrize(
    ("version", "expected_theme", "expected_bookends"),
    [("1.5", "editorial", False), ("1.6", "editorial", False), ("1.3", "whiteboard", True)],
)
def test_versioned_defaults_apply_to_nonfixture_versions(
    version: str, expected_theme: str, expected_bookends: bool
) -> None:
    doc = _parse({"version": version, "scenes": []})

    assert doc.meta.theme == expected_theme
    assert doc.meta.video_bookends is expected_bookends


@pytest.mark.parametrize("version", [None, "1.5"])
def test_v15_rejects_misplaced_top_level_metadata(version: str | None) -> None:
    document: dict[str, object] = {"title": "Misplaced", "theme": "editorial", "scenes": []}
    if version is not None:
        document["version"] = version

    with pytest.raises(ValueError, match="Move `theme`, `title` under `meta`"):
        _parse(document)


def test_legacy_document_keeps_permissive_top_level_parsing() -> None:
    doc = _parse(
        {
            "version": LEGACY_DSL_VERSION,
            "title": "Legacy title",
            "theme": "modern",
            "scenes": [],
        }
    )

    assert doc.meta.theme == "whiteboard"
    assert doc.meta.video_bookends is True


def test_align_equals_is_opt_in() -> None:
    doc = _parse(
        {
            "version": "1.5",
            "scenes": [
                {
                    "objects": [
                        {"id": "default", "type": "text", "content": "a = b"},
                        {
                            "id": "aligned",
                            "type": "text",
                            "content": "c = d",
                            "align_equals": True,
                        },
                    ]
                }
            ],
        }
    )

    default, aligned = doc.scenes[0].objects
    assert default.align_equals is False
    assert aligned.align_equals is True


def test_flow_requires_a_connector_draw_that_it_follows() -> None:
    doc = _parse(
        {
            "version": "1.5",
            "scenes": [
                {
                    "id": "signal",
                    "objects": [
                        {"id": "source", "type": "box"},
                        {"id": "destination", "type": "box"},
                        {
                            "id": "path",
                            "type": "connector",
                            "from": "source",
                            "to": "destination",
                        },
                    ],
                    "animations": [
                        {"id": "draw_path", "action": "draw", "target": "path", "at": "1s"},
                        {
                            "id": "flow_path",
                            "action": "flow",
                            "target": "path",
                            "after": "draw_path",
                        },
                    ],
                }
            ],
        }
    )

    assert doc.scenes[0].animations[1].action.value == "flow"


def test_flow_rejects_a_non_connector_target() -> None:
    with pytest.raises(ValueError, match="must target a connector"):
        _parse(
            {
                "version": "1.5",
                "scenes": [
                    {
                        "objects": [{"id": "box", "type": "box"}],
                        "animations": [
                            {"id": "draw_box", "action": "draw", "target": "box", "at": "0s"},
                            {"action": "flow", "target": "box", "after": "draw_box"},
                        ],
                    }
                ],
            }
        )


def test_flow_rejects_an_overlapping_draw() -> None:
    with pytest.raises(ValueError, match="must follow the draw animation"):
        _parse(
            {
                "version": "1.5",
                "scenes": [
                    {
                        "objects": [
                            {"id": "source", "type": "box"},
                            {"id": "destination", "type": "box"},
                            {
                                "id": "path",
                                "type": "connector",
                                "from": "source",
                                "to": "destination",
                            },
                        ],
                        "animations": [
                            {
                                "id": "draw_path",
                                "action": "draw",
                                "target": "path",
                                "at": "1s",
                                "duration": "1s",
                            },
                            {"action": "flow", "target": "path", "at": "1.5s"},
                        ],
                    }
                ],
            }
        )


def test_version_drift_warning_distinguishes_legacy_and_future_documents() -> None:
    legacy_warning = version_drift_warning(LEGACY_DSL_VERSION)
    future_warning = version_drift_warning("1.6")

    assert legacy_warning is not None
    assert 'Update the "version" field to "1.5"' in legacy_warning
    assert version_drift_warning("1.5") is None
    assert future_warning is not None
    assert "newer schema" in future_warning
    assert 'Update the "version" field' not in future_warning
