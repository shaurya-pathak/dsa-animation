from __future__ import annotations

import json
from pathlib import Path

from kaivra.themes.editorial import EDITORIAL
from kaivra.themes.material import MATERIAL
from kaivra.themes.modern import MODERN
from kaivra.themes.registry import get_theme, load_theme_file, register_theme


def _contrast_ratio(first: str, second: str) -> float:
    def luminance(value: str) -> float:
        components = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4
            for component in components
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_material_theme_is_registered() -> None:
    theme = get_theme("material")

    assert theme is MATERIAL
    assert theme.accent == "#1976D2"
    assert theme.box_corner_radius == 16.0


def test_editorial_theme_is_flat_and_registered() -> None:
    theme = get_theme("editorial")

    assert theme is EDITORIAL
    assert theme.shadow is False
    assert theme.box_corner_radius == 0.0
    assert theme.font_family == "Inter Variable"
    assert theme.resolve_style("annotation")["font_family"] == "Inter Variable"
    assert theme.resolve_style("metric-warning")["font_size"] > theme.font_size_heading
    assert theme.resolve_style("metric-coral")["color"] == "#A63D36"
    assert theme.resolve_style("metric-cyan")["color"] == "#267684"
    assert theme.resolve_style("metric-gold")["color"] == "#846000"

    # Normal editorial text and explanatory channels meet the documented
    # 4.5:1 and 3:1 thresholds, respectively, against the editorial canvas.
    assert _contrast_ratio(theme.text_light, theme.background_color) >= 4.5
    for status_color in (theme.success, theme.warning, theme.error):
        assert _contrast_ratio(status_color, theme.background_color) >= 4.5
    for channel in (theme.channel_coral, theme.channel_cyan, theme.channel_gold, theme.muted):
        assert _contrast_ratio(channel, theme.background_color) >= 3.0


def test_register_theme_accepts_name_and_dict_data() -> None:
    name = "test-name-dict-theme"
    theme = register_theme(
        name,
        {
            **MODERN.to_dict(),
            "accent": "#14b8a6",
        },
    )

    assert theme.name == name
    assert get_theme(name).accent == "#14b8a6"


def test_register_theme_keeps_existing_themespec_signature() -> None:
    name = "test-themespec-theme"
    theme = MODERN.__class__(**{**MODERN.to_dict(), "name": name, "accent": "#ef4444"})

    registered = register_theme(theme)

    assert registered is theme
    assert get_theme(name).accent == "#ef4444"


def test_load_theme_file_registers_theme_from_json(tmp_path: Path) -> None:
    path = tmp_path / "mint.json"
    path.write_text(
        json.dumps(
            {
                **MODERN.to_dict(),
                "name": "test-file-theme",
                "accent": "#22c55e",
            }
        ),
        encoding="utf-8",
    )

    theme = load_theme_file(path)

    assert theme.name == "test-file-theme"
    assert get_theme("test-file-theme").accent == "#22c55e"
