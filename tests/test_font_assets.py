from __future__ import annotations

from importlib import resources
from pathlib import Path


def test_inter_font_assets_are_packaged_with_license() -> None:
    font_root = resources.files("kaivra.assets.fonts")

    assert (font_root / "InterVariable.woff2").is_file()
    assert (font_root / "InterVariable.ttf").is_file()
    license_text = (font_root / "LICENSE.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_text


def test_private_beta_render_image_installs_the_bundled_inter_font() -> None:
    dockerfile = Path(__file__).parents[1] / "docker" / "precommit.Dockerfile"
    contents = dockerfile.read_text(encoding="utf-8")

    assert "COPY src/kaivra/assets/fonts/InterVariable.ttf" in contents
    assert "fc-cache -f" in contents
