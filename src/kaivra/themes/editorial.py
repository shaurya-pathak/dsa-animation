"""Restrained flat theme for content-first educational explainers.

Inter Variable is the editorial typeface. The private-beta image installs it for Cairo;
the self-contained web preview embeds the licensed WOFF2 and falls back through
an explicit browser font stack when an older preview lacks that asset.
"""

from kaivra.themes.base import ThemeSpec

EDITORIAL = ThemeSpec(
    name="editorial",
    background_color="#F7F4EC",
    # This matches the family metadata in the bundled InterVariable.ttf.
    font_family="Inter Variable",
    font_family_annotation="Inter Variable",
    font_size_heading=64,
    font_size_section_heading=42,
    font_size_body=28,
    font_size_caption=19,
    font_size_code=22,
    primary="#17202A",
    accent="#267684",
    success="#2E7D32",
    warning="#846000",
    error="#A63D36",
    muted="#747A82",
    text_color="#17202A",
    text_light="#59616A",
    channel_coral="#A63D36",
    channel_cyan="#267684",
    channel_gold="#846000",
    box_fill="#F7F4EC",
    box_border="#17202A",
    box_border_width=2.5,
    box_corner_radius=0.0,
    box_padding=14.0,
    box_min_width=118.0,
    box_min_height=72.0,
    token_fill="#F7F4EC",
    token_border="#17202A",
    token_padding=8.0,
    token_corner_radius=0.0,
    connector_color="#59616A",
    connector_width=3.0,
    arrow_size=11.0,
    gap_small=14.0,
    gap_medium=28.0,
    gap_large=58.0,
    margin=72.0,
    sketch_effect=False,
    shadow=False,
)
