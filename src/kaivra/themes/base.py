"""Theme specification and base class."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ThemeSpec:
    """Complete theme specification for rendering."""

    name: str

    # Canvas
    background_color: str = "#FFFDF7"

    # Typography
    font_family: str = "sans-serif"
    font_family_annotation: str = "monospace"
    font_size_heading: int = 48
    font_size_section_heading: int = 36
    font_size_body: int = 24
    font_size_caption: int = 18
    font_size_code: int = 20
    metric_sign_scale: float = 0.3
    metric_sign_gap: float = 8.0

    # Colors
    primary: str = "#2D3436"
    accent: str = "#0984E3"
    success: str = "#00B894"
    warning: str = "#FDCB6E"
    error: str = "#D63031"
    muted: str = "#B2BEC3"
    text_color: str = "#2D3436"
    text_light: str = "#636E72"

    # Neutral explanatory channels. These intentionally sit apart from status
    # styles (success/warning/error): a diagram path is not itself a status.
    channel_coral: str = "#A63D36"
    channel_cyan: str = "#267684"
    channel_gold: str = "#846000"

    # Box styling
    box_fill: str = "#FFFFFF"
    box_border: str = "#2D3436"
    box_border_width: float = 2.0
    box_corner_radius: float = 8.0
    box_padding: float = 16.0
    box_min_width: float = 120.0
    box_min_height: float = 50.0

    # Token styling
    token_fill: str = "#DFE6E9"
    token_border: str = "#636E72"
    token_padding: float = 8.0
    token_corner_radius: float = 4.0

    # Connector
    connector_color: str = "#636E72"
    connector_width: float = 2.0
    arrow_size: float = 10.0

    # Spacing / gaps
    gap_small: float = 12.0
    gap_medium: float = 24.0
    gap_large: float = 48.0

    # Layout margins
    margin: float = 60.0

    # Effects
    sketch_effect: bool = False
    sketch_roughness: float = 2.0
    shadow: bool = False
    shadow_offset: float = 4.0
    shadow_blur: float = 8.0
    shadow_color: str = "#00000033"

    def resolve_gap(self, gap: str) -> float:
        """Convert gap name to pixels."""
        gaps = {"small": self.gap_small, "medium": self.gap_medium, "large": self.gap_large}
        if gap in gaps:
            return gaps[gap]
        try:
            return float(gap)
        except ValueError:
            return self.gap_medium

    def resolve_style(self, style: str | None) -> dict:
        """Resolve a style name to rendering properties."""
        styles = {
            "hero-heading": {
                "font_size": round(self.font_size_heading * 1.35),
                "font_weight": "bold",
                "color": self.text_color,
            },
            "heading": {
                "font_size": self.font_size_heading,
                "font_weight": "bold",
                "color": self.text_color,
            },
            "section-heading": {
                "font_size": self.font_size_section_heading,
                "font_weight": "bold",
                "color": self.text_color,
            },
            "body": {
                "font_size": self.font_size_body,
                "color": self.text_color,
            },
            "caption": {
                "font_size": self.font_size_caption,
                "color": self.text_light,
            },
            "code": {
                "font_size": self.font_size_code,
                "font_family": "monospace",
                "color": self.text_color,
            },
            "metric": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.primary,
            },
            "metric-accent": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.accent,
            },
            "metric-success": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.success,
            },
            "metric-warning": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.warning,
            },
            "metric-error": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.error,
            },
            "metric-coral": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.channel_coral,
            },
            "metric-cyan": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.channel_cyan,
            },
            "metric-gold": {
                "font_size": round(self.font_size_heading * 1.9),
                "font_weight": "bold",
                "color": self.channel_gold,
            },
            "annotation": {
                "font_size": self.font_size_code,
                "font_family": self.font_family_annotation,
                "color": self.text_light,
            },
            "operator": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.primary,
            },
            "operator-accent": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.accent,
            },
            "operator-warning": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.warning,
            },
            "operator-error": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.error,
            },
            "operator-coral": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.channel_coral,
            },
            "operator-cyan": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.channel_cyan,
            },
            "operator-gold": {
                "font_size": round(self.font_size_heading * 0.82),
                "font_weight": "bold",
                "color": self.channel_gold,
            },
            "result": {
                "font_size": self.font_size_heading,
                "font_weight": "bold",
                "color": self.success,
            },
            "primary": {"fill": self.accent, "color": "#FFFFFF"},
            "accent": {"fill": self.accent, "border": self.accent},
            "muted": {"fill": self.muted, "color": self.text_light},
            "success": {"fill": self.success, "border": self.success, "color": "#FFFFFF"},
            "warning": {"fill": self.warning, "border": self.warning, "color": "#111827"},
            "error": {"fill": self.error, "border": self.error, "color": "#FFFFFF"},
            "dark": {"fill": self.primary, "border": self.primary, "color": "#FFFFFF"},
            "coral": {
                "fill": self.channel_coral,
                "border": self.channel_coral,
                "color": "#FFFFFF",
            },
            "cyan": {
                "fill": self.channel_cyan,
                "border": self.channel_cyan,
                "color": "#FFFFFF",
            },
            "gold": {
                "fill": self.channel_gold,
                "border": self.channel_gold,
                "color": "#FFFFFF",
            },
            "focus-coral": {
                "fill": self.channel_coral,
                "border": self.channel_coral,
                "color": "#FFFFFF",
                "font_size": self.font_size_section_heading,
                "font_weight": "bold",
            },
            "focus-cyan": {
                "fill": self.channel_cyan,
                "border": self.channel_cyan,
                "color": "#FFFFFF",
                "font_size": self.font_size_section_heading,
                "font_weight": "bold",
            },
            "focus-gold": {
                "fill": self.channel_gold,
                "border": self.channel_gold,
                "color": "#FFFFFF",
                "font_size": self.font_size_section_heading,
                "font_weight": "bold",
            },
            "focus-dark": {
                "fill": self.primary,
                "border": self.primary,
                "color": "#FFFFFF",
                "font_size": self.font_size_section_heading,
                "font_weight": "bold",
            },
        }
        return styles.get(style, {"font_size": self.font_size_body, "color": self.text_color})

    def resolve_color(self, color_name: str | None) -> str:
        """Resolve a named color to hex."""
        if color_name is None:
            return self.text_color
        color_map = {
            "primary": self.primary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "muted": self.muted,
            "coral": self.channel_coral,
            "cyan": self.channel_cyan,
            "gold": self.channel_gold,
        }
        return color_map.get(color_name, color_name)

    def to_dict(self) -> dict:
        """Serialize the theme to a JSON-friendly dict."""
        return asdict(self)
