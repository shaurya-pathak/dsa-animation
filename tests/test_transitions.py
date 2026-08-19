from __future__ import annotations

import json

from kaivra.dsl.parser import parse_string
from kaivra.render.cairo_renderer import CairoRenderer
from kaivra.scene_graph.builder import build_scene_graph
from kaivra.themes.registry import get_theme
from kaivra.utils.color import hex_to_rgba


def test_fade_transition_uses_theme_background_without_scene_overlap() -> None:
    doc = parse_string(
        json.dumps(
            {
                "version": "1.5",
                "meta": {
                    "title": "Fade through background",
                    "theme": "editorial",
                    "resolution": [320, 180],
                    "video_bookends": False,
                },
                "scenes": [
                    {
                        "id": "outgoing",
                        "duration": "2s",
                        "transition": {"type": "fade", "duration": "1s"},
                        "objects": [
                            {
                                "id": "outgoing_title",
                                "type": "text",
                                "content": "OUTGOING",
                                "visible": True,
                            }
                        ],
                    },
                    {
                        "id": "incoming",
                        "duration": "2s",
                        "objects": [
                            {
                                "id": "incoming_title",
                                "type": "text",
                                "content": "INCOMING",
                                "visible": True,
                            }
                        ],
                    },
                ],
            }
        ),
        format="json",
    )
    theme = get_theme("editorial")
    graph = build_scene_graph(doc, theme)

    surface = CairoRenderer(theme).render_frame(graph, 2.0)
    surface.flush()

    expected_r, expected_g, expected_b, expected_a = (
        round(channel * 255) for channel in hex_to_rgba(theme.background_color)
    )
    data = bytes(surface.get_data())
    stride = surface.get_stride()
    for x, y in ((0, 0), (graph.width // 2, graph.height // 2)):
        offset = y * stride + x * 4
        blue, green, red, alpha = data[offset : offset + 4]
        assert (red, green, blue, alpha) == (
            expected_r,
            expected_g,
            expected_b,
            expected_a,
        )
