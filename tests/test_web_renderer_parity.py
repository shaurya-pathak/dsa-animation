from __future__ import annotations

import json

from kaivra.dsl.parser import parse_string
from kaivra.render.web.exporter import build_web_preview_html


def _parity_document():
    return parse_string(
        json.dumps(
            {
                "version": "1.5",
                "meta": {
                    "title": "Web renderer parity",
                    "theme": "editorial",
                    "video_bookends": False,
                },
                "scenes": [
                    {
                        "id": "parity",
                        "duration": "3s",
                        "layout": {"type": "flow", "direction": "horizontal"},
                        "objects": [
                            {
                                "id": "negative_metric",
                                "type": "text",
                                "content": "−0.40",
                                "style": "metric-coral",
                            },
                            {
                                "id": "positive_metric",
                                "type": "text",
                                "content": "0.70",
                                "style": "metric-cyan",
                            },
                            {
                                "id": "equation_a",
                                "type": "text",
                                "content": "× 0.50 = 0.35",
                                "style": "section-heading",
                                "align_equals": True,
                            },
                            {
                                "id": "equation_b",
                                "type": "text",
                                "content": "× −0.30 = 0.12",
                                "style": "section-heading",
                                "align_equals": True,
                            },
                            {
                                "id": "signal_path",
                                "type": "connector",
                                "from": "negative_metric",
                                "to": "positive_metric",
                                "style": "coral",
                            },
                        ],
                        "animations": [
                            {
                                "id": "draw_path",
                                "action": "draw",
                                "target": "signal_path",
                                "at": "0s",
                                "duration": "0.8s",
                            },
                            {
                                "id": "flow_path",
                                "action": "flow",
                                "target": "signal_path",
                                "after": "draw_path",
                                "duration": "0.8s",
                            },
                        ],
                    }
                ],
            }
        ),
        format="json",
    )


def test_web_preview_serializes_editorial_parity_features() -> None:
    html = build_web_preview_html(_parity_document())

    # The capture fixture covers the four v1.5 parity-sensitive constructs.
    assert '"alignEquals": true' in html
    assert '"action": "draw"' in html
    assert '"action": "flow"' in html
    assert '"style": "metric-coral"' in html

    # Canvas centering is based on actual glyph bounds, not a font-size guess.
    assert "function centeredTextBaseline(ctx, rect, text, fallbackFontSize)" in html
    assert "actualBoundingBoxAscent" in html
    assert "actualBoundingBoxDescent" in html
    assert "centeredTextBaseline(ctx, node.rect, magnitude, fontSize)" in html
    assert "centeredTextBaseline(ctx, node.rect, '=', fontSize)" in html

    # A completed draw has the same directional arrow geometry as Cairo before
    # a subsequent flow signal traverses the connector.
    assert "if (node._drawProgress >= 0.9)" in html
    assert "const arrowSize = THEME.arrowSize;" in html
    assert "Math.atan2(endY - sy, endX - sx)" in html
    assert "Math.cos(angle - 0.4)" in html
    assert "Math.cos(angle + 0.4)" in html


def test_web_preview_exposes_a_deterministic_capture_hook() -> None:
    html = build_web_preview_html(_parity_document())

    assert "window.__kaivraPreviewReady = false;" in html
    assert "function renderPreviewAt(time)" in html
    assert "const maxRenderableTime = Math.max(0, GRAPH.totalDuration - frameDuration);" in html
    assert "window.__kaivraRenderAt = renderPreviewAt;" in html
    assert "function setCanvasCaptureMode(enabled)" in html
    assert "const displayWidth = enabled ? GRAPH.width" in html
    assert "window.__kaivraSetCaptureMode = setCanvasCaptureMode;" in html
    assert "window.__kaivraCaptureAt = (time) => {" in html
    assert "function captureTimeFromQuery()" in html
    assert "new URLSearchParams(window.location.search).get('capture_time')" in html
    assert "document.body.classList.toggle('capture-mode', Boolean(enabled));" in html
    assert "window.__kaivraPreviewReady = true;" in html
