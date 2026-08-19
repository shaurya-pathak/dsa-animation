"""Web preview exporter — generates a self-contained HTML file with a Canvas-based player."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import webbrowser
from importlib import resources
from pathlib import Path

from kaivra.dsl.schema import DocumentSpec
from kaivra.dsl.timing import TimingConfig
from kaivra.render.orchestration import build_render_graph


def build_web_preview_html(
    doc: DocumentSpec,
    *,
    theme_search_roots: list[str | Path] | None = None,
    timing_config: TimingConfig | None = None,
) -> str:
    """Build the self-contained HTML preview for a document."""
    graph, theme = build_render_graph(
        doc,
        theme_search_roots=theme_search_roots,
        timing_config=timing_config,
    )

    # Serialize scene graph to JSON for the JS player
    scenes_data = []
    for scene in graph.scenes:
        nodes_data = []
        for node in scene.nodes:
            nodes_data.append(_serialize_node(node))

        timeline_data = []
        for kf in scene.timeline:
            timeline_data.append(
                {
                    "target_id": kf.target_id,
                    "action": kf.action.value,
                    "start_time": kf.start_time,
                    "duration": kf.duration,
                    "easing": kf.easing,
                    "to_value": kf.to_value,
                    "from_value": kf.from_value,
                    "style": kf.style,
                    "color": kf.color,
                    "stagger": kf.stagger,
                    "phases": kf.phases,
                    "to_id": kf.to_id,
                    "translate": kf.translate.model_dump(mode="json") if kf.translate else None,
                    "from_translate": (
                        kf.from_translate.model_dump(mode="json") if kf.from_translate else None
                    ),
                    "offset_x": kf.offset_x,
                    "offset_y": kf.offset_y,
                    "from_offset_x": kf.from_offset_x,
                    "from_offset_y": kf.from_offset_y,
                    "with_id": kf.with_id,
                }
            )

        scenes_data.append(
            {
                "id": scene.id,
                "duration": scene.duration,
                "nodes": nodes_data,
                "timeline": timeline_data,
                "transition": {
                    "type": scene.transition.type,
                    "duration": scene.transition.duration,
                }
                if scene.transition
                else None,
                "narration": scene.narration,
                "showProgressBar": scene.show_progress_bar,
            }
        )

    graph_json = json.dumps(
        {
            "width": graph.width,
            "height": graph.height,
            "fps": graph.fps,
            "theme": graph.theme_name,
            "totalDuration": graph.total_duration,
            "showNarration": graph.show_narration,
            "scenes": scenes_data,
        }
    )

    # Theme data for the JS renderer
    theme_json = json.dumps(
        {
            "backgroundColor": theme.background_color,
            "textColor": theme.text_color,
            "textLight": theme.text_light,
            "primary": theme.primary,
            "accent": theme.accent,
            "success": theme.success,
            "warning": theme.warning,
            "error": theme.error,
            "channelCoral": theme.channel_coral,
            "channelCyan": theme.channel_cyan,
            "channelGold": theme.channel_gold,
            "muted": theme.muted,
            "boxFill": theme.box_fill,
            "boxBorder": theme.box_border,
            "boxBorderWidth": theme.box_border_width,
            "boxCornerRadius": theme.box_corner_radius,
            "boxPadding": theme.box_padding,
            "tokenFill": theme.token_fill,
            "tokenBorder": theme.token_border,
            "connectorColor": theme.connector_color,
            "connectorWidth": theme.connector_width,
            "arrowSize": theme.arrow_size,
            "fontSizeHeading": theme.font_size_heading,
            "fontSizeSectionHeading": theme.font_size_section_heading,
            "fontSizeBody": theme.font_size_body,
            "fontSizeCaption": theme.font_size_caption,
            "fontFamily": theme.font_family,
            "metricSignScale": theme.metric_sign_scale,
            "metricSignGap": theme.metric_sign_gap,
        }
    )

    font_face_css = (
        _inter_font_face_css() if theme.font_family in {"Inter", "Inter Variable"} else ""
    )
    return _generate_html(graph_json, theme_json, font_face_css=font_face_css)


def write_web_preview(
    doc: DocumentSpec,
    path: str | Path,
    *,
    theme_search_roots: list[str | Path] | None = None,
    timing_config: TimingConfig | None = None,
) -> Path:
    """Write the self-contained HTML preview to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_web_preview_html(
            doc,
            theme_search_roots=theme_search_roots,
            timing_config=timing_config,
        ),
        encoding="utf-8",
    )
    return path


def export_web_preview(
    doc: DocumentSpec,
    *,
    serve: bool = False,
    port: int = 8080,
    theme_search_roots: list[str | Path] | None = None,
    timing_config: TimingConfig | None = None,
) -> None:
    """Export an HTML preview and optionally serve it."""
    html = build_web_preview_html(
        doc,
        theme_search_roots=theme_search_roots,
        timing_config=timing_config,
    )

    if serve:
        _serve_with_reload(html, port)
    else:
        # Write to temp file and open in browser
        tmpdir = tempfile.mkdtemp(prefix="kaivra-")
        path = os.path.join(tmpdir, "preview.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Preview saved to {path}")
        webbrowser.open(f"file://{path}")


def _serialize_node(node) -> dict:
    """Serialize a SceneNode to JSON-compatible dict."""
    data = {
        "id": node.id,
        "type": node.obj_type.value,
        "rect": {"x": node.rect.x, "y": node.rect.y, "w": node.rect.width, "h": node.rect.height},
        "content": node.content,
        "style": node.style,
        "stylePops": node.style_props,
        "persistent": node.persistent,
        "label": node.label,
        "tokenId": node.token_id,
        "petKind": node.pet_kind.value,
        "petHighlights": [feature.value for feature in node.pet_highlights],
        "showFeatureLabels": node.show_feature_labels,
        "iconName": node.icon_name,
        "meterValue": node.meter_value,
        "baseMeterValue": (
            node.base_meter_value if node.base_meter_value is not None else node.meter_value
        ),
        "meterMin": node.meter_min,
        "meterMax": node.meter_max,
        "meterLeftLabel": node.meter_left_label,
        "meterCenterLabel": node.meter_center_label,
        "meterRightLabel": node.meter_right_label,
        "meterValueLabel": node.meter_value_label,
        "meterCaption": node.meter_caption,
        "sigmoidInput": node.sigmoid_input,
        "sigmoidInputLabel": node.sigmoid_input_label,
        "sigmoidOutputLabel": node.sigmoid_output_label,
        "sigmoidCaption": node.sigmoid_caption,
        "fromId": node.from_id,
        "toId": node.to_id,
        "idlePreset": node.idle_preset,
        "idleIntensity": node.idle_intensity,
        "idleSpeed": node.idle_speed,
        "idleAxis": node.idle_axis,
        "defaultVisible": node.default_visible,
        "scaleText": node.scale_text,
        "baseScaleX": node.base_scale_x,
        "baseScaleY": node.base_scale_y,
        "layoutRole": node.layout_role,
        "alignEquals": getattr(node, "align_equals", False),
        "children": [_serialize_node(c) for c in node.children],
    }
    return data


def _inter_font_face_css() -> str:
    """Embed Inter so an exported preview remains a single portable HTML file."""
    try:
        font_bytes = (
            resources.files("kaivra.assets.fonts").joinpath("InterVariable.woff2").read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError):
        # Editable installs made before the font package was added remain
        # readable through the explicit CSS fallback stack below.
        return ""

    encoded = base64.b64encode(font_bytes).decode("ascii")
    return f"""  @font-face {{
    font-family: "Inter";
    src: url("data:font/woff2;base64,{encoded}") format("woff2");
    font-style: normal;
    font-weight: 100 900;
    font-display: block;
  }}
"""


def _generate_html(graph_json: str, theme_json: str, *, font_face_css: str = "") -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>kaivra Preview</title>
<style>
{font_face_css}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; font-family: "Inter", "Helvetica Neue", Arial, sans-serif; }}
  #container {{ position: relative; }}
  canvas {{ border-radius: 8px; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }}
  #controls {{ display: flex; align-items: center; gap: 12px; margin-top: 16px; color: #eee; }}
  button {{ background: #0984E3; color: white; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #0770c2; }}
  #timeline {{ width: 400px; accent-color: #0984E3; }}
  #time {{ font-variant-numeric: tabular-nums; min-width: 100px; text-align: center; }}
  #narration {{ color: #ccc; margin-top: 8px; font-style: italic; max-width: 600px; text-align: center; }}
  body.capture-mode {{ display: block; min-height: 0; background: transparent; }}
  body.capture-mode #controls, body.capture-mode #narration {{ display: none; }}
  body.capture-mode canvas {{ display: block; border-radius: 0; box-shadow: none; }}
</style>
</head>
<body>
<div id="container">
  <canvas id="canvas"></canvas>
</div>
<div id="controls">
  <button id="playBtn">Play</button>
  <input type="range" id="timeline" min="0" max="1000" value="0">
  <span id="time">0:00 / 0:00</span>
</div>
<div id="narration"></div>
<script>
const GRAPH = {graph_json};
const THEME = {theme_json};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
canvas.width = GRAPH.width;
canvas.height = GRAPH.height;

function setCanvasCaptureMode(enabled) {{
  // Interactive previews stay responsive. Evidence captures opt into a 1:1
  // CSS size so a canvas locator screenshot retains its authored resolution.
  const displayWidth = enabled ? GRAPH.width : Math.min(window.innerWidth - 40, 1200);
  const scale = displayWidth / GRAPH.width;
  canvas.style.width = (GRAPH.width * scale) + 'px';
  canvas.style.height = (GRAPH.height * scale) + 'px';
  document.body.classList.toggle('capture-mode', Boolean(enabled));
}}

setCanvasCaptureMode(false);
window.__kaivraSetCaptureMode = setCanvasCaptureMode;

let playing = false;
let currentTime = 0;
let sceneTime = 0;
let lastTimestamp = null;
// Deterministic-capture callers wait for this after the embedded font has loaded.
window.__kaivraPreviewReady = false;

const playBtn = document.getElementById('playBtn');
const timelineSlider = document.getElementById('timeline');
const timeDisplay = document.getElementById('time');
const narrationDiv = document.getElementById('narration');

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  if (playing) lastTimestamp = performance.now();
}});

timelineSlider.addEventListener('input', () => {{
  currentTime = (timelineSlider.value / 1000) * GRAPH.totalDuration;
  render(currentTime);
}});

function formatTime(s) {{
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}}

// Easing functions
const easings = {{
  'linear': t => t,
  'ease-in': t => t * t,
  'ease-out': t => 1 - (1 - t) * (1 - t),
  'ease-in-out': t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2,
  'spring': t => {{ const c4 = (2 * Math.PI) / 3; return t <= 0 ? 0 : t >= 1 ? 1 : -(Math.pow(2, 10 * t - 10)) * Math.sin((t * 10 - 10.75) * c4) + 1; }},
  'bounce': t => {{ const n1 = 7.5625, d1 = 2.75; if (t < 1/d1) return n1*t*t; if (t < 2/d1) {{ t -= 1.5/d1; return n1*t*t+0.75; }} if (t < 2.5/d1) {{ t -= 2.25/d1; return n1*t*t+0.9375; }} t -= 2.625/d1; return n1*t*t+0.984375; }},
}};

function resolveRelativeTranslate(spec, node, targetNode) {{
  if (!spec) return [0, 0];
  const basisNode = spec.basis === 'target' && targetNode ? targetNode : node;
  const width = basisNode.rect.w || 1;
  const height = basisNode.rect.h || 1;
  return [(spec.x || 0) * width, (spec.y || 0) * height];
}}

function getEasing(name) {{ return easings[name] || easings['ease-in-out']; }}

function getSceneAtTime(t) {{
  let elapsed = 0;
  for (let index = 0; index < GRAPH.scenes.length; index++) {{
    const scene = GRAPH.scenes[index];
    const localTime = t - elapsed;
    if (localTime < scene.duration) {{
      let sceneAlpha = 1;
      if (scene.transition && scene.transition.duration > 0 && index + 1 < GRAPH.scenes.length) {{
        const fadeSpan = scene.transition.duration / 2;
        const timeRemaining = scene.duration - localTime;
        if (fadeSpan > 0 && timeRemaining < fadeSpan) {{
          sceneAlpha = Math.max(0, Math.min(1, timeRemaining / fadeSpan));
        }}
      }}
      if (index > 0) {{
        const previousTransition = GRAPH.scenes[index - 1].transition;
        if (previousTransition && previousTransition.duration > 0) {{
          const fadeSpan = previousTransition.duration / 2;
          if (fadeSpan > 0 && localTime < fadeSpan) {{
            sceneAlpha = Math.min(sceneAlpha, Math.max(0, Math.min(1, localTime / fadeSpan)));
          }}
        }}
      }}
      return {{ scene, sceneIndex: index, localTime, sceneAlpha }};
    }}
    elapsed += scene.duration;
  }}
  return {{ scene: null, sceneIndex: -1, localTime: 0, sceneAlpha: 1 }};
}}

function getProgress(kf, t) {{
  if (t < kf.start_time) return null;
  if (kf.duration <= 0) return t >= kf.start_time ? 1 : null;
  const raw = (t - kf.start_time) / kf.duration;
  if (raw > 1) return null;
  return getEasing(kf.easing)(Math.max(0, Math.min(1, raw)));
}}

function highlightIntensity(progress) {{
  if (progress < 0.25) return progress / 0.25;
  if (progress > 0.75) return (1 - progress) / 0.25;
  return 1;
}}

function applyAnimations(nodeMap, keyframes, t) {{
  for (const id in nodeMap) {{
    if (nodeMap[id].persistent) {{
      nodeMap[id]._visible = true;
      nodeMap[id]._opacity = 1;
      nodeMap[id]._drawProgress = 1;
    }} else if (nodeMap[id].defaultVisible) {{
      nodeMap[id]._visible = true;
      nodeMap[id]._opacity = 1;
      nodeMap[id]._drawProgress = 1;
    }} else {{
      nodeMap[id]._visible = false;
      nodeMap[id]._opacity = 0;
      nodeMap[id]._drawProgress = 0;
    }}
    nodeMap[id]._scaleX = nodeMap[id].baseScaleX || 1;
    nodeMap[id]._scaleY = nodeMap[id].baseScaleY || 1;
    nodeMap[id]._translateX = 0;
    nodeMap[id]._translateY = 0;
    nodeMap[id]._highlightIntensity = 0;
    nodeMap[id]._flowProgress = null;
    nodeMap[id].meterValue = nodeMap[id].baseMeterValue;
  }}
  for (const kf of keyframes) {{
    const node = nodeMap[kf.target_id];
    if (!node) continue;
    const p = getProgress(kf, t);
    const done = t >= kf.start_time + kf.duration;

    switch(kf.action) {{
      case 'appear':
        if (kf.duration > 0) {{
          if (p !== null) {{ node._visible = true; node._opacity = Math.max(node._opacity, p); node._drawProgress = 1; }}
          else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        }} else if (t >= kf.start_time) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        break;
      case 'disappear':
        if (t < kf.start_time) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        else {{ node._visible = false; node._opacity = 0; }}
        break;
      case 'fade-in':
        if (p !== null) {{ node._visible = true; node._opacity = Math.max(node._opacity, p); node._drawProgress = 1; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        break;
      case 'fade-out':
        if (p !== null) {{ node._visible = true; node._opacity = 1 - p; node._drawProgress = 1; }}
        else if (done) {{ node._visible = false; node._opacity = 0; }}
        break;
      case 'type': case 'draw':
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = p; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        break;
      case 'flow':
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._flowProgress = p; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        break;
      case 'scale':
        const to = (kf.to_value !== undefined && kf.to_value !== null) ? kf.to_value : 1;
        const from = (kf.from_value !== undefined && kf.from_value !== null) ? kf.from_value : 1;
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; const s = from + (to - from) * p; node._scaleX = s; node._scaleY = s; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._scaleX = to; node._scaleY = to; }}
        break;
      case 'meter-to': {{
        const to = kf.to_value;
        const from = node.meterValue;
        if (to === undefined || to === null) break;
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node.meterValue = from + (to - from) * p; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node.meterValue = to; }}
        break;
      }}
      case 'move': {{
        let mdx, mdy, fdx, fdy;
        if (kf.translate || kf.from_translate) {{
          [mdx, mdy] = resolveRelativeTranslate(kf.translate, node, null);
          [fdx, fdy] = resolveRelativeTranslate(kf.from_translate, node, null);
        }} else {{
          mdx = kf.offset_x || 0; mdy = kf.offset_y || 0;
          fdx = kf.from_offset_x || 0; fdy = kf.from_offset_y || 0;
        }}
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._translateX = fdx + (mdx - fdx) * p; node._translateY = fdy + (mdy - fdy) * p; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._translateX = mdx; node._translateY = mdy; }}
        break;
      }}
      case 'move-to':
        const tnode = kf.to_id ? nodeMap[kf.to_id] : null;
        const [offsetDx, offsetDy] = kf.translate ? resolveRelativeTranslate(kf.translate, node, tnode) : [kf.offset_x || 0, kf.offset_y || 0];
        const dx = tnode ? (tnode.rect.x + tnode.rect.w/2 - (node.rect.x + node.rect.w/2)) + offsetDx : offsetDx;
        const dy = tnode ? (tnode.rect.y + tnode.rect.h/2 - (node.rect.y + node.rect.h/2)) + offsetDy : offsetDy;
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._translateX = dx * p; node._translateY = dy * p; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._translateX = dx; node._translateY = dy; }}
        break;
      case 'highlight': case 'pulse':
        if (p !== null) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; node._highlightIntensity = highlightIntensity(p); node._highlightColor = kf.color; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
        break;
      case 'build':
        if (kf.phases) {{
          for (const phase of kf.phases) {{
            const ps = parseFloat(phase.at);
            const pd = parseFloat(phase.duration);
            if (t >= ps) {{ node._visible = true; node._opacity = 1; node._drawProgress = pd > 0 ? Math.min(1, (t - ps) / pd) : 1; }}
          }}
        }}
        break;
      case 'replace': {{
        const replacement = kf.with_id ? nodeMap[kf.with_id] : null;
        if (p !== null) {{
          node._visible = true;
          node._opacity = Math.max(0, 1 - p);
          node._drawProgress = 1;
          if (replacement) {{
            replacement._visible = true;
            replacement._opacity = Math.max(replacement._opacity || 0, p);
            replacement._drawProgress = 1;
          }}
        }} else if (done) {{
          node._visible = false;
          node._opacity = 0;
          if (replacement) {{
            replacement._visible = true;
            replacement._opacity = 1;
            replacement._drawProgress = 1;
          }}
        }}
        break;
      }}
      default:
        if (p !== null) {{ node._visible = true; node._opacity = p; node._drawProgress = p; }}
        else if (done) {{ node._visible = true; node._opacity = 1; node._drawProgress = 1; }}
    }}
  }}
}}

function hexToRgba(hex, alpha) {{
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha !== undefined ? alpha : 1}})`;
}}

function cssFontFamily(name) {{
  const value = name || 'sans-serif';
  if (value === 'Inter' || value === 'Inter Variable') {{
    // Inter is embedded above for editorial exports. The ordered fallback is
    // intentionally explicit for older previews that lack the packaged asset.
    return '"Inter", "Helvetica Neue", Arial, sans-serif';
  }}
  if (value === 'monospace') {{
    return 'ui-monospace, "SFMono-Regular", Menlo, monospace';
  }}
  return value;
}}

function hexToRgbComponents(hex) {{
  const h = hex.replace('#', '');
  return [parseInt(h.substring(0, 2), 16), parseInt(h.substring(2, 4), 16), parseInt(h.substring(4, 6), 16)];
}}

function roundedRect(ctx, x, y, w, h, r) {{
  r = Math.min(r, w/2, h/2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}}

function drawNode(ctx, node, nodeMap) {{
  if (!node._visible) return;
  ctx.save();
  ctx.globalAlpha = node._opacity;

  const r = node.rect;
  let idleDx = 0, idleDy = 0, idleScale = 1;
  if (node.idlePreset) {{
    const preset = node.idlePreset;
    const speed = node.idleSpeed || 1.5;
    const intensity = (node.idleIntensity !== undefined && node.idleIntensity !== null)
      ? node.idleIntensity
      : (preset === 'breathe' ? 0.03 : 6);
    if (preset === 'float' || preset === 'jitter') {{
      const freq = preset === 'jitter' ? speed * 3.0 : speed;
      const axis = node.idleAxis || 'both';
      if (axis === 'x' || axis === 'both') idleDx = Math.sin(sceneTime * freq) * intensity;
      if (axis === 'y' || axis === 'both') idleDy = Math.cos(sceneTime * freq * 1.3) * intensity;
    }}
    if (preset === 'breathe') {{
      idleScale = 1 + Math.sin(sceneTime * speed) * intensity;
    }}
  }}

  const tx = (node._translateX || 0) + idleDx;
  const ty = (node._translateY || 0) + idleDy;
  if (tx || ty) {{
    ctx.translate(tx, ty);
  }}
  const sx = (node._scaleX || 1) * idleScale;
  const sy = (node._scaleY || 1) * idleScale;
  const shellOnlyScale = (node.scaleText === false) && (node.type === 'box' || node.type === 'token');
  if (shellOnlyScale && (sx !== 1 || sy !== 1)) {{
    ctx.save();
    const cx = r.x + r.w / 2, cy = r.y + r.h / 2;
    ctx.translate(cx, cy);
    ctx.scale(sx, sy);
    ctx.translate(-cx, -cy);
    drawNodeShell(ctx, node, nodeMap);
    if (node._highlightIntensity > 0) drawHighlight(ctx, node);
    ctx.restore();
    drawNodeTextLayer(ctx, node);
    ctx.restore();
    return;
  }}
  if (sx !== 1 || sy !== 1) {{
    const cx = r.x + r.w / 2, cy = r.y + r.h / 2;
    ctx.translate(cx, cy);
    ctx.scale(sx, sy);
    ctx.translate(-cx, -cy);
  }}

  drawNodeVisual(ctx, node, nodeMap);
  if (node._highlightIntensity > 0) drawHighlight(ctx, node);

  ctx.restore();
}}

function resolveThemeColor(name) {{
  if (!name) return THEME.accent;
  const mapped = THEME[name];
  return mapped || name;
}}

function drawHighlight(ctx, node) {{
  const intensity = (node._highlightIntensity || 0) * node._opacity;
  if (intensity <= 0) return;
  const color = resolveThemeColor(node._highlightColor);
  const [hr, hg, hb] = hexToRgbComponents(color);

  if (node.type === 'circle' || node.type === 'pet_portrait' || node.type === 'pet') {{
    const r = node.rect;
    const cx = r.x + r.w / 2;
    const cy = r.y + r.h / 2;
    const radius = Math.min(r.w, r.h) / 2;
    const glowRadius = radius * 2.8;
    const grad = ctx.createRadialGradient(cx, cy, radius * 0.8, cx, cy, glowRadius);
    grad.addColorStop(0.0, `rgba(${{hr}},${{hg}},${{hb}},${{intensity * 0.55}})`);
    grad.addColorStop(0.35, `rgba(${{hr}},${{hg}},${{hb}},${{intensity * 0.25}})`);
    grad.addColorStop(1.0, `rgba(${{hr}},${{hg}},${{hb}},0)`);
    ctx.beginPath();
    ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
  }} else {{
    const r = node.rect;
    roundedRect(
      ctx,
      r.x - 4,
      r.y - 4,
      r.w + 8,
      r.h + 8,
      THEME.boxCornerRadius + 4,
    );
    ctx.fillStyle = hexToRgba(color, intensity * 0.3);
    ctx.fill();
  }}
}}

function drawNodeVisual(ctx, node, nodeMap) {{
  switch(node.type) {{
    case 'text': drawText(ctx, node); break;
    case 'box': drawBox(ctx, node); break;
    case 'token': drawToken(ctx, node); break;
    case 'circle': drawCircle(ctx, node); break;
    case 'linear_meter': drawLinearMeter(ctx, node); break;
    case 'sigmoid_plot': drawSigmoidPlot(ctx, node); break;
    case 'pet_portrait': drawPetPortrait(ctx, node); break;
    case 'pet': drawPetPortrait(ctx, node); break;
    case 'semantic_icon': drawSemanticIcon(ctx, node); break;
    case 'connector': drawConnector(ctx, node, nodeMap); break;
    case 'group': drawGroup(ctx, node, nodeMap); break;
    default: drawBox(ctx, node); break;
  }}
}}

function drawNodeShell(ctx, node, nodeMap) {{
  switch(node.type) {{
    case 'box': drawBoxShell(ctx, node); break;
    case 'token': drawTokenShell(ctx, node); break;
    case 'circle': drawCircle(ctx, node); break;
    case 'pet_portrait': drawPetPortrait(ctx, node); break;
    case 'pet': drawPetPortrait(ctx, node); break;
    case 'semantic_icon': drawSemanticIcon(ctx, node); break;
    default: drawNodeVisual(ctx, node, nodeMap); break;
  }}
}}

function drawNodeTextLayer(ctx, node) {{
  switch(node.type) {{
    case 'box': drawBoxText(ctx, node); break;
    case 'token': drawTokenText(ctx, node); break;
  }}
}}

function drawText(ctx, node) {{
  if (!node.content) return;
  const sp = node.stylePops || {{}};
  const fontSize = sp.font_size || THEME.fontSizeBody;
  const fontFamily = cssFontFamily(sp.font_family || THEME.fontFamily);
  const color = sp.color || THEME.textColor;
  const weight = sp.font_weight === 'bold' ? 'bold' : 'normal';
  ctx.font = `${{weight}} ${{fontSize}}px ${{fontFamily}}`;
  ctx.fillStyle = hexToRgba(color, node._opacity);
  let text = node.content;
  const isMetric = isMetricStyle(node.style);
  const signedMetric = splitMetricSign(text, node.style);
  if (isMetric) {{
    const progress = node._drawProgress == null ? 1 : node._drawProgress;
    const charsToShow = Math.floor(text.length * progress);
    if (charsToShow <= 0) return;
    const magnitude = signedMetric
      ? signedMetric.magnitude.substring(0, Math.max(0, charsToShow - 1))
      : text.substring(0, charsToShow);
    drawMetricText(
      ctx,
      node,
      signedMetric ? signedMetric.sign : null,
      magnitude,
      fontSize,
      fontFamily,
      weight,
    );
    return;
  }}
  if (node.alignEquals && splitEquation(node.content)) {{
    const equation = alignedEquationReveal(node.content, node._drawProgress == null ? 1 : node._drawProgress);
    if (equation) {{
      drawEquationText(ctx, node, equation.left, equation.showEquals, equation.right, fontSize);
      return;
    }}
  }}
  if (node._drawProgress < 1) text = text.substring(0, Math.floor(text.length * node._drawProgress));
  if (!text) {{
    return;
  }}
  const m = ctx.measureText(text);
  ctx.fillText(
    text,
    node.rect.x + (node.rect.w - m.width) / 2,
    centeredTextBaseline(ctx, node.rect, text, fontSize),
  );
}}

function centeredTextBaseline(ctx, rect, text, fallbackFontSize) {{
  // Canvas and Cairo expose different text APIs, but both can center the ink
  // box rather than guessing from a font-size multiplier. This also keeps a
  // metric's small sign centered on its magnitude across renderers.
  const metrics = ctx.measureText(text || ' ');
  const ascent = metrics.actualBoundingBoxAscent;
  const descent = metrics.actualBoundingBoxDescent;
  if (Number.isFinite(ascent) && Number.isFinite(descent) && (ascent || descent)) {{
    return rect.y + rect.h / 2 + (ascent - descent) / 2;
  }}
  // The fallback supports older canvas implementations that do not expose
  // actual bounding boxes while retaining the pre-1.5 placement behavior.
  return rect.y + rect.h / 2 + fallbackFontSize * 0.35;
}}

function isMetricStyle(style) {{
  return Boolean(style && style.startsWith('metric'));
}}

function splitMetricSign(content, style) {{
  if (!isMetricStyle(style) || !content || content.length < 2) return null;
  const rawSign = content[0];
  if (!['+', '-', '−'].includes(rawSign)) return null;
  return {{sign: rawSign === '-' ? '−' : rawSign, magnitude: content.slice(1)}};
}}

function splitEquation(content) {{
  if (!content || content.split('=').length !== 2) return null;
  const [left, right] = content.split('=');
  if (!left.trim() || !right.trim()) return null;
  return {{left: left.trim(), right: right.trim()}};
}}

function alignedEquationReveal(content, progress) {{
  if (!splitEquation(content)) return null;
  const visibleChars = Math.floor(content.length * Math.max(0, Math.min(1, progress)));
  const equalsIndex = content.indexOf('=');
  const left = content.substring(0, Math.min(visibleChars, equalsIndex)).replace(/\\s+$/, '');
  const showEquals = visibleChars > equalsIndex;
  const rightChars = Math.max(0, visibleChars - equalsIndex - 1);
  const right = content.substring(equalsIndex + 1, equalsIndex + 1 + rightChars).trim();
  return {{left, showEquals, right}};
}}

function drawEquationText(ctx, node, left, showEquals, right, fontSize) {{
  const equalsWidth = ctx.measureText('=').width;
  const leftWidth = ctx.measureText(left).width;
  const gap = Math.max(8, fontSize * 0.28);
  const equalsX = node.rect.x + node.rect.w / 2 - equalsWidth / 2;
  const baselineY = centeredTextBaseline(ctx, node.rect, '=', fontSize);
  if (left) ctx.fillText(left, equalsX - gap - leftWidth, baselineY);
  if (showEquals) ctx.fillText('=', equalsX, baselineY);
  if (right) ctx.fillText(right, equalsX + equalsWidth + gap, baselineY);
}}

function metricSignSlotWidth(fontSize) {{
  return THEME.metricSignGap + fontSize * THEME.metricSignScale * 0.70;
}}

function drawMetricText(ctx, node, sign, magnitude, fontSize, fontFamily, weight) {{
  ctx.font = `${{weight}} ${{fontSize}}px ${{fontFamily}}`;
  const magnitudeWidth = ctx.measureText(magnitude).width;
  const signSlot = metricSignSlotWidth(fontSize);
  const magnitudeRegionX = node.rect.x + signSlot;
  const magnitudeRegionWidth = Math.max(0, node.rect.w - signSlot);
  const magnitudeX = magnitudeRegionX + (magnitudeRegionWidth - magnitudeWidth) / 2;
  const baselineY = centeredTextBaseline(ctx, node.rect, magnitude, fontSize);
  if (!sign) {{
    ctx.fillText(magnitude, magnitudeX, baselineY);
    return;
  }}
  const signSize = fontSize * THEME.metricSignScale;
  ctx.font = `${{weight}} ${{signSize}}px ${{fontFamily}}`;
  const signWidth = ctx.measureText(sign).width;
  const signBaselineY = centeredTextBaseline(ctx, node.rect, sign, signSize);
  const signX = Math.max(node.rect.x, magnitudeRegionX - THEME.metricSignGap - signWidth);
  ctx.fillText(sign, signX, signBaselineY);
  ctx.font = `${{weight}} ${{fontSize}}px ${{fontFamily}}`;
  ctx.fillText(magnitude, magnitudeX, baselineY);
}}

function drawBox(ctx, node) {{
  drawBoxShell(ctx, node);
  drawBoxText(ctx, node);
}}

function drawBoxShell(ctx, node) {{
  const r = node.rect;
  const sp = node.stylePops || {{}};
  const fill = sp.fill || THEME.boxFill;
  const border = sp.border || THEME.boxBorder;
  const strokeScale = sp.size_variant === 'compact' ? 0.72 : (sp.size_variant === 'hero' ? 1.15 : 1.0);
  roundedRect(ctx, r.x, r.y, r.w, r.h, THEME.boxCornerRadius);
  ctx.fillStyle = hexToRgba(fill, node._opacity);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(border, node._opacity);
  ctx.lineWidth = THEME.boxBorderWidth * strokeScale;
  ctx.stroke();
}}

function drawBoxText(ctx, node) {{
  const r = node.rect;
  if (node.content) {{
    const sp = node.stylePops || {{}};
    const fontSize = sp.font_size || THEME.fontSizeBody;
    const color = sp.color || THEME.textColor;
    const weight = sp.font_weight === 'bold' ? '700' : '400';
    ctx.font = `${{weight}} ${{fontSize}}px ${{cssFontFamily(sp.font_family || THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(color, node._opacity);
    let text = node.content;
    if (node._drawProgress < 1) text = text.substring(0, Math.floor(text.length * node._drawProgress));
    const m = ctx.measureText(text);
    ctx.fillText(text, r.x + (r.w - m.width) / 2, centeredTextBaseline(ctx, r, text, fontSize));
  }}
}}

function drawToken(ctx, node) {{
  drawTokenShell(ctx, node);
  drawTokenText(ctx, node);
}}

function drawTokenShell(ctx, node) {{
  const r = node.rect;
  const sp = node.stylePops || {{}};
  const fill = sp.fill || THEME.tokenFill;
  const border = sp.border || THEME.tokenBorder;
  const strokeScale = sp.size_variant === 'compact' ? 0.72 : (sp.size_variant === 'hero' ? 1.15 : 1.0);
  roundedRect(ctx, r.x, r.y, r.w, r.h, 4);
  ctx.fillStyle = hexToRgba(fill, node._opacity);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(border, node._opacity);
  ctx.lineWidth = 1.5 * strokeScale;
  ctx.stroke();
}}

function drawTokenText(ctx, node) {{
  const r = node.rect;
  if (node.content) {{
    const sp = node.stylePops || {{}};
    const fontSize = sp.font_size || THEME.fontSizeBody;
    const color = sp.color || THEME.textColor;
    const weight = sp.font_weight === 'bold' ? '700' : '400';
    ctx.font = `${{weight}} ${{fontSize}}px ${{cssFontFamily(sp.font_family || THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(color, node._opacity);
    const text = node.content.trim();
    const m = ctx.measureText(text);
    ctx.fillText(text, r.x + (r.w - m.width) / 2, centeredTextBaseline(ctx, r, text, fontSize));
  }}
  if (node.tokenId != null) {{
    ctx.font = `12px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(THEME.textLight, node._opacity * 0.8);
    const tid = String(node.tokenId);
    const m = ctx.measureText(tid);
    ctx.fillText(tid, r.x + (r.w - m.width) / 2, r.y + r.h + 14);
  }}
}}

function drawCircle(ctx, node) {{
  const cx = node.rect.x + node.rect.w / 2;
  const cy = node.rect.y + node.rect.h / 2;
  const radius = Math.min(node.rect.w, node.rect.h) / 2;
  const sp = node.stylePops || {{}};
  const fill = sp.fill || THEME.boxFill;
  const border = sp.border || THEME.boxBorder;
  const color = sp.color || THEME.textColor;
  const fontSize = sp.font_size || THEME.fontSizeBody;
  const strokeScale = sp.size_variant === 'compact' ? 0.72 : (sp.size_variant === 'hero' ? 1.15 : 1.0);
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(fill, node._opacity);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(border, node._opacity);
  ctx.lineWidth = THEME.boxBorderWidth * strokeScale;
  ctx.stroke();
  if (node.content) {{
    const weight = sp.font_weight === 'bold' ? '700' : '400';
    ctx.font = `${{weight}} ${{fontSize}}px ${{cssFontFamily(sp.font_family || THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(color, node._opacity);
    const m = ctx.measureText(node.content);
    ctx.fillText(
      node.content,
      cx - m.width / 2,
      centeredTextBaseline(ctx, node.rect, node.content, fontSize),
    );
  }}
}}

function drawLinearMeter(ctx, node) {{
  // Keep the geometry semantic: authors declare a range and interpretation,
  // while both renderers derive the same bounded rail, fill, and pointer.
  const r = node.rect;
  const sp = node.stylePops || {{}};
  const variant = sp.size_variant || 'default';
  const variantScale = variant === 'compact' ? 0.78 : (variant === 'hero' ? 1.16 : 1.0);
  let meterColor = sp.fill || sp.border;
  if (!meterColor && node.style) meterColor = sp.color;
  meterColor = meterColor || THEME.accent;

  const meterMin = node.meterMin == null ? 0 : node.meterMin;
  const meterMax = node.meterMax == null ? 100 : node.meterMax;
  const meterValue = node.meterValue == null ? 0 : node.meterValue;
  const span = meterMax - meterMin;
  const ratio = Math.max(0, Math.min(1, (meterValue - meterMin) / span));

  const captionFont = Math.max(12, Math.min(22 * variantScale, r.h * 0.19));
  const valueFont = Math.max(12, Math.min(23 * variantScale, r.h * 0.21));
  const labelFont = Math.max(11, Math.min(18 * variantScale, r.h * 0.16));
  const hasLabels = Boolean(node.meterLeftLabel || node.meterCenterLabel || node.meterRightLabel);
  const paddingX = Math.max(14 * variantScale, r.w * 0.055);
  const paddingY = Math.max(6 * variantScale, r.h * 0.055);
  const captionSpace = node.meterCaption ? captionFont * 1.38 : 0;
  const valueSpace = node.meterValueLabel ? valueFont * 1.42 : 0;
  const labelsSpace = hasLabels ? labelFont * 1.42 : 0;
  const freeHeight = Math.max(8, r.h - (paddingY * 2 + captionSpace + valueSpace + labelsSpace));
  const trackHeight = Math.max(8 * variantScale, Math.min(20 * variantScale, freeHeight * 0.48));
  const trackX = r.x + paddingX;
  const trackWidth = Math.max(trackHeight * 2, r.w - paddingX * 2);
  const trackY = r.y + paddingY + captionSpace + valueSpace + (freeHeight - trackHeight) / 2;
  const trackCenterY = trackY + trackHeight / 2;
  const pointerX = trackX + trackWidth * ratio;

  function drawText(text, x, baselineY, align, color) {{
    const width = ctx.measureText(text).width;
    const textX = align === 'left' ? x : (align === 'right' ? x - width : x - width / 2);
    ctx.fillStyle = hexToRgba(color, node._opacity);
    ctx.fillText(text, textX, baselineY);
  }}

  roundedRect(ctx, trackX, trackY, trackWidth, trackHeight, trackHeight / 2);
  ctx.fillStyle = hexToRgba(THEME.muted, node._opacity * 0.22);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(THEME.primary, node._opacity * 0.36);
  ctx.lineWidth = Math.max(1, THEME.boxBorderWidth * 0.62 * variantScale);
  ctx.stroke();

  const fillWidth = trackWidth * ratio;
  if (fillWidth > 0.5) {{
    roundedRect(ctx, trackX, trackY, fillWidth, trackHeight, Math.min(trackHeight / 2, fillWidth / 2));
    ctx.fillStyle = hexToRgba(meterColor, node._opacity * 0.64);
    ctx.fill();
  }}

  const tickHeight = trackHeight * 1.28;
  ctx.strokeStyle = hexToRgba(THEME.primary, node._opacity * 0.44);
  ctx.lineWidth = Math.max(1, THEME.boxBorderWidth * 0.48 * variantScale);
  for (const tickX of [trackX, trackX + trackWidth / 2, trackX + trackWidth]) {{
    ctx.beginPath();
    ctx.moveTo(tickX, trackCenterY - tickHeight / 2);
    ctx.lineTo(tickX, trackCenterY + tickHeight / 2);
    ctx.stroke();
  }}

  const pointerRadius = Math.max(6 * variantScale, trackHeight * 0.72);
  ctx.beginPath();
  ctx.arc(pointerX, trackCenterY, pointerRadius, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(THEME.backgroundColor, node._opacity);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(meterColor, node._opacity);
  ctx.lineWidth = Math.max(1.6, THEME.boxBorderWidth * variantScale);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(pointerX, trackCenterY, pointerRadius * 0.38, 0, Math.PI * 2);
  ctx.fillStyle = hexToRgba(meterColor, node._opacity);
  ctx.fill();

  const fontFamily = cssFontFamily(sp.font_family || THEME.fontFamily);
  if (node.meterCaption) {{
    ctx.font = `bold ${{captionFont}}px ${{fontFamily}}`;
    const captionRect = {{x: r.x, y: r.y + paddingY, w: r.w, h: captionFont}};
    drawText(
      node.meterCaption,
      r.x + r.w / 2,
      centeredTextBaseline(ctx, captionRect, node.meterCaption, captionFont),
      'center',
      THEME.textColor,
    );
  }}

  if (node.meterValueLabel) {{
    ctx.font = `bold ${{valueFont}}px ${{fontFamily}}`;
    const valueWidth = ctx.measureText(node.meterValueLabel).width;
    const valueX = Math.min(
      Math.max(pointerX, trackX + valueWidth / 2),
      trackX + trackWidth - valueWidth / 2,
    );
    drawText(node.meterValueLabel, valueX, trackY - valueFont * 0.28, 'center', meterColor);
  }}

  if (hasLabels) {{
    ctx.font = `400 ${{labelFont}}px ${{fontFamily}}`;
    const labelBaseline = trackY + trackHeight + labelFont * 1.18;
    if (node.meterLeftLabel) {{
      drawText(node.meterLeftLabel, trackX, labelBaseline, 'left', THEME.textLight);
    }}
    if (node.meterCenterLabel) {{
      drawText(node.meterCenterLabel, trackX + trackWidth / 2, labelBaseline, 'center', THEME.textLight);
    }}
    if (node.meterRightLabel) {{
      drawText(node.meterRightLabel, trackX + trackWidth, labelBaseline, 'right', THEME.textLight);
    }}
  }}
}}

function drawSigmoidPlot(ctx, node) {{
  const r = node.rect;
  const sp = node.stylePops || {{}};
  const variant = sp.size_variant || 'default';
  const variantScale = variant === 'compact' ? 0.82 : (variant === 'hero' ? 1.14 : 1.0);
  const left = r.x + Math.max(42, r.w * 0.13);
  const right = r.x + r.w - Math.max(18, r.w * 0.055);
  const top = r.y + Math.max(38, r.h * 0.17);
  const bottom = r.y + r.h - Math.max(40, r.h * 0.17);
  const plotWidth = Math.max(1, right - left);
  const plotHeight = Math.max(1, bottom - top);
  const axisX = left + plotWidth / 2;
  const midY = top + plotHeight / 2;
  const rawInput = node.sigmoidInput == null ? 0 : node.sigmoidInput;
  const inputValue = Math.max(-4, Math.min(4, rawInput));
  const probability = 1 / (1 + Math.exp(-inputValue));
  const pointX = left + ((inputValue + 4) / 8) * plotWidth;
  const pointY = bottom - probability * plotHeight;

  function drawLabel(text, x, y, align = 'center') {{
    const width = ctx.measureText(text).width;
    const textX = align === 'left' ? x : (align === 'right' ? x - width : x - width / 2);
    ctx.fillText(text, textX, y);
  }}

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = hexToRgba(THEME.muted, node._opacity * 0.34);
  ctx.lineWidth = Math.max(1, THEME.boxBorderWidth * 0.48 * variantScale);
  ctx.setLineDash([5 * variantScale, 6 * variantScale]);
  ctx.beginPath();
  ctx.moveTo(left, midY);
  ctx.lineTo(right, midY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = hexToRgba(THEME.primary, node._opacity * 0.68);
  ctx.lineWidth = Math.max(1.2, THEME.boxBorderWidth * 0.62 * variantScale);
  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.moveTo(axisX, top);
  ctx.lineTo(axisX, bottom);
  ctx.stroke();

  const sampleCount = 96;
  const progress = Math.max(0, Math.min(1, node._drawProgress == null ? 1 : node._drawProgress));
  const visibleSamples = Math.max(2, Math.round(sampleCount * progress));
  ctx.strokeStyle = hexToRgba(THEME.channelGold, node._opacity);
  ctx.lineWidth = Math.max(3, THEME.connectorWidth * 1.35 * variantScale);
  ctx.beginPath();
  for (let index = 0; index < visibleSamples; index += 1) {{
    const xValue = -4 + 8 * index / (sampleCount - 1);
    const yValue = 1 / (1 + Math.exp(-xValue));
    const x = left + ((xValue + 4) / 8) * plotWidth;
    const y = bottom - yValue * plotHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  const captionFont = Math.max(13, Math.min(20 * variantScale, r.h * 0.08));
  const labelFont = Math.max(11, Math.min(17 * variantScale, r.h * 0.068));
  const fontFamily = cssFontFamily(sp.font_family || THEME.fontFamily);
  ctx.font = `bold ${{captionFont}}px ${{fontFamily}}`;
  ctx.fillStyle = hexToRgba(THEME.textColor, node._opacity);
  if (node.sigmoidCaption) drawLabel(node.sigmoidCaption, r.x + r.w / 2, r.y + captionFont);

  ctx.font = `bold ${{labelFont}}px ${{fontFamily}}`;
  ctx.fillStyle = hexToRgba(THEME.textLight, node._opacity);
  drawLabel('SCORE', right, r.y + r.h - 8, 'right');
  drawLabel('CHANCE', left, top - 8, 'left');
  drawLabel('50%', left - 8, midY + labelFont * 0.35, 'right');

  if (progress >= 0.9) {{
    ctx.strokeStyle = hexToRgba(THEME.channelGold, node._opacity * 0.62);
    ctx.lineWidth = Math.max(1.2, THEME.boxBorderWidth * 0.55 * variantScale);
    ctx.setLineDash([4 * variantScale, 5 * variantScale]);
    ctx.beginPath();
    ctx.moveTo(pointX, bottom);
    ctx.lineTo(pointX, pointY);
    ctx.lineTo(left, pointY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(pointX, pointY, 7 * variantScale, 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(THEME.backgroundColor, node._opacity);
    ctx.fill();
    ctx.strokeStyle = hexToRgba(THEME.channelGold, node._opacity);
    ctx.lineWidth = Math.max(2, THEME.boxBorderWidth * variantScale);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(pointX, pointY, 2.7 * variantScale, 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(THEME.channelGold, node._opacity);
    ctx.fill();

    const inputLabel = node.sigmoidInputLabel || `${{rawInput >= 0 ? '+' : ''}}${{rawInput.toFixed(2)}}`;
    const outputLabel = node.sigmoidOutputLabel || `${{Math.round(probability * 100)}}%`;
    ctx.fillStyle = hexToRgba(THEME.channelGold, node._opacity);
    drawLabel(inputLabel, pointX, bottom + labelFont * 1.45);
    drawLabel(outputLabel, left - 8, pointY + labelFont * 0.35, 'right');
  }}

  ctx.restore();
}}

function drawPetPortrait(ctx, node) {{
  // The native portrait uses the same fixed geometry as Cairo. Feature
  // highlights turn a concrete visual cue into an explanation anchor.
  const r = node.rect;
  const captionHeight = node.content ? Math.min(36, r.h * 0.18) : 0;
  const illustrationHeight = Math.max(1, r.h - captionHeight);
  const side = Math.min(r.w, illustrationHeight);
  const cx = r.x + r.w / 2;
  const cy = r.y + illustrationHeight * 0.54;
  const radius = side * 0.29;
  const sp = node.stylePops || {{}};
  const outline = THEME.primary;
  const accent = sp.border || THEME.accent;
  const outlineWidth = Math.max(1.6, THEME.boxBorderWidth * (
    sp.size_variant === 'compact' ? 0.72 : (sp.size_variant === 'hero' ? 1.15 : 1.0)
  ));
  const faceFill = '#F2C99D';
  const floppyFill = '#B97854';
  const catFill = '#C88466';
  const innerEarFill = '#EAA295';
  const muzzleFill = '#FFF3E5';
  const featureFill = '#392D29';
  const kind = node.petKind || 'mystery';
  const highlights = new Set(node.petHighlights || []);

  function paintPath(fill, border = outline, width = outlineWidth) {{
    ctx.fillStyle = hexToRgba(fill, node._opacity);
    ctx.fill();
    ctx.strokeStyle = hexToRgba(border, node._opacity);
    ctx.lineWidth = width;
    ctx.stroke();
  }}

  function floppyEarPath(direction) {{
    const x = (value) => cx + direction * value * radius;
    const y = (value) => cy + value * radius;
    ctx.beginPath();
    ctx.moveTo(x(0.60), y(-0.56));
    ctx.bezierCurveTo(x(1.26), y(-1.00), x(1.43), y(-0.08), x(0.99), y(0.33));
    ctx.bezierCurveTo(x(0.80), y(0.49), x(0.66), y(0.17), x(0.60), y(-0.18));
    ctx.closePath();
  }}

  function catEarPath(direction) {{
    const x = (value) => cx + direction * value * radius;
    const y = (value) => cy + value * radius;
    ctx.beginPath();
    ctx.moveTo(x(0.48), y(-0.57));
    ctx.lineTo(x(0.83), y(-1.34));
    ctx.lineTo(x(0.14), y(-0.91));
    ctx.lineTo(x(0.08), y(-0.37));
    ctx.closePath();
  }}

  function drawFloppyEar(direction) {{
    floppyEarPath(direction);
    paintPath(floppyFill);
  }}

  function drawCatEar(direction) {{
    catEarPath(direction);
    paintPath(catFill);
    const x = (value) => cx + direction * value * radius;
    const y = (value) => cy + value * radius;
    ctx.beginPath();
    ctx.moveTo(x(0.52), y(-0.64));
    ctx.lineTo(x(0.76), y(-1.13));
    ctx.lineTo(x(0.24), y(-0.84));
    ctx.closePath();
    ctx.fillStyle = hexToRgba(innerEarFill, node._opacity);
    ctx.fill();
  }}

  function drawSnout(border = null, width = outlineWidth) {{
    ctx.save();
    ctx.translate(cx, cy + radius * 0.31);
    ctx.scale(1.42, 0.72);
    ctx.beginPath();
    ctx.arc(0, 0, radius * 0.36, 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(muzzleFill, node._opacity);
    if (border === null) {{
      ctx.fill();
    }} else {{
      ctx.fill();
      ctx.strokeStyle = hexToRgba(border, node._opacity);
      ctx.lineWidth = width;
      ctx.stroke();
    }}
    ctx.restore();
  }}

  function outlineCurrentPath(color, width) {{
    ctx.strokeStyle = hexToRgba(color, node._opacity);
    ctx.lineWidth = width;
    ctx.stroke();
  }}

  function drawFeatureLabel(text, color, centerX, centerY, targetX, targetY, attachFrom) {{
    const fontSize = Math.max(11, Math.min(18, side * 0.055));
    ctx.font = `bold ${{fontSize}}px ${{cssFontFamily(THEME.fontFamily)}}`;
    const measure = ctx.measureText(text);
    const boxWidth = measure.width + 18;
    const boxHeight = Math.max(fontSize * 1.52, 24);
    const boxX = centerX - boxWidth / 2;
    const boxY = centerY - boxHeight / 2;
    const lineX = attachFrom === 'left' ? boxX : boxX + boxWidth;

    ctx.strokeStyle = hexToRgba(color, node._opacity);
    ctx.lineWidth = Math.max(1.35, outlineWidth * 0.62);
    ctx.beginPath();
    ctx.moveTo(lineX, centerY);
    ctx.lineTo(targetX, targetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(targetX, targetY, Math.max(2.5, outlineWidth), 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(color, node._opacity);
    ctx.fill();

    roundedRect(ctx, boxX, boxY, boxWidth, boxHeight, boxHeight / 2);
    ctx.fillStyle = hexToRgba(THEME.backgroundColor, node._opacity);
    ctx.fill();
    ctx.strokeStyle = hexToRgba(color, node._opacity);
    ctx.lineWidth = Math.max(1.35, outlineWidth * 0.62);
    ctx.stroke();
    ctx.fillStyle = hexToRgba(color, node._opacity);
    ctx.fillText(text, centerX - measure.width / 2, centeredTextBaseline(ctx, {{x: boxX, y: boxY, w: boxWidth, h: boxHeight}}, text, fontSize));
  }}

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  if (kind === 'dog') {{
    drawFloppyEar(-1);
    drawFloppyEar(1);
  }} else if (kind === 'cat') {{
    drawCatEar(-1);
    drawCatEar(1);
  }} else {{
    drawFloppyEar(-1);
    drawCatEar(1);
  }}

  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  paintPath(faceFill);

  for (const eyeX of [cx - radius * 0.34, cx + radius * 0.34]) {{
    ctx.beginPath();
    ctx.arc(eyeX, cy - radius * 0.08, Math.max(2.2, radius * 0.072), 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(featureFill, node._opacity);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(eyeX - radius * 0.018, cy - radius * 0.105, Math.max(0.8, radius * 0.018), 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba('#FFFFFF', node._opacity);
    ctx.fill();
  }}

  // The elongated oval makes the snout visually clear before any teaching
  // annotation outlines it in cyan.
  drawSnout('#D5A273', Math.max(1.2, outlineWidth * 0.46));

  ctx.beginPath();
  ctx.moveTo(cx, cy + radius * 0.22);
  ctx.lineTo(cx - radius * 0.11, cy + radius * 0.10);
  ctx.lineTo(cx + radius * 0.11, cy + radius * 0.10);
  ctx.closePath();
  ctx.fillStyle = hexToRgba(featureFill, node._opacity);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(featureFill, node._opacity);
  ctx.lineWidth = Math.max(1.4, outlineWidth * 0.72);
  ctx.beginPath();
  ctx.moveTo(cx, cy + radius * 0.22);
  ctx.bezierCurveTo(cx - radius * 0.03, cy + radius * 0.40, cx - radius * 0.21, cy + radius * 0.43, cx - radius * 0.26, cy + radius * 0.32);
  ctx.moveTo(cx, cy + radius * 0.22);
  ctx.bezierCurveTo(cx + radius * 0.03, cy + radius * 0.40, cx + radius * 0.21, cy + radius * 0.43, cx + radius * 0.26, cy + radius * 0.32);
  ctx.stroke();

  const whiskerSides = kind === 'cat' ? [-1, 1] : (kind === 'mystery' ? [1] : []);
  for (const direction of whiskerSides) {{
    for (const verticalOffset of [-0.03, 0.10]) {{
      ctx.beginPath();
      ctx.moveTo(cx + direction * radius * 0.53, cy + radius * (0.22 + verticalOffset));
      ctx.lineTo(cx + direction * radius * 1.10, cy + radius * (0.16 + verticalOffset * 1.8));
      ctx.stroke();
    }}
  }}

  if (highlights.has('ears')) {{
    const coral = THEME.channelCoral || '#A63D36';
    const earWidth = Math.max(2.4, outlineWidth * 1.38);
    if (kind === 'dog') {{
      for (const direction of [-1, 1]) {{
        floppyEarPath(direction);
        outlineCurrentPath(coral, earWidth);
      }}
    }} else if (kind === 'cat') {{
      for (const direction of [-1, 1]) {{
        catEarPath(direction);
        outlineCurrentPath(coral, earWidth);
      }}
    }} else {{
      // For the mystery animal, point at the floppy dog ear: that is the
      // observation the following explanation calls a feature.
      floppyEarPath(-1);
      outlineCurrentPath(coral, earWidth);
    }}
  }}

  if (highlights.has('snout')) {{
    drawSnout(THEME.channelCyan || '#267684', Math.max(2.4, outlineWidth * 1.38));
  }}

  if (kind === 'mystery') {{
    const badgeRadius = Math.max(10, radius * 0.27);
    const badgeX = cx + radius * 0.87;
    const badgeY = cy - radius * 0.95;
    ctx.beginPath();
    ctx.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);
    paintPath('#FFFDF7', accent, Math.max(1.5, outlineWidth * 0.78));
    ctx.font = `bold ${{Math.max(14, radius * 0.52)}}px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(accent, node._opacity);
    const mark = ctx.measureText('?');
    const badgeRect = {{x: badgeX - badgeRadius, y: badgeY - badgeRadius, w: badgeRadius * 2, h: badgeRadius * 2}};
    ctx.fillText('?', badgeX - mark.width / 2, centeredTextBaseline(ctx, badgeRect, '?', Math.max(14, radius * 0.52)));
  }}
  ctx.restore();

  if (node.showFeatureLabels) {{
    const gutter = Math.max(0, (r.w - side) / 2);
    if (highlights.has('ears')) {{
      drawFeatureLabel(
        kind === 'cat' ? 'POINTED EARS' : 'FLOPPY EARS',
        THEME.channelCoral || '#A63D36',
        r.x + gutter / 2,
        cy - radius * 0.52,
        cx - radius * 1.01,
        cy - radius * 0.36,
        'right',
      );
    }}
    if (highlights.has('snout')) {{
      drawFeatureLabel(
        'LONG SNOUT',
        THEME.channelCyan || '#267684',
        r.x + r.w - gutter / 2,
        cy + radius * 0.36,
        cx + radius * 0.50,
        cy + radius * 0.31,
        'left',
      );
    }}
  }}

  if (node.content) {{
    const fontSize = Math.min(sp.font_size || THEME.fontSizeCaption, Math.max(13, side * 0.12));
    const weight = sp.font_weight === 'bold' ? '700' : '400';
    ctx.font = `${{weight}} ${{fontSize}}px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(THEME.textColor, node._opacity);
    const label = ctx.measureText(node.content);
    const captionRect = {{x: r.x, y: r.y + r.h - captionHeight, w: r.w, h: captionHeight}};
    ctx.fillText(node.content, cx - label.width / 2, centeredTextBaseline(ctx, captionRect, node.content, fontSize));
  }}
}}

function drawSemanticIcon(ctx, node) {{
  // Fixed, flat geometry is deliberate: these illustrations need no image
  // assets, and their shape stays identical in the browser and render worker.
  const r = node.rect;
  const captionHeight = node.content ? Math.min(30, r.h * 0.22) : 0;
  const side = Math.min(r.w, Math.max(1, r.h - captionHeight));
  const x = r.x + (r.w - side) / 2;
  const y = r.y;
  const scale = side / 100;
  const sp = node.stylePops || {{}};
  const ink = THEME.primary;
  const paper = THEME.backgroundColor;
  const cyan = sp.border || THEME.channelCyan || '#267684';
  const coral = THEME.channelCoral || '#A63D36';
  const gold = THEME.channelGold || '#846000';
  const strokeScale = sp.size_variant === 'compact' ? 0.72 : (sp.size_variant === 'hero' ? 1.15 : 1.0);
  const lineWidth = Math.max(1.5, THEME.boxBorderWidth * strokeScale);

  function fillStroke(fill, border = ink, width = lineWidth) {{
    ctx.fillStyle = hexToRgba(fill, node._opacity);
    ctx.fill();
    ctx.strokeStyle = hexToRgba(border, node._opacity);
    ctx.lineWidth = width;
    ctx.stroke();
  }}

  function stroke(color = ink, width = lineWidth) {{
    ctx.strokeStyle = hexToRgba(color, node._opacity);
    ctx.lineWidth = width;
    ctx.stroke();
  }}

  function arrow(x0, y0, x1, y1, color) {{
    ctx.strokeStyle = hexToRgba(color, node._opacity);
    ctx.lineWidth = lineWidth * 1.1;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    const angle = Math.atan2(y1 - y0, x1 - x0);
    const head = 8;
    ctx.beginPath();
    for (const direction of [Math.PI * 0.82, -Math.PI * 0.82]) {{
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 + Math.cos(angle + direction) * head, y1 + Math.sin(angle + direction) * head);
    }}
    ctx.stroke();
  }}

  function cashNote(x0, y0, width, height) {{
    roundedRect(ctx, x0, y0, width, height, 5);
    fillStroke('#E4F0ED', cyan);
    ctx.beginPath();
    ctx.arc(x0 + width / 2, y0 + height / 2, height * 0.22, 0, Math.PI * 2);
    fillStroke(paper, cyan, lineWidth * 0.7);
    ctx.font = `bold ${{height * 0.50}}px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(ink, node._opacity);
    const mark = ctx.measureText('$');
    ctx.fillText('$', x0 + width / 2 - mark.width / 2, centeredTextBaseline(ctx, {{x: x0, y: y0, w: width, h: height}}, '$', height * 0.50));
  }}

  const icon = node.iconName || 'fund';
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(scale, scale);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  if (icon === 'fund') {{
    for (const [baseX, height, color] of [[18, 22, coral], [39, 38, cyan], [63, 29, gold]]) {{
      roundedRect(ctx, baseX, 79 - height, 18, height, 4);
      fillStroke(paper, color);
      for (let lineY = Math.floor(79 - height + 7); lineY < 79; lineY += 8) {{
        ctx.beginPath();
        ctx.moveTo(baseX + 3, lineY);
        ctx.lineTo(baseX + 15, lineY);
        stroke(color, lineWidth * 0.55);
      }}
    }}
    ctx.beginPath();
    ctx.arc(50, 24, 15, 0, Math.PI * 2);
    fillStroke('#F6E9C9', gold);
    ctx.font = `bold 17px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(ink, node._opacity);
    const mark = ctx.measureText('$');
    ctx.fillText('$', 50 - mark.width / 2, centeredTextBaseline(ctx, {{x: 35, y: 9, w: 30, h: 30}}, '$', 17));
  }} else if (icon === 'storefront') {{
    roundedRect(ctx, 17, 46, 66, 37, 3);
    fillStroke(paper, ink);
    ctx.beginPath();
    ctx.moveTo(12, 45); ctx.lineTo(88, 45); ctx.lineTo(80, 28); ctx.lineTo(20, 28); ctx.closePath();
    fillStroke('#E4F0ED', cyan);
    for (const stripeX of [23, 38, 53, 68]) {{
      ctx.beginPath();
      ctx.moveTo(stripeX, 29); ctx.lineTo(stripeX + 9, 29); ctx.lineTo(stripeX + 3, 45); ctx.lineTo(stripeX - 6, 45); ctx.closePath();
      ctx.fillStyle = hexToRgba([23, 53].includes(stripeX) ? coral : paper, node._opacity);
      ctx.fill();
    }}
    roundedRect(ctx, 43, 59, 14, 24, 2);
    fillStroke('#F6E9C9', gold);
    ctx.beginPath(); ctx.arc(53, 70, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(ink, node._opacity); ctx.fill();
  }} else if (icon === 'warehouse') {{
    ctx.beginPath();
    ctx.moveTo(12, 43); ctx.lineTo(50, 18); ctx.lineTo(88, 43); ctx.lineTo(88, 82); ctx.lineTo(12, 82); ctx.closePath();
    fillStroke('#E4F0ED', cyan);
    for (const doorX of [24, 45, 66]) {{
      roundedRect(ctx, doorX, 55, 12, 27, 1);
      fillStroke(paper, ink, lineWidth * 0.7);
    }}
    ctx.beginPath(); ctx.moveTo(12, 43); ctx.lineTo(88, 43); stroke(ink);
  }} else if (icon === 'cash') {{
    cashNote(14, 28, 72, 44);
  }} else if (icon === 'shares') {{
    for (const [offsetX, offsetY, color] of [[12, 34, coral], [20, 25, gold], [28, 16, cyan]]) {{
      roundedRect(ctx, offsetX, offsetY, 58, 48, 4);
      fillStroke(paper, color);
      ctx.beginPath(); ctx.moveTo(offsetX + 13, offsetY + 17); ctx.lineTo(offsetX + 45, offsetY + 17); ctx.moveTo(offsetX + 13, offsetY + 28); ctx.lineTo(offsetX + 35, offsetY + 28); stroke(ink, lineWidth * 0.62);
      ctx.beginPath(); ctx.arc(offsetX + 45, offsetY + 32, 5, 0, Math.PI * 2); ctx.fillStyle = hexToRgba(color, node._opacity); ctx.fill();
    }}
  }} else if (icon === 'borrow') {{
    cashNote(18, 18, 48, 30);
    arrow(45, 53, 45, 76, coral);
    ctx.beginPath();
    ctx.moveTo(20, 78); ctx.lineTo(30, 67); ctx.lineTo(60, 67); ctx.lineTo(74, 78); ctx.lineTo(70, 84); ctx.lineTo(25, 84); ctx.closePath();
    fillStroke('#F6E9C9', gold);
  }} else if (icon === 'loan') {{
    ctx.beginPath(); ctx.arc(29, 48, 18, 0, Math.PI * 2); fillStroke('#F6E9C9', gold);
    ctx.font = `bold 20px ${{cssFontFamily(THEME.fontFamily)}}`; ctx.fillStyle = hexToRgba(ink, node._opacity);
    const mark = ctx.measureText('$'); ctx.fillText('$', 29 - mark.width / 2, centeredTextBaseline(ctx, {{x: 11, y: 30, w: 36, h: 36}}, '$', 20));
    arrow(49, 48, 77, 48, cyan);
    roundedRect(ctx, 63, 59, 24, 23, 3); fillStroke(paper, ink);
    ctx.beginPath(); ctx.moveTo(69, 67); ctx.lineTo(81, 67); ctx.moveTo(69, 74); ctx.lineTo(78, 74); stroke(ink, lineWidth * 0.65);
  }} else if (icon === 'handshake') {{
    ctx.beginPath(); ctx.moveTo(13, 44); ctx.lineTo(32, 30); ctx.lineTo(53, 47); ctx.lineTo(44, 59); ctx.closePath(); fillStroke('#E4F0ED', cyan);
    ctx.beginPath(); ctx.moveTo(87, 44); ctx.lineTo(68, 30); ctx.lineTo(46, 49); ctx.lineTo(55, 61); ctx.closePath(); fillStroke('#F6E9C9', gold);
    for (let index = 0; index < 3; index += 1) {{
      const x0 = 46 + index * 5; const y0 = 53 + index * 5;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + 9, y0 + 7); stroke(coral, lineWidth * 0.75);
    }}
  }}
  ctx.restore();

  if (node.content) {{
    const fontSize = Math.min(sp.font_size || THEME.fontSizeCaption, Math.max(12, side * 0.15));
    const weight = sp.font_weight === 'bold' ? '700' : '400';
    ctx.font = `${{weight}} ${{fontSize}}px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(THEME.textColor, node._opacity);
    const label = ctx.measureText(node.content);
    const captionRect = {{x: r.x, y: r.y + r.h - captionHeight, w: r.w, h: captionHeight}};
    ctx.fillText(node.content, r.x + r.w / 2 - label.width / 2, centeredTextBaseline(ctx, captionRect, node.content, fontSize));
  }}
}}

function drawConnector(ctx, node, nodeMap) {{
  if (!node.fromId || !node.toId) return;
  const from = nodeMap[node.fromId], to = nodeMap[node.toId];
  if (!from || !to) return;
  const horizontalOverlap = Math.min(from.rect.x + from.rect.w, to.rect.x + to.rect.w) - Math.max(from.rect.x, to.rect.x);
  const verticalOverlap = Math.min(from.rect.y + from.rect.h, to.rect.y + to.rect.h) - Math.max(from.rect.y, to.rect.y);
  const fromCx = from.rect.x + from.rect.w / 2;
  const fromCy = from.rect.y + from.rect.h / 2;
  const toCx = to.rect.x + to.rect.w / 2;
  const toCy = to.rect.y + to.rect.h / 2;
  const dx = toCx - fromCx;
  const dy = toCy - fromCy;

  let sx, sy, ex, ey;
  if (horizontalOverlap > 0 && Math.abs(dy) > 1e-6) {{
    sx = fromCx;
    ex = toCx;
    if (dy > 0) {{
      sy = from.rect.y + from.rect.h;
      ey = to.rect.y;
    }} else {{
      sy = from.rect.y;
      ey = to.rect.y + to.rect.h;
    }}
  }} else if (verticalOverlap > 0 && Math.abs(dx) > 1e-6) {{
    sy = fromCy;
    ey = toCy;
    if (dx > 0) {{
      sx = from.rect.x + from.rect.w;
      ex = to.rect.x;
    }} else {{
      sx = from.rect.x;
      ex = to.rect.x + to.rect.w;
    }}
  }} else if (Math.abs(dy) > Math.abs(dx)) {{
    sx = fromCx;
    ex = toCx;
    if (dy > 0) {{
      sy = from.rect.y + from.rect.h;
      ey = to.rect.y;
    }} else {{
      sy = from.rect.y;
      ey = to.rect.y + to.rect.h;
    }}
  }} else {{
    sy = fromCy;
    ey = toCy;
    if (dx > 0) {{
      sx = from.rect.x + from.rect.w;
      ex = to.rect.x;
    }} else {{
      sx = from.rect.x;
      ex = to.rect.x + to.rect.w;
    }}
  }}
  const sp = node.stylePops || {{}};
  const connectorColor = sp.border || sp.color || THEME.connectorColor;
  ctx.strokeStyle = hexToRgba(connectorColor, node._opacity);
  ctx.lineWidth = THEME.connectorWidth;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  const endX = sx + (ex - sx) * node._drawProgress;
  const endY = sy + (ey - sy) * node._drawProgress;
  ctx.lineTo(endX, endY);
  ctx.stroke();
  if (node._drawProgress >= 0.9) {{
    const arrowSize = THEME.arrowSize;
    const angle = Math.atan2(endY - sy, endX - sx);
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowSize * Math.cos(angle - 0.4),
      endY - arrowSize * Math.sin(angle - 0.4),
    );
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowSize * Math.cos(angle + 0.4),
      endY - arrowSize * Math.sin(angle + 0.4),
    );
    ctx.stroke();
  }}
  if (node._flowProgress !== null && node._flowProgress !== undefined) {{
    const signalX = sx + (ex - sx) * node._flowProgress;
    const signalY = sy + (ey - sy) * node._flowProgress;
    ctx.beginPath();
    ctx.arc(signalX, signalY, Math.max(5, THEME.connectorWidth * 2.2), 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(connectorColor, node._opacity);
    ctx.fill();
  }}
}}

function drawGroup(ctx, node, nodeMap) {{
  if (node.label) {{
    ctx.font = `bold ${{THEME.fontSizeCaption}}px ${{cssFontFamily(THEME.fontFamily)}}`;
    ctx.fillStyle = hexToRgba(THEME.textLight, node._opacity);
    const m = ctx.measureText(node.label);
    ctx.fillText(node.label, node.rect.x + (node.rect.w - m.width) / 2, node.rect.y - 8);
  }}
  for (const child of (node.children || [])) {{
    const prevVisible = child._visible;
    const prevOpacity = child._opacity;
    const prevDrawProgress = child._drawProgress;
    child._visible = node._visible && prevVisible;
    child._opacity = node._opacity * prevOpacity;
    child._drawProgress = node._drawProgress * prevDrawProgress;
    drawNode(ctx, child, nodeMap);
    child._visible = prevVisible;
    child._opacity = prevOpacity;
    child._drawProgress = prevDrawProgress;
  }}
}}

function buildNodeMap(nodes) {{
  const nodeMap = {{}};
  function mapNodes(items) {{
    for (const n of items) {{
      nodeMap[n.id] = n;
      if (n.children) mapNodes(n.children);
    }}
  }}
  mapNodes(nodes);
  return nodeMap;
}}

function renderScene(ctx, scene, localTime, alpha = 1) {{
  sceneTime = localTime;
  const nodeMap = buildNodeMap(scene.nodes);
  applyAnimations(nodeMap, scene.timeline, localTime);

  ctx.save();
  ctx.globalAlpha = alpha;

  for (const node of scene.nodes) drawNode(ctx, node, nodeMap);
  ctx.restore();
}}

function render(t) {{
  const {{ scene, localTime, sceneAlpha }} = getSceneAtTime(t);
  sceneTime = localTime;
  ctx.clearRect(0, 0, GRAPH.width, GRAPH.height);

  // Background
  ctx.fillStyle = THEME.backgroundColor;
  ctx.fillRect(0, 0, GRAPH.width, GRAPH.height);

  if (!scene) return;

  renderScene(ctx, scene, localTime, sceneAlpha);

  // Narration
  if (GRAPH.showNarration) {{
    narrationDiv.textContent = scene.narration || '';
  }} else {{
    narrationDiv.textContent = '';
  }}
}}

function renderPreviewAt(time) {{
  // A capture at totalDuration would otherwise select no scene. Clamp to the
  // final renderable frame and pause playback so a screenshot is repeatable.
  const requestedTime = Number(time);
  const frameDuration = 1 / Math.max(1, GRAPH.fps || 1);
  const maxRenderableTime = Math.max(0, GRAPH.totalDuration - frameDuration);
  currentTime = Math.min(
    maxRenderableTime,
    Math.max(0, Number.isFinite(requestedTime) ? requestedTime : 0),
  );
  playing = false;
  playBtn.textContent = 'Play';
  timelineSlider.value = GRAPH.totalDuration > 0
    ? (currentTime / GRAPH.totalDuration) * 1000
    : 0;
  render(currentTime);
  return currentTime;
}}

// Stable public hooks for headless evidence captures. Wait for
// window.__kaivraPreviewReady, call __kaivraCaptureAt(sceneTime), then
// screenshot #canvas (not the responsive player chrome).
window.__kaivraRenderAt = renderPreviewAt;
window.__kaivraCaptureAt = (time) => {{
  setCanvasCaptureMode(true);
  return renderPreviewAt(time);
}};

function captureTimeFromQuery() {{
  const rawTime = new URLSearchParams(window.location.search).get('capture_time');
  if (rawTime === null || rawTime.trim() === '') return null;
  const value = Number(rawTime);
  return Number.isFinite(value) ? value : null;
}}

function tick(timestamp) {{
  if (playing) {{
    const dt = (timestamp - lastTimestamp) / 1000;
    lastTimestamp = timestamp;
    currentTime += dt;
    if (currentTime >= GRAPH.totalDuration) {{
      currentTime = 0;
    }}
    timelineSlider.value = (currentTime / GRAPH.totalDuration) * 1000;
  }}
  timeDisplay.textContent = formatTime(currentTime) + ' / ' + formatTime(GRAPH.totalDuration);
  render(currentTime);
  requestAnimationFrame(tick);
}}

function startPlayer() {{
  const captureTime = captureTimeFromQuery();
  if (captureTime !== null) {{
    setCanvasCaptureMode(true);
    renderPreviewAt(captureTime);
  }} else {{
    renderPreviewAt(currentTime);
  }}
  window.__kaivraPreviewReady = true;
  requestAnimationFrame(tick);
}}

if ((THEME.fontFamily === 'Inter' || THEME.fontFamily === 'Inter Variable') && document.fonts && document.fonts.load) {{
  document.fonts.load('400 16px "Inter"').catch(() => {{}}).then(startPlayer);
}} else {{
  startPlayer();
}}
</script>
</body>
</html>"""


def _serve_with_reload(html: str, port: int) -> None:
    """Serve the HTML with a simple HTTP server."""
    import http.server
    import os
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="kaivra-")
    path = os.path.join(tmpdir, "index.html")
    with open(path, "w") as f:
        f.write(html)

    os.chdir(tmpdir)
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("", port), handler)

    print(f"Serving preview at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()
