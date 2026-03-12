"""Cairo-based frame renderer.

Renders a single frame of the scene graph to a Cairo surface.
Handles all object types, animation state, camera transforms, and theme styling.
"""

from __future__ import annotations

import math
import random

import cairo

from dsa_anim.dsl.schema import ObjectType
from dsa_anim.scene_graph.models import SceneGraph, ResolvedScene, SceneNode, CameraState
from dsa_anim.scene_graph.timeline import apply_animations_at_time, compute_camera_at_time
from dsa_anim.themes.base import ThemeSpec
from dsa_anim.themes.registry import get_theme
from dsa_anim.utils.color import hex_to_rgba
from dsa_anim.utils.geometry import Rect


class CairoRenderer:
    """Renders scene graph frames using Cairo."""

    def __init__(self, theme: ThemeSpec):
        self.theme = theme

    def render_frame(self, graph: SceneGraph, time: float) -> cairo.ImageSurface:
        """Render a single frame at the given time, with crossfade transitions."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, graph.width, graph.height)
        ctx = cairo.Context(surface)

        scene_idx, scene, scene_time = self._locate_scene(graph, time)
        if scene is None:
            self._fill_background(ctx, graph.width, graph.height)
            return surface

        # Check if we're in a crossfade window with the next scene
        blend = 0.0
        next_scene_time = 0.0
        next_scene = None
        if scene.transition and scene.transition.duration > 0:
            trans_dur = scene.transition.duration
            time_remaining = scene.duration - scene_time
            if time_remaining < trans_dur and scene_idx + 1 < len(graph.scenes):
                blend = max(0.0, min(1.0, 1.0 - time_remaining / trans_dur))
                next_scene_time = trans_dur - time_remaining
                next_scene = graph.scenes[scene_idx + 1]

        if blend > 0 and next_scene is not None:
            # Render both scenes and composite
            curr_surf = self._render_scene_to_surface(graph, scene, scene_time)
            next_surf = self._render_scene_to_surface(graph, next_scene, next_scene_time)
            ctx.set_source_surface(curr_surf, 0, 0)
            ctx.paint()
            ctx.set_source_surface(next_surf, 0, 0)
            ctx.paint_with_alpha(blend)
        else:
            self._draw_scene(ctx, graph, scene, scene_time)

        return surface

    def _render_scene_to_surface(self, graph: SceneGraph, scene: "ResolvedScene", scene_time: float) -> cairo.ImageSurface:
        """Render a single scene to an offscreen surface."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, graph.width, graph.height)
        ctx = cairo.Context(surface)
        self._draw_scene(ctx, graph, scene, scene_time)
        return surface

    def _draw_scene(self, ctx: cairo.Context, graph: SceneGraph, scene: "ResolvedScene", scene_time: float) -> None:
        """Draw a complete scene into an existing context."""
        apply_animations_at_time(scene.node_map, scene.timeline, scene_time)
        camera = compute_camera_at_time(
            scene.camera_initial, scene.camera_keyframes,
            scene_time, graph.width, graph.height,
        )
        self._fill_background(ctx, graph.width, graph.height)
        ctx.save()
        self._apply_camera(ctx, camera, graph.width, graph.height)
        for node in scene.nodes:
            self._draw_node(ctx, node, scene.node_map)
        ctx.restore()
        ctx.save()
        if scene.narration:
            self._draw_narration(ctx, scene.narration, scene_time, scene.duration, graph.width, graph.height)
        self._draw_progress_bar(ctx, scene_time, scene.duration, graph.width)
        ctx.restore()

    def render_frame_to_file(self, graph: SceneGraph, time: float, path: str) -> None:
        """Render a frame and save as PNG."""
        surface = self.render_frame(graph, time)
        surface.write_to_png(path)

    def render_frame_to_bytes(self, graph: SceneGraph, time: float) -> bytes:
        """Render a frame and return raw ARGB pixel data."""
        surface = self.render_frame(graph, time)
        return bytes(surface.get_data())

    # --- Internal ---

    def _locate_scene(self, graph: SceneGraph, time: float) -> tuple[int, "ResolvedScene | None", float]:
        """Find (index, scene, local_scene_time) for the given global time."""
        elapsed = 0.0
        for i, scene in enumerate(graph.scenes):
            if time < elapsed + scene.duration:
                return i, scene, time - elapsed
            elapsed += scene.duration
        return -1, None, 0.0

    def _fill_background(self, ctx: cairo.Context, w: int, h: int) -> None:
        r, g, b, a = hex_to_rgba(self.theme.background_color)
        ctx.set_source_rgba(r, g, b, a)
        ctx.rectangle(0, 0, w, h)
        ctx.fill()

    def _apply_camera(self, ctx: cairo.Context, camera: CameraState, w: int, h: int) -> None:
        if camera.zoom != 1.0:
            cx, cy = w / 2, h / 2
            ctx.translate(cx, cy)
            ctx.scale(camera.zoom, camera.zoom)
            ctx.translate(-cx, -cy)

    def _draw_node(self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode]) -> None:
        if not node.visible:
            return

        ctx.save()

        # Apply translate (move animation)
        if node.translate_x != 0.0 or node.translate_y != 0.0:
            ctx.translate(node.translate_x, node.translate_y)

        # Apply scale transform around center
        if node.scale_x != 1.0 or node.scale_y != 1.0:
            cx, cy = node.rect.center.x, node.rect.center.y
            ctx.translate(cx, cy)
            ctx.scale(node.scale_x, node.scale_y)
            ctx.translate(-cx, -cy)

        match node.obj_type:
            case ObjectType.TEXT:
                self._draw_text(ctx, node)
            case ObjectType.BOX:
                self._draw_box(ctx, node)
            case ObjectType.TOKEN:
                self._draw_token(ctx, node)
            case ObjectType.CONNECTOR:
                self._draw_connector(ctx, node, node_map)
            case ObjectType.GROUP:
                self._draw_group(ctx, node, node_map)
            case ObjectType.MATRIX:
                self._draw_matrix(ctx, node)
            case ObjectType.ATTENTION_MAP:
                self._draw_attention_map(ctx, node)
            case ObjectType.PROBABILITY_BAR:
                self._draw_probability_bar(ctx, node)
            case ObjectType.CIRCLE:
                self._draw_circle(ctx, node)
            case ObjectType.CALLOUT:
                self._draw_callout(ctx, node, node_map)
            case _:
                self._draw_box(ctx, node)

        # Draw highlight overlay if active
        if node.highlight_intensity > 0:
            self._draw_highlight(ctx, node)

        ctx.restore()

    def _draw_text(self, ctx: cairo.Context, node: SceneNode) -> None:
        if not node.content:
            return

        style = node.style_props
        font_size = style.get("font_size", self.theme.font_size_body)
        color = style.get("color", self.theme.text_color)
        weight = cairo.FONT_WEIGHT_BOLD if style.get("font_weight") == "bold" else cairo.FONT_WEIGHT_NORMAL

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, weight)
        ctx.set_font_size(font_size)

        r, g, b, _ = hex_to_rgba(color)
        ctx.set_source_rgba(r, g, b, node.opacity)

        # Handle typewriter effect
        text = node.content
        if node.draw_progress < 1.0:
            chars_to_show = int(len(text) * node.draw_progress)
            text = text[:chars_to_show]

        # Measure and center
        extents = ctx.text_extents(text)
        x = node.rect.x + (node.rect.width - extents.width) / 2
        y = node.rect.y + (node.rect.height + extents.height) / 2

        ctx.move_to(x, y)
        ctx.show_text(text)

    def _draw_box(self, ctx: cairo.Context, node: SceneNode) -> None:
        r = node.rect
        cr = self.theme.box_corner_radius

        # Shadow
        if self.theme.shadow:
            sr, sg, sb, sa = hex_to_rgba(self.theme.shadow_color)
            self._rounded_rect(ctx, r.x + self.theme.shadow_offset, r.y + self.theme.shadow_offset, r.width, r.height, cr)
            ctx.set_source_rgba(sr, sg, sb, sa * node.opacity)
            ctx.fill()

        # Fill
        fr, fg, fb, _ = hex_to_rgba(self.theme.box_fill)
        self._rounded_rect(ctx, r.x, r.y, r.width, r.height, cr)
        ctx.set_source_rgba(fr, fg, fb, node.opacity)
        ctx.fill_preserve()

        # Border
        br, bg, bb, _ = hex_to_rgba(self.theme.box_border)
        ctx.set_source_rgba(br, bg, bb, node.opacity)
        ctx.set_line_width(self.theme.box_border_width)
        if self.theme.sketch_effect:
            self._sketch_stroke(ctx)
        else:
            ctx.stroke()

        # Content text
        if node.content:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(self.theme.font_size_body)
            tr, tg, tb, _ = hex_to_rgba(self.theme.text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)

            text = node.content
            if node.draw_progress < 1.0:
                text = text[:int(len(text) * node.draw_progress)]

            extents = ctx.text_extents(text)
            x = r.x + (r.width - extents.width) / 2
            y = r.y + (r.height + extents.height) / 2
            ctx.move_to(x, y)
            ctx.show_text(text)

    def _draw_token(self, ctx: cairo.Context, node: SceneNode) -> None:
        r = node.rect
        cr = self.theme.token_corner_radius

        # Fill — fade in with draw_progress
        fr, fg, fb, _ = hex_to_rgba(self.theme.token_fill)
        self._rounded_rect(ctx, r.x, r.y, r.width, r.height, cr)
        fill_opacity = node.opacity * min(1.0, node.draw_progress * 2)  # fill appears first half
        ctx.set_source_rgba(fr, fg, fb, fill_opacity)
        ctx.fill()

        # Border — draws progressively (perimeter stroke animation)
        br, bg, bb, _ = hex_to_rgba(self.theme.token_border)
        ctx.set_source_rgba(br, bg, bb, node.opacity)
        ctx.set_line_width(2.0)
        if node.draw_progress < 1.0:
            # Compute perimeter and use dash to reveal progressively
            perimeter = 2 * (r.width + r.height)
            visible = perimeter * node.draw_progress
            self._rounded_rect(ctx, r.x, r.y, r.width, r.height, cr)
            ctx.set_dash([visible, perimeter])
            ctx.stroke()
            ctx.set_dash([])
        else:
            self._rounded_rect(ctx, r.x, r.y, r.width, r.height, cr)
            ctx.stroke()

        # Token text
        if node.content:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(self.theme.font_size_body)
            tr, tg, tb, _ = hex_to_rgba(self.theme.text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)

            text = node.content.strip()
            extents = ctx.text_extents(text)
            x = r.x + (r.width - extents.width) / 2
            y = r.y + (r.height + extents.height) / 2
            ctx.move_to(x, y)
            ctx.show_text(text)

        # Token ID badge (small text below)
        if node.token_id is not None:
            ctx.set_font_size(12)
            tid_text = str(node.token_id)
            mr, mg, mb, _ = hex_to_rgba(self.theme.text_light)
            ctx.set_source_rgba(mr, mg, mb, node.opacity * 0.8)
            extents = ctx.text_extents(tid_text)
            x = r.x + (r.width - extents.width) / 2
            y = r.bottom + 14
            ctx.move_to(x, y)
            ctx.show_text(tid_text)

    def _draw_connector(self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode]) -> None:
        if not node.from_id or not node.to_id:
            return

        from_node = node_map.get(node.from_id)
        to_node = node_map.get(node.to_id)
        if not from_node or not to_node:
            return

        # Auto-detect direction based on relative positions
        dx = abs(to_node.rect.center.x - from_node.rect.center.x)
        dy = abs(to_node.rect.center.y - from_node.rect.center.y)
        if dy > dx:
            # Vertical: connect bottom→top
            if to_node.rect.center.y > from_node.rect.center.y:
                start = from_node.rect.bottom_center
                end = to_node.rect.top_center
            else:
                start = from_node.rect.top_center
                end = to_node.rect.bottom_center
        else:
            # Horizontal: connect right→left
            if to_node.rect.center.x > from_node.rect.center.x:
                start = from_node.rect.right_center
                end = to_node.rect.left_center
            else:
                start = from_node.rect.left_center
                end = to_node.rect.right_center

        cr, cg, cb, _ = hex_to_rgba(self.theme.connector_color)
        ctx.set_source_rgba(cr, cg, cb, node.opacity)
        ctx.set_line_width(self.theme.connector_width)

        # Draw line (with draw_progress for stroke animation)
        if node.draw_progress < 1.0:
            end_x = start.x + (end.x - start.x) * node.draw_progress
            end_y = start.y + (end.y - start.y) * node.draw_progress
        else:
            end_x, end_y = end.x, end.y

        ctx.move_to(start.x, start.y)
        ctx.line_to(end_x, end_y)
        ctx.stroke()

        # Arrow head
        if node.draw_progress >= 0.9:
            arrow_size = self.theme.arrow_size
            angle = math.atan2(end_y - start.y, end_x - start.x)
            ctx.move_to(end_x, end_y)
            ctx.line_to(
                end_x - arrow_size * math.cos(angle - 0.4),
                end_y - arrow_size * math.sin(angle - 0.4),
            )
            ctx.move_to(end_x, end_y)
            ctx.line_to(
                end_x - arrow_size * math.cos(angle + 0.4),
                end_y - arrow_size * math.sin(angle + 0.4),
            )
            ctx.stroke()

    def _draw_group(self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode]) -> None:
        # Draw label if present
        if node.label:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(self.theme.font_size_caption)
            lr, lg, lb, _ = hex_to_rgba(self.theme.text_light)
            ctx.set_source_rgba(lr, lg, lb, node.opacity)
            extents = ctx.text_extents(node.label)
            x = node.rect.x + (node.rect.width - extents.width) / 2
            ctx.move_to(x, node.rect.y - 8)
            ctx.show_text(node.label)

        # Draw children
        for child in node.children:
            # Inherit parent visibility/opacity
            child.visible = node.visible
            child.opacity = node.opacity
            child.draw_progress = node.draw_progress
            self._draw_node(ctx, child, node_map)

    def _draw_matrix(self, ctx: cairo.Context, node: SceneNode) -> None:
        rows = node.matrix_rows or 4
        cols = node.matrix_cols or 4
        r = node.rect
        cell_size = min(r.width / (cols + 2), r.height / (rows + 1))
        cell_size = min(cell_size, 40)

        label_offset = 80 if node.matrix_labels else 0
        grid_x = r.x + label_offset + (r.width - label_offset - cols * cell_size) / 2
        grid_y = r.y + (r.height - rows * cell_size) / 2

        # Seed random for consistent heatmap
        rng = random.Random(42)

        cells_to_show = int(rows * cols * node.draw_progress)
        cell_idx = 0

        for row in range(rows):
            # Row label
            if node.matrix_labels and node.matrix_labels.get("rows") and row < len(node.matrix_labels["rows"]):
                ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(14)
                tr, tg, tb, _ = hex_to_rgba(self.theme.text_color)
                ctx.set_source_rgba(tr, tg, tb, node.opacity)
                label = node.matrix_labels["rows"][row]
                extents = ctx.text_extents(label)
                ctx.move_to(grid_x - extents.width - 8, grid_y + row * cell_size + cell_size / 2 + extents.height / 2)
                ctx.show_text(label)

            for col in range(cols):
                if cell_idx >= cells_to_show:
                    break
                x = grid_x + col * cell_size
                y = grid_y + row * cell_size

                # Heatmap color
                val = rng.random()
                hr = 0.04 + val * 0.3
                hg = 0.52 + val * 0.4
                hb = 0.89

                ctx.rectangle(x + 1, y + 1, cell_size - 2, cell_size - 2)
                ctx.set_source_rgba(hr, hg, hb, node.opacity * 0.8)
                ctx.fill_preserve()
                ctx.set_source_rgba(0.8, 0.85, 0.9, node.opacity * 0.5)
                ctx.set_line_width(0.5)
                ctx.stroke()

                cell_idx += 1

    def _draw_attention_map(self, ctx: cairo.Context, node: SceneNode) -> None:
        if not node.tokens:
            return

        r = node.rect
        n = len(node.tokens)

        # Arrange tokens in a horizontal line
        token_width = min(80, r.width / (n + 1))
        total_w = n * token_width + (n - 1) * 8
        start_x = r.x + (r.width - total_w) / 2
        token_y = r.y + r.height * 0.7

        # Draw tokens
        token_positions = {}
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(16)

        for i, token in enumerate(node.tokens):
            tx = start_x + i * (token_width + 8)
            token_positions[token] = (tx + token_width / 2, token_y)

            # Token chip
            fr, fg, fb, _ = hex_to_rgba(self.theme.token_fill)
            self._rounded_rect(ctx, tx, token_y - 15, token_width, 30, 4)
            ctx.set_source_rgba(fr, fg, fb, node.opacity)
            ctx.fill_preserve()
            br, bg, bb, _ = hex_to_rgba(self.theme.token_border)
            ctx.set_source_rgba(br, bg, bb, node.opacity)
            ctx.set_line_width(1)
            ctx.stroke()

            # Token text
            tr, tg, tb, _ = hex_to_rgba(self.theme.text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)
            extents = ctx.text_extents(token)
            ctx.move_to(tx + (token_width - extents.width) / 2, token_y + 5)
            ctx.show_text(token)

        # Draw attention arcs
        if node.highlight_pairs and node.draw_progress > 0.3:
            arc_progress = min(1.0, (node.draw_progress - 0.3) / 0.7)
            pairs_to_show = int(len(node.highlight_pairs) * arc_progress)

            for pair in node.highlight_pairs[:pairs_to_show]:
                from_token = pair.get("from")
                to_token = pair.get("to")
                weight = pair.get("weight", 0.5)

                if from_token in token_positions and to_token in token_positions:
                    fx, fy = token_positions[from_token]
                    tx_pos, ty = token_positions[to_token]

                    # Draw arc above tokens
                    mid_x = (fx + tx_pos) / 2
                    arc_height = abs(tx_pos - fx) * 0.4 + 30

                    ar, ag, ab, _ = hex_to_rgba(self.theme.accent)
                    ctx.set_source_rgba(ar, ag, ab, node.opacity * weight)
                    ctx.set_line_width(2 + weight * 3)

                    ctx.move_to(fx, fy - 15)
                    ctx.curve_to(fx, fy - arc_height, tx_pos, fy - arc_height, tx_pos, ty - 15)
                    ctx.stroke()

    def _draw_probability_bar(self, ctx: cairo.Context, node: SceneNode) -> None:
        if not node.prob_items:
            return

        r = node.rect
        n = len(node.prob_items)
        bar_height = 28
        bar_gap = 8
        label_width = 80
        max_bar_width = r.width - label_width - 80

        total_h = n * (bar_height + bar_gap) - bar_gap
        y_start = r.y + (r.height - total_h) / 2

        max_val = max(item["value"] for item in node.prob_items)

        for i, item in enumerate(node.prob_items):
            y = y_start + i * (bar_height + bar_gap)
            val = item["value"]
            bar_w = (val / max_val) * max_bar_width * node.draw_progress

            # Label
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(18)
            tr, tg, tb, _ = hex_to_rgba(self.theme.text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)
            ctx.move_to(r.x, y + bar_height * 0.7)
            ctx.show_text(item["label"])

            # Bar
            bar_x = r.x + label_width
            if i == 0:
                br, bg, bb, _ = hex_to_rgba(self.theme.accent)
            else:
                br, bg, bb, _ = hex_to_rgba(self.theme.muted)

            self._rounded_rect(ctx, bar_x, y, bar_w, bar_height, 4)
            ctx.set_source_rgba(br, bg, bb, node.opacity * 0.8)
            ctx.fill()

            # Value label
            if node.draw_progress > 0.5:
                ctx.set_font_size(14)
                pct = f"{val * 100:.0f}%"
                ctx.set_source_rgba(tr, tg, tb, node.opacity)
                ctx.move_to(bar_x + bar_w + 8, y + bar_height * 0.7)
                ctx.show_text(pct)

    def _draw_circle(self, ctx: cairo.Context, node: SceneNode) -> None:
        cx, cy = node.rect.center.x, node.rect.center.y
        radius = min(node.rect.width, node.rect.height) / 2

        fr, fg, fb, _ = hex_to_rgba(self.theme.box_fill)
        ctx.arc(cx, cy, radius, 0, 2 * math.pi)
        ctx.set_source_rgba(fr, fg, fb, node.opacity)
        ctx.fill_preserve()

        br, bg, bb, _ = hex_to_rgba(self.theme.box_border)
        ctx.set_source_rgba(br, bg, bb, node.opacity)
        ctx.set_line_width(self.theme.box_border_width)
        ctx.stroke()

        if node.content:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(self.theme.font_size_body)
            tr, tg, tb, _ = hex_to_rgba(self.theme.text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)
            extents = ctx.text_extents(node.content)
            ctx.move_to(cx - extents.width / 2, cy + extents.height / 2)
            ctx.show_text(node.content)

    def _draw_highlight(self, ctx: cairo.Context, node: SceneNode) -> None:
        """Draw a glow/highlight overlay on the node."""
        color = node.highlight_color or "accent"
        hex_color = self.theme.resolve_color(color)
        hr, hg, hb, _ = hex_to_rgba(hex_color)

        r = node.rect
        intensity = node.highlight_intensity * 0.3
        ctx.set_source_rgba(hr, hg, hb, intensity * node.opacity)
        self._rounded_rect(ctx, r.x - 4, r.y - 4, r.width + 8, r.height + 8, self.theme.box_corner_radius + 4)
        ctx.fill()

    def _draw_narration(self, ctx: cairo.Context, text: str, scene_time: float, scene_duration: float, w: int, h: int) -> None:
        """Draw cinematic narration subtitle at the bottom of the screen."""
        # Fade in/out narration
        fade_dur = 0.8
        if scene_time < fade_dur:
            opacity = scene_time / fade_dur
        elif scene_time > scene_duration - fade_dur:
            opacity = (scene_duration - scene_time) / fade_dur
        else:
            opacity = 1.0
        opacity = max(0.0, min(1.0, opacity))

        if opacity <= 0:
            return

        # Semi-transparent backdrop
        bar_h = 80
        bar_y = h - bar_h - 30
        ctx.rectangle(0, bar_y, w, bar_h)
        ctx.set_source_rgba(0, 0, 0, 0.55 * opacity)
        ctx.fill()

        # Narration text — wrap if needed
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(22)
        ctx.set_source_rgba(1, 1, 1, 0.95 * opacity)

        # Simple word wrapping
        max_width = w - 160
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test = (current_line + " " + word).strip()
            ext = ctx.text_extents(test)
            if ext.width > max_width and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        line_height = 28
        total_text_h = len(lines) * line_height
        text_y = bar_y + (bar_h - total_text_h) / 2 + 20

        for i, line in enumerate(lines):
            ext = ctx.text_extents(line)
            x = (w - ext.width) / 2
            ctx.move_to(x, text_y + i * line_height)
            ctx.show_text(line)

    def _draw_progress_bar(self, ctx: cairo.Context, scene_time: float, scene_duration: float, w: int) -> None:
        """Draw a thin accent-colored progress bar at the top of the screen."""
        if scene_duration <= 0:
            return
        progress = scene_time / scene_duration
        bar_h = 3
        ar, ag, ab, _ = hex_to_rgba(self.theme.accent)
        ctx.set_source_rgba(ar, ag, ab, 0.6)
        ctx.rectangle(0, 0, w * progress, bar_h)
        ctx.fill()

    def _draw_callout(self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode]) -> None:
        """Draw a callout annotation — a labeled pointer to another element."""
        if not node.content:
            return

        r = node.rect

        # Callout bubble
        padding = 12
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(16)

        # Word wrap the content
        max_w = 280
        words = node.content.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if ctx.text_extents(test).width > max_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        line_h = 22
        bubble_w = min(max_w + padding * 2, r.width)
        bubble_h = len(lines) * line_h + padding * 2

        bx = r.x + (r.width - bubble_w) / 2
        by = r.y + (r.height - bubble_h) / 2

        # Background
        self._rounded_rect(ctx, bx, by, bubble_w, bubble_h, 8)
        ctx.set_source_rgba(0.05, 0.05, 0.15, 0.85 * node.opacity)
        ctx.fill_preserve()
        ar, ag, ab, _ = hex_to_rgba(self.theme.accent)
        ctx.set_source_rgba(ar, ag, ab, 0.8 * node.opacity)
        ctx.set_line_width(1.5)
        ctx.stroke()

        # Text
        ctx.set_source_rgba(1, 1, 1, 0.95 * node.opacity)
        for i, line in enumerate(lines):
            ext = ctx.text_extents(line)
            ctx.move_to(bx + (bubble_w - ext.width) / 2, by + padding + 14 + i * line_h)
            ctx.show_text(line)

        # Pointer line to target if from_id is set
        if node.from_id:
            target = node_map.get(node.from_id)
            if target:
                # Draw a line from bubble to target
                start_x = bx + bubble_w / 2
                start_y = by + bubble_h
                end_x = target.rect.center.x
                end_y = target.rect.y

                ctx.set_source_rgba(ar, ag, ab, 0.6 * node.opacity)
                ctx.set_line_width(1.5)
                ctx.set_dash([4, 4])
                ctx.move_to(start_x, start_y)
                ctx.line_to(end_x, end_y)
                ctx.stroke()
                ctx.set_dash([])

    def _rounded_rect(self, ctx: cairo.Context, x: float, y: float, w: float, h: float, r: float) -> None:
        """Draw a rounded rectangle path."""
        r = min(r, w / 2, h / 2)
        ctx.new_sub_path()
        ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()

    def _sketch_stroke(self, ctx: cairo.Context) -> None:
        """Stroke with a sketchy/hand-drawn effect by adding slight jitter."""
        # Get the current path, jitter it, then stroke
        path = ctx.copy_path()
        ctx.new_path()

        rng = random.Random(hash(str(path)))
        roughness = self.theme.sketch_roughness

        for segment in path:
            seg_type = segment[0]
            if seg_type == 0:  # MOVE_TO
                x, y = segment[1]
                ctx.move_to(x + rng.gauss(0, roughness), y + rng.gauss(0, roughness))
            elif seg_type == 1:  # LINE_TO
                x, y = segment[1]
                ctx.line_to(x + rng.gauss(0, roughness), y + rng.gauss(0, roughness))
            elif seg_type == 2:  # CURVE_TO
                x1, y1, x2, y2, x3, y3 = segment[1]
                ctx.curve_to(
                    x1 + rng.gauss(0, roughness), y1 + rng.gauss(0, roughness),
                    x2 + rng.gauss(0, roughness), y2 + rng.gauss(0, roughness),
                    x3 + rng.gauss(0, roughness), y3 + rng.gauss(0, roughness),
                )
            elif seg_type == 3:  # CLOSE_PATH
                ctx.close_path()

        ctx.stroke()
