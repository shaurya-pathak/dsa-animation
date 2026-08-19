"""Cairo-based frame renderer.

Renders a single frame of the scene graph to a Cairo surface.
Handles all object types, animation state, transitions, and theme styling.
"""

from __future__ import annotations

import math
import random

import cairo

from kaivra.dsl.schema import ObjectType
from kaivra.scene_graph.models import ResolvedScene, SceneGraph, SceneNode
from kaivra.scene_graph.timeline import apply_animations_at_time
from kaivra.themes.base import ThemeSpec
from kaivra.utils.color import hex_to_rgba
from kaivra.utils.geometry import connector_endpoints
from kaivra.utils.typography import (
    aligned_equation_reveal,
    is_metric_style,
    metric_sign_slot_width,
    split_equation,
    split_metric_sign,
)


class CairoRenderer:
    """Renders scene graph frames using Cairo."""

    def __init__(self, theme: ThemeSpec):
        self.theme = theme

    def render_frame(self, graph: SceneGraph, time: float) -> cairo.ImageSurface:
        """Render a frame, fading scene boundaries through the neutral background."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, graph.width, graph.height)
        ctx = cairo.Context(surface)

        scene_idx, scene, scene_time = self._locate_scene(graph, time)
        if scene is None:
            self._fill_background(ctx, graph.width, graph.height)
            return surface

        # A fade spans both sides of the boundary: the outgoing scene fades to
        # the theme background, then the incoming scene fades up from it. Scenes
        # never overlap and the next scene is never pre-advanced then reset.
        scene_alpha = 1.0
        if scene.transition and scene.transition.duration > 0 and scene_idx + 1 < len(graph.scenes):
            fade_span = scene.transition.duration / 2.0
            time_remaining = scene.duration - scene_time
            if fade_span > 0 and time_remaining < fade_span:
                scene_alpha = max(0.0, min(1.0, time_remaining / fade_span))

        if scene_idx > 0:
            previous_transition = graph.scenes[scene_idx - 1].transition
            if previous_transition and previous_transition.duration > 0:
                fade_span = previous_transition.duration / 2.0
                if fade_span > 0 and scene_time < fade_span:
                    scene_alpha = min(
                        scene_alpha,
                        max(0.0, min(1.0, scene_time / fade_span)),
                    )

        if scene_alpha < 1.0:
            self._fill_background(ctx, graph.width, graph.height)
            scene_surface = self._render_scene_to_surface(graph, scene, scene_time)
            ctx.set_source_surface(scene_surface, 0, 0)
            ctx.paint_with_alpha(scene_alpha)
        else:
            self._draw_scene(ctx, graph, scene, scene_time)

        return surface

    def _render_scene_to_surface(
        self, graph: SceneGraph, scene: "ResolvedScene", scene_time: float
    ) -> cairo.ImageSurface:
        """Render a single scene to an offscreen surface."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, graph.width, graph.height)
        ctx = cairo.Context(surface)
        self._draw_scene(ctx, graph, scene, scene_time)
        return surface

    def _draw_scene(
        self, ctx: cairo.Context, graph: SceneGraph, scene: "ResolvedScene", scene_time: float
    ) -> None:
        """Draw a complete scene into an existing context."""
        apply_animations_at_time(scene.node_map, scene.timeline, scene_time)
        self._fill_background(ctx, graph.width, graph.height)
        ctx.save()
        for node in scene.nodes:
            self._draw_node(ctx, node, scene.node_map, scene_time)
        ctx.restore()
        ctx.save()
        if graph.show_narration and scene.narration:
            self._draw_narration(
                ctx, scene.narration, scene_time, scene.duration, graph.width, graph.height
            )
        if scene.show_progress_bar:
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

    def _locate_scene(
        self, graph: SceneGraph, time: float
    ) -> tuple[int, "ResolvedScene | None", float]:
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

    def _draw_node(
        self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode], scene_time: float
    ) -> None:
        if not node.visible:
            return

        ctx.save()

        # Idle motion (subtle float/jitter/breathe)
        idle_dx = 0.0
        idle_dy = 0.0
        idle_scale = 1.0
        if node.idle_preset:
            preset = node.idle_preset
            speed = node.idle_speed or 1.5
            if preset in {"float", "jitter"}:
                intensity = node.idle_intensity if node.idle_intensity is not None else 6.0
                freq = speed * (3.0 if preset == "jitter" else 1.0)
                axis = node.idle_axis or "both"
                if axis in {"x", "both"}:
                    idle_dx = math.sin(scene_time * freq) * intensity
                if axis in {"y", "both"}:
                    idle_dy = math.cos(scene_time * freq * 1.3) * intensity
            elif preset == "breathe":
                intensity = node.idle_intensity if node.idle_intensity is not None else 0.03
                idle_scale = 1.0 + math.sin(scene_time * speed) * intensity

        # Apply translate (move animation + idle)
        tx = node.translate_x + idle_dx
        ty = node.translate_y + idle_dy
        if tx != 0.0 or ty != 0.0:
            ctx.translate(tx, ty)

        sx = node.scale_x * idle_scale
        sy = node.scale_y * idle_scale
        shell_only_scale = not node.scale_text and node.obj_type in {
            ObjectType.BOX,
            ObjectType.TOKEN,
        }
        if shell_only_scale and (sx != 1.0 or sy != 1.0):
            ctx.save()
            cx, cy = node.rect.center.x, node.rect.center.y
            ctx.translate(cx, cy)
            ctx.scale(sx, sy)
            ctx.translate(-cx, -cy)
            self._draw_node_shell(ctx, node, node_map, scene_time)
            if node.highlight_intensity > 0:
                self._draw_highlight(ctx, node)
            ctx.restore()
            self._draw_node_text(ctx, node)
            ctx.restore()
            return

        # Apply scale transform around center
        if sx != 1.0 or sy != 1.0:
            cx, cy = node.rect.center.x, node.rect.center.y
            ctx.translate(cx, cy)
            ctx.scale(sx, sy)
            ctx.translate(-cx, -cy)

        self._draw_node_visual(ctx, node, node_map, scene_time)

        if node.highlight_intensity > 0:
            self._draw_highlight(ctx, node)

        ctx.restore()

    def _draw_node_visual(
        self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode], scene_time: float
    ) -> None:
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
                self._draw_group(ctx, node, node_map, scene_time)
            case ObjectType.CIRCLE:
                self._draw_circle(ctx, node)
            case ObjectType.LINEAR_METER:
                self._draw_linear_meter(ctx, node)
            case ObjectType.SIGMOID_PLOT:
                self._draw_sigmoid_plot(ctx, node)
            case ObjectType.PET_PORTRAIT | ObjectType.PET:
                self._draw_pet_portrait(ctx, node)
            case ObjectType.SEMANTIC_ICON:
                self._draw_semantic_icon(ctx, node)
            case ObjectType.CALLOUT:
                self._draw_callout(ctx, node, node_map)
            case _:
                self._draw_box(ctx, node)

    def _draw_node_shell(
        self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode], scene_time: float
    ) -> None:
        match node.obj_type:
            case ObjectType.BOX:
                self._draw_box_shell(ctx, node)
            case ObjectType.TOKEN:
                self._draw_token_shell(ctx, node)
            case _:
                self._draw_node_visual(ctx, node, node_map, scene_time)

    def _draw_node_text(self, ctx: cairo.Context, node: SceneNode) -> None:
        match node.obj_type:
            case ObjectType.BOX:
                self._draw_box_text(ctx, node)
            case ObjectType.TOKEN:
                self._draw_token_text(ctx, node)

    def _draw_text(self, ctx: cairo.Context, node: SceneNode) -> None:
        if not node.content:
            return

        style = node.style_props
        font_size = style.get("font_size", self.theme.font_size_body)
        font_family = style.get("font_family", self.theme.font_family)
        color = style.get("color", self.theme.text_color)
        weight = (
            cairo.FONT_WEIGHT_BOLD
            if style.get("font_weight") == "bold"
            else cairo.FONT_WEIGHT_NORMAL
        )

        ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, weight)
        ctx.set_font_size(font_size)

        r, g, b, _ = hex_to_rgba(color)
        ctx.set_source_rgba(r, g, b, node.opacity)

        # Handle typewriter effect without ever briefly rendering a metric sign
        # at full numeral scale.
        text = node.content
        sign, magnitude = split_metric_sign(text, node.style)
        is_metric = is_metric_style(node.style)
        if node.draw_progress < 1.0:
            chars_to_show = int(len(text) * node.draw_progress)
            if is_metric:
                if chars_to_show <= 0:
                    return
                magnitude = magnitude[: max(0, chars_to_show - (1 if sign is not None else 0))]
            else:
                text = text[:chars_to_show]

        if is_metric:
            self._draw_metric_text(
                ctx,
                node,
                sign=sign,
                magnitude=magnitude,
                font_family=font_family,
                font_size=font_size,
                weight=weight,
            )
            return

        if getattr(node, "align_equals", False) and split_equation(node.content) is not None:
            reveal = aligned_equation_reveal(node.content, node.draw_progress)
            if reveal is not None:
                self._draw_equation_text(
                    ctx,
                    node,
                    left=reveal[0],
                    show_equals=reveal[1],
                    right=reveal[2],
                )
                return

        if not text:
            return

        extents = ctx.text_extents(text)
        x = node.rect.x + (node.rect.width - extents.width) / 2
        y = node.rect.y + (node.rect.height + extents.height) / 2
        ctx.move_to(x, y)
        ctx.show_text(text)

    def _draw_metric_text(
        self,
        ctx: cairo.Context,
        node: SceneNode,
        *,
        sign: str | None,
        magnitude: str,
        font_family: str,
        font_size: float,
        weight: int,
    ) -> None:
        """Draw a metric inside a sign-reserved magnitude column."""
        ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, weight)
        ctx.set_font_size(font_size)
        magnitude_extents = ctx.text_extents(magnitude)
        sign_slot = metric_sign_slot_width(
            font_size,
            sign_scale=self.theme.metric_sign_scale,
            sign_gap=self.theme.metric_sign_gap,
        )
        magnitude_region_x = node.rect.x + sign_slot
        magnitude_region_width = max(0.0, node.rect.width - sign_slot)
        magnitude_x = magnitude_region_x + (magnitude_region_width - magnitude_extents.width) / 2
        baseline_y = node.rect.y + (node.rect.height + magnitude_extents.height) / 2

        if sign is None:
            ctx.move_to(magnitude_x, baseline_y)
            ctx.show_text(magnitude)
            return

        sign_size = font_size * self.theme.metric_sign_scale
        ctx.set_font_size(sign_size)
        sign_extents = ctx.text_extents(sign)
        sign_x = max(
            node.rect.x,
            magnitude_region_x - self.theme.metric_sign_gap - sign_extents.width,
        )
        magnitude_center_y = baseline_y + magnitude_extents.y_bearing + magnitude_extents.height / 2
        sign_baseline_y = magnitude_center_y - (sign_extents.y_bearing + sign_extents.height / 2)
        ctx.move_to(sign_x, sign_baseline_y)
        ctx.show_text(sign)

        ctx.set_font_size(font_size)
        ctx.move_to(magnitude_x, baseline_y)
        ctx.show_text(magnitude)

    def _draw_equation_text(
        self,
        ctx: cairo.Context,
        node: SceneNode,
        *,
        left: str,
        show_equals: bool,
        right: str,
    ) -> None:
        """Use the equals sign as a stable anchor across stacked equations."""
        equals = "="
        left_extents = ctx.text_extents(left)
        equals_extents = ctx.text_extents(equals)
        gap = max(8.0, ctx.get_font_matrix().xx * 0.28)
        equals_x = node.rect.center.x - equals_extents.width / 2
        baseline_y = node.rect.y + (node.rect.height + equals_extents.height) / 2
        if left:
            ctx.move_to(equals_x - gap - left_extents.width, baseline_y)
            ctx.show_text(left)
        if show_equals:
            ctx.move_to(equals_x, baseline_y)
            ctx.show_text(equals)
        if right:
            ctx.move_to(equals_x + equals_extents.width + gap, baseline_y)
            ctx.show_text(right)

    def _draw_box(self, ctx: cairo.Context, node: SceneNode) -> None:
        self._draw_box_shell(ctx, node)
        self._draw_box_text(ctx, node)

    def _draw_box_shell(self, ctx: cairo.Context, node: SceneNode) -> None:
        r = node.rect
        cr = self.theme.box_corner_radius
        style = node.style_props
        fill_color = style.get("fill", self.theme.box_fill)
        border_color = style.get("border", self.theme.box_border)

        # Shadow
        if self.theme.shadow:
            sr, sg, sb, sa = hex_to_rgba(self.theme.shadow_color)
            self._rounded_rect(
                ctx,
                r.x + self.theme.shadow_offset,
                r.y + self.theme.shadow_offset,
                r.width,
                r.height,
                cr,
            )
            ctx.set_source_rgba(sr, sg, sb, sa * node.opacity)
            ctx.fill()

        # Fill
        fr, fg, fb, _ = hex_to_rgba(fill_color)
        self._rounded_rect(ctx, r.x, r.y, r.width, r.height, cr)
        ctx.set_source_rgba(fr, fg, fb, node.opacity)
        ctx.fill_preserve()

        # Border
        br, bg, bb, _ = hex_to_rgba(border_color)
        ctx.set_source_rgba(br, bg, bb, node.opacity)
        ctx.set_line_width(self.theme.box_border_width * self._stroke_scale(node))
        if self.theme.sketch_effect:
            self._sketch_stroke(ctx)
        else:
            ctx.stroke()

    def _draw_box_text(self, ctx: cairo.Context, node: SceneNode) -> None:
        r = node.rect
        if node.content:
            style = node.style_props
            weight = (
                cairo.FONT_WEIGHT_BOLD
                if style.get("font_weight") == "bold"
                else cairo.FONT_WEIGHT_NORMAL
            )
            font_size = style.get("font_size", self.theme.font_size_body)
            font_family = style.get("font_family", self.theme.font_family)
            text_color = style.get("color", self.theme.text_color)
            ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, weight)
            ctx.set_font_size(font_size)
            tr, tg, tb, _ = hex_to_rgba(text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)

            text = node.content
            if node.draw_progress < 1.0:
                text = text[: int(len(text) * node.draw_progress)]

            extents = ctx.text_extents(text)
            x = r.x + (r.width - extents.width) / 2
            y = r.y + (r.height + extents.height) / 2
            ctx.move_to(x, y)
            ctx.show_text(text)

    def _draw_token(self, ctx: cairo.Context, node: SceneNode) -> None:
        self._draw_token_shell(ctx, node)
        self._draw_token_text(ctx, node)

    def _draw_token_shell(self, ctx: cairo.Context, node: SceneNode) -> None:
        r = node.rect
        cr = self.theme.token_corner_radius
        style = node.style_props
        fill_color = style.get("fill", self.theme.token_fill)
        border_color = style.get("border", self.theme.token_border)

        # Fill — fade in with draw_progress
        fr, fg, fb, _ = hex_to_rgba(fill_color)
        self._rounded_rect(ctx, r.x, r.y, r.width, r.height, cr)
        fill_opacity = node.opacity * min(1.0, node.draw_progress * 2)  # fill appears first half
        ctx.set_source_rgba(fr, fg, fb, fill_opacity)
        ctx.fill()

        # Border — draws progressively (perimeter stroke animation)
        br, bg, bb, _ = hex_to_rgba(border_color)
        ctx.set_source_rgba(br, bg, bb, node.opacity)
        ctx.set_line_width(2.0 * self._stroke_scale(node))
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

    def _draw_token_text(self, ctx: cairo.Context, node: SceneNode) -> None:
        r = node.rect
        if node.content:
            style = node.style_props
            weight = (
                cairo.FONT_WEIGHT_BOLD
                if style.get("font_weight") == "bold"
                else cairo.FONT_WEIGHT_NORMAL
            )
            font_size = style.get("font_size", self.theme.font_size_body)
            font_family = style.get("font_family", self.theme.font_family)
            text_color = style.get("color", self.theme.text_color)
            ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, weight)
            ctx.set_font_size(font_size)
            tr, tg, tb, _ = hex_to_rgba(text_color)
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

    def _draw_connector(
        self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode]
    ) -> None:
        if not node.from_id or not node.to_id:
            return

        from_node = node_map.get(node.from_id)
        to_node = node_map.get(node.to_id)
        if not from_node or not to_node:
            return

        start, end = connector_endpoints(from_node.rect, to_node.rect)

        connector_color = node.style_props.get(
            "border", node.style_props.get("color", self.theme.connector_color)
        )
        cr, cg, cb, _ = hex_to_rgba(connector_color)
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

        # A flow animation carries one restrained signal along an already drawn
        # connector. It makes causality visible without adding a new object type.
        if node.flow_progress is not None:
            signal_x = start.x + (end.x - start.x) * node.flow_progress
            signal_y = start.y + (end.y - start.y) * node.flow_progress
            ctx.new_path()
            ctx.arc(signal_x, signal_y, max(5.0, self.theme.connector_width * 2.2), 0, 2 * math.pi)
            ctx.set_source_rgba(cr, cg, cb, node.opacity)
            ctx.fill()

    def _draw_group(
        self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode], scene_time: float
    ) -> None:
        # Draw label if present
        if node.label:
            ctx.select_font_face(
                self.theme.font_family,
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD,
            )
            ctx.set_font_size(self.theme.font_size_caption)
            lr, lg, lb, _ = hex_to_rgba(self.theme.text_light)
            ctx.set_source_rgba(lr, lg, lb, node.opacity)
            extents = ctx.text_extents(node.label)
            x = node.rect.x + (node.rect.width - extents.width) / 2
            ctx.move_to(x, node.rect.y - 8)
            ctx.show_text(node.label)

        # Draw children
        for child in node.children:
            # Children keep their own animation state, but inherit the parent's envelope.
            child_visible = child.visible
            child_opacity = child.opacity
            child_draw_progress = child.draw_progress
            child.visible = node.visible and child_visible
            child.opacity = node.opacity * child_opacity
            child.draw_progress = node.draw_progress * child_draw_progress
            self._draw_node(ctx, child, node_map, scene_time)
            child.visible = child_visible
            child.opacity = child_opacity
            child.draw_progress = child_draw_progress

    def _draw_circle(self, ctx: cairo.Context, node: SceneNode) -> None:
        cx, cy = node.rect.center.x, node.rect.center.y
        radius = min(node.rect.width, node.rect.height) / 2
        style = node.style_props
        fill_color = style.get("fill", self.theme.box_fill)
        border_color = style.get("border", self.theme.box_border)
        text_color = style.get("color", self.theme.text_color)
        font_size = style.get("font_size", self.theme.font_size_body)

        fr, fg, fb, _ = hex_to_rgba(fill_color)
        ctx.new_path()
        ctx.arc(cx, cy, radius, 0, 2 * math.pi)
        ctx.set_source_rgba(fr, fg, fb, node.opacity)
        ctx.fill_preserve()

        br, bg, bb, _ = hex_to_rgba(border_color)
        ctx.set_source_rgba(br, bg, bb, node.opacity)
        ctx.set_line_width(self.theme.box_border_width * self._stroke_scale(node))
        ctx.stroke()

        if node.content:
            font_family = style.get("font_family", self.theme.font_family)
            weight = (
                cairo.FONT_WEIGHT_BOLD
                if style.get("font_weight") == "bold"
                else cairo.FONT_WEIGHT_NORMAL
            )
            ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, weight)
            ctx.set_font_size(font_size)
            tr, tg, tb, _ = hex_to_rgba(text_color)
            ctx.set_source_rgba(tr, tg, tb, node.opacity)
            extents = ctx.text_extents(node.content)
            ctx.move_to(cx - extents.width / 2, cy + extents.height / 2)
            ctx.show_text(node.content)

    def _draw_linear_meter(self, ctx: cairo.Context, node: SceneNode) -> None:
        """Draw a semantic, bounded horizontal teaching scale.

        The DSL supplies only the values and labels. Fixed internal geometry
        reserves enough room for the causal pointer, the numeric value, and
        the left/middle/right interpretation without authoring pixel offsets.
        """
        rect = node.rect
        style = node.style_props
        variant = node.size_variant.value
        variant_scale = 0.78 if variant == "compact" else (1.16 if variant == "hero" else 1.0)
        meter_color = style.get("fill") or style.get("border")
        if meter_color is None and node.style:
            meter_color = style.get("color")
        meter_color = meter_color or self.theme.accent

        # Keep an out-of-range value legible rather than drawing outside its
        # declared scale. The raw value remains in the scene graph for audits.
        span = node.meter_max - node.meter_min
        ratio = (node.meter_value - node.meter_min) / span
        ratio = max(0.0, min(1.0, ratio))

        caption_font = max(12.0, min(22.0 * variant_scale, rect.height * 0.19))
        value_font = max(12.0, min(23.0 * variant_scale, rect.height * 0.21))
        label_font = max(11.0, min(18.0 * variant_scale, rect.height * 0.16))
        has_labels = any((node.meter_left_label, node.meter_center_label, node.meter_right_label))
        padding_x = max(14.0 * variant_scale, rect.width * 0.055)
        padding_y = max(6.0 * variant_scale, rect.height * 0.055)
        caption_space = caption_font * 1.38 if node.meter_caption else 0.0
        value_space = value_font * 1.42 if node.meter_value_label else 0.0
        labels_space = label_font * 1.42 if has_labels else 0.0
        free_height = max(
            8.0, rect.height - (padding_y * 2 + caption_space + value_space + labels_space)
        )
        track_height = max(8.0 * variant_scale, min(20.0 * variant_scale, free_height * 0.48))
        track_x = rect.x + padding_x
        track_width = max(track_height * 2, rect.width - padding_x * 2)
        track_y = (
            rect.y + padding_y + caption_space + value_space + (free_height - track_height) / 2
        )
        track_center_y = track_y + track_height / 2
        pointer_x = track_x + track_width * ratio

        def set_hex(hex_color: str, alpha: float) -> None:
            red, green, blue, base_alpha = hex_to_rgba(hex_color)
            ctx.set_source_rgba(red, green, blue, base_alpha * alpha * node.opacity)

        def draw_text(text: str, x: float, baseline_y: float, align: str, color: str) -> None:
            extents = ctx.text_extents(text)
            if align == "left":
                text_x = x - extents.x_bearing
            elif align == "right":
                text_x = x - extents.width - extents.x_bearing
            else:
                text_x = x - extents.width / 2 - extents.x_bearing
            set_hex(color, 1.0)
            ctx.move_to(text_x, baseline_y)
            ctx.show_text(text)

        # A softly outlined rail makes the full possible range visible before
        # the fill shows where this particular input lands.
        self._rounded_rect(ctx, track_x, track_y, track_width, track_height, track_height / 2)
        set_hex(self.theme.muted, 0.22)
        ctx.fill_preserve()
        set_hex(self.theme.primary, 0.36)
        ctx.set_line_width(max(1.0, self.theme.box_border_width * 0.62 * variant_scale))
        ctx.stroke()

        fill_width = track_width * ratio
        if fill_width > 0.5:
            self._rounded_rect(
                ctx,
                track_x,
                track_y,
                fill_width,
                track_height,
                min(track_height / 2, fill_width / 2),
            )
            set_hex(meter_color, 0.64)
            ctx.fill()

        # Fixed endpoint and midpoint ticks support both unipolar clue
        # strengths and bipolar CAT / 0 / DOG interpretations.
        tick_height = track_height * 1.28
        set_hex(self.theme.primary, 0.44)
        ctx.set_line_width(max(1.0, self.theme.box_border_width * 0.48 * variant_scale))
        for tick_x in (track_x, track_x + track_width / 2, track_x + track_width):
            ctx.move_to(tick_x, track_center_y - tick_height / 2)
            ctx.line_to(tick_x, track_center_y + tick_height / 2)
            ctx.stroke()

        pointer_radius = max(6.0 * variant_scale, track_height * 0.72)
        ctx.new_path()
        ctx.arc(pointer_x, track_center_y, pointer_radius, 0, 2 * math.pi)
        set_hex(self.theme.background_color, 1.0)
        ctx.fill_preserve()
        set_hex(meter_color, 1.0)
        ctx.set_line_width(max(1.6, self.theme.box_border_width * variant_scale))
        ctx.stroke()
        ctx.new_path()
        ctx.arc(pointer_x, track_center_y, pointer_radius * 0.38, 0, 2 * math.pi)
        set_hex(meter_color, 1.0)
        ctx.fill()

        label_color = self.theme.text_light
        if node.meter_caption:
            ctx.select_font_face(
                style.get("font_family", self.theme.font_family),
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD,
            )
            ctx.set_font_size(caption_font)
            caption_extents = ctx.text_extents(node.meter_caption)
            caption_baseline = (
                rect.y
                + padding_y
                + caption_space / 2
                - (caption_extents.y_bearing + caption_extents.height / 2)
            )
            draw_text(
                node.meter_caption,
                rect.center.x,
                caption_baseline,
                "center",
                self.theme.text_color,
            )

        if node.meter_value_label:
            ctx.select_font_face(
                style.get("font_family", self.theme.font_family),
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD,
            )
            ctx.set_font_size(value_font)
            value_extents = ctx.text_extents(node.meter_value_label)
            value_x = min(
                max(pointer_x, track_x + value_extents.width / 2),
                track_x + track_width - value_extents.width / 2,
            )
            value_baseline = track_y - value_font * 0.28
            draw_text(node.meter_value_label, value_x, value_baseline, "center", meter_color)

        if has_labels:
            ctx.select_font_face(
                style.get("font_family", self.theme.font_family),
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_NORMAL,
            )
            ctx.set_font_size(label_font)
            label_baseline = track_y + track_height + label_font * 1.18
            if node.meter_left_label:
                draw_text(node.meter_left_label, track_x, label_baseline, "left", label_color)
            if node.meter_center_label:
                draw_text(
                    node.meter_center_label,
                    track_x + track_width / 2,
                    label_baseline,
                    "center",
                    label_color,
                )
            if node.meter_right_label:
                draw_text(
                    node.meter_right_label,
                    track_x + track_width,
                    label_baseline,
                    "right",
                    label_color,
                )

    def _draw_sigmoid_plot(self, ctx: cairo.Context, node: SceneNode) -> None:
        """Draw an S-curve with a traceable score-to-probability point."""
        rect = node.rect
        variant = node.size_variant.value
        variant_scale = 0.82 if variant == "compact" else (1.14 if variant == "hero" else 1.0)
        left = rect.x + max(42.0, rect.width * 0.13)
        right = rect.right - max(18.0, rect.width * 0.055)
        top = rect.y + max(38.0, rect.height * 0.17)
        bottom = rect.bottom - max(40.0, rect.height * 0.17)
        plot_width = max(1.0, right - left)
        plot_height = max(1.0, bottom - top)
        axis_x = left + plot_width / 2.0
        mid_y = top + plot_height / 2.0
        input_value = max(-4.0, min(4.0, node.sigmoid_input))
        probability = 1.0 / (1.0 + math.exp(-input_value))
        point_x = left + ((input_value + 4.0) / 8.0) * plot_width
        point_y = bottom - probability * plot_height

        def set_hex(color: str, alpha: float = 1.0) -> None:
            red, green, blue, base_alpha = hex_to_rgba(color)
            ctx.set_source_rgba(red, green, blue, base_alpha * alpha * node.opacity)

        def draw_label(text: str, x: float, y: float, *, align: str = "center") -> None:
            extents = ctx.text_extents(text)
            if align == "left":
                text_x = x - extents.x_bearing
            elif align == "right":
                text_x = x - extents.width - extents.x_bearing
            else:
                text_x = x - extents.width / 2.0 - extents.x_bearing
            ctx.move_to(text_x, y)
            ctx.show_text(text)

        ctx.save()
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        set_hex(self.theme.muted, 0.34)
        ctx.set_line_width(max(1.0, self.theme.box_border_width * 0.48 * variant_scale))
        ctx.set_dash([5.0 * variant_scale, 6.0 * variant_scale])
        ctx.move_to(left, mid_y)
        ctx.line_to(right, mid_y)
        ctx.stroke()
        ctx.set_dash([])

        set_hex(self.theme.primary, 0.68)
        ctx.set_line_width(max(1.2, self.theme.box_border_width * 0.62 * variant_scale))
        ctx.move_to(left, bottom)
        ctx.line_to(right, bottom)
        ctx.move_to(axis_x, top)
        ctx.line_to(axis_x, bottom)
        ctx.stroke()

        sample_count = 96
        progress = max(0.0, min(1.0, node.draw_progress))
        visible_samples = max(2, round(sample_count * progress))
        set_hex(self.theme.channel_gold)
        ctx.set_line_width(max(3.0, self.theme.connector_width * 1.35 * variant_scale))
        ctx.new_path()
        for index in range(visible_samples):
            x_value = -4.0 + 8.0 * index / (sample_count - 1)
            y_value = 1.0 / (1.0 + math.exp(-x_value))
            x = left + ((x_value + 4.0) / 8.0) * plot_width
            y = bottom - y_value * plot_height
            if index == 0:
                ctx.move_to(x, y)
            else:
                ctx.line_to(x, y)
        ctx.stroke()

        caption_font = max(13.0, min(20.0 * variant_scale, rect.height * 0.08))
        label_font = max(11.0, min(17.0 * variant_scale, rect.height * 0.068))
        ctx.select_font_face(
            self.theme.font_family,
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_BOLD,
        )
        ctx.set_font_size(caption_font)
        set_hex(self.theme.text_color)
        if node.sigmoid_caption:
            draw_label(node.sigmoid_caption, rect.x + rect.width / 2.0, rect.y + caption_font)

        ctx.set_font_size(label_font)
        set_hex(self.theme.text_light)
        draw_label("SCORE", right, rect.bottom - 8.0, align="right")
        draw_label("CHANCE", left, top - 8.0, align="left")
        draw_label("50%", left - 8.0, mid_y + label_font * 0.35, align="right")

        if progress >= 0.9:
            set_hex(self.theme.channel_gold, 0.62)
            ctx.set_dash([4.0 * variant_scale, 5.0 * variant_scale])
            ctx.set_line_width(max(1.2, self.theme.box_border_width * 0.55 * variant_scale))
            ctx.move_to(point_x, bottom)
            ctx.line_to(point_x, point_y)
            ctx.line_to(left, point_y)
            ctx.stroke()
            ctx.set_dash([])

            set_hex(self.theme.background_color)
            ctx.new_path()
            ctx.arc(point_x, point_y, 7.0 * variant_scale, 0, 2 * math.pi)
            ctx.fill_preserve()
            set_hex(self.theme.channel_gold)
            ctx.set_line_width(max(2.0, self.theme.box_border_width * variant_scale))
            ctx.stroke()
            ctx.new_path()
            ctx.arc(point_x, point_y, 2.7 * variant_scale, 0, 2 * math.pi)
            ctx.fill()

            ctx.set_font_size(label_font)
            set_hex(self.theme.channel_gold)
            input_label = node.sigmoid_input_label or f"{node.sigmoid_input:+.2f}"
            output_label = node.sigmoid_output_label or f"{probability * 100:.0f}%"
            draw_label(input_label, point_x, bottom + label_font * 1.45)
            draw_label(output_label, left - 8.0, point_y + label_font * 0.35, align="right")

        ctx.restore()

    def _draw_pet_portrait(self, ctx: cairo.Context, node: SceneNode) -> None:
        """Draw a deterministic pet portrait with optional semantic feature cues."""
        rect = node.rect
        caption_height = min(36.0, rect.height * 0.18) if node.content else 0.0
        illustration_height = max(1.0, rect.height - caption_height)
        side = min(rect.width, illustration_height)
        cx = rect.x + rect.width / 2
        cy = rect.y + illustration_height * 0.54
        radius = side * 0.29
        style = node.style_props
        outline = self.theme.primary
        accent = style.get("border", self.theme.accent)
        outline_width = max(1.6, self.theme.box_border_width * self._stroke_scale(node))
        face_fill = "#F2C99D"
        floppy_fill = "#B97854"
        cat_fill = "#C88466"
        inner_ear_fill = "#EAA295"
        muzzle_fill = "#FFF3E5"
        feature_fill = "#392D29"
        kind = getattr(node.pet_kind, "value", node.pet_kind)
        highlights = {getattr(feature, "value", feature) for feature in (node.pet_highlights or [])}

        def set_fill(color: str) -> None:
            red, green, blue, _ = hex_to_rgba(color)
            ctx.set_source_rgba(red, green, blue, node.opacity)

        def paint_path(fill: str, border: str = outline, width: float = outline_width) -> None:
            set_fill(fill)
            ctx.fill_preserve()
            set_fill(border)
            ctx.set_line_width(width)
            ctx.stroke()

        def fill_path(fill: str) -> None:
            set_fill(fill)
            ctx.fill()

        def floppy_ear_path(direction: float) -> None:
            def x(value: float) -> float:
                return cx + direction * value * radius

            def y(value: float) -> float:
                return cy + value * radius

            ctx.new_path()
            ctx.move_to(x(0.60), y(-0.56))
            ctx.curve_to(x(1.26), y(-1.00), x(1.43), y(-0.08), x(0.99), y(0.33))
            ctx.curve_to(x(0.80), y(0.49), x(0.66), y(0.17), x(0.60), y(-0.18))
            ctx.close_path()

        def cat_ear_path(direction: float) -> None:
            def x(value: float) -> float:
                return cx + direction * value * radius

            def y(value: float) -> float:
                return cy + value * radius

            ctx.new_path()
            ctx.move_to(x(0.48), y(-0.57))
            ctx.line_to(x(0.83), y(-1.34))
            ctx.line_to(x(0.14), y(-0.91))
            ctx.line_to(x(0.08), y(-0.37))
            ctx.close_path()

        def draw_floppy_ear(direction: float) -> None:
            floppy_ear_path(direction)
            paint_path(floppy_fill)

        def draw_cat_ear(direction: float) -> None:
            cat_ear_path(direction)
            paint_path(cat_fill)

            def x(value: float) -> float:
                return cx + direction * value * radius

            def y(value: float) -> float:
                return cy + value * radius

            ctx.new_path()
            ctx.move_to(x(0.52), y(-0.64))
            ctx.line_to(x(0.76), y(-1.13))
            ctx.line_to(x(0.24), y(-0.84))
            ctx.close_path()
            fill_path(inner_ear_fill)

        def draw_snout(border: str | None = None, width: float = outline_width) -> None:
            """Use one long light oval so the feature reads as a snout, not a mouth."""
            ctx.save()
            ctx.translate(cx, cy + radius * 0.31)
            ctx.scale(1.42, 0.72)
            ctx.new_path()
            ctx.arc(0, 0, radius * 0.36, 0, 2 * math.pi)
            set_fill(muzzle_fill)
            if border is None:
                ctx.fill()
            else:
                ctx.fill_preserve()
                set_fill(border)
                ctx.set_line_width(width)
                ctx.stroke()
            ctx.restore()

        def outline_current_path(color: str, width: float) -> None:
            set_fill(color)
            ctx.set_line_width(width)
            ctx.stroke()

        def draw_feature_label(
            text: str,
            color: str,
            center_x: float,
            center_y: float,
            target_x: float,
            target_y: float,
            attach_from: str,
        ) -> None:
            font_size = max(11.0, min(18.0, side * 0.055))
            ctx.select_font_face(
                self.theme.font_family,
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD,
            )
            ctx.set_font_size(font_size)
            extents = ctx.text_extents(text)
            box_width = extents.width + 18.0
            box_height = max(font_size * 1.52, 24.0)
            box_x = center_x - box_width / 2
            box_y = center_y - box_height / 2
            line_x = box_x if attach_from == "left" else box_x + box_width

            # A compact leader and endpoint dot make the color an explanation
            # anchor, rather than a decorative border.
            set_fill(color)
            ctx.set_line_width(max(1.35, outline_width * 0.62))
            ctx.move_to(line_x, center_y)
            ctx.line_to(target_x, target_y)
            ctx.stroke()
            ctx.new_path()
            ctx.arc(target_x, target_y, max(2.5, outline_width), 0, 2 * math.pi)
            ctx.fill()

            ctx.new_path()
            self._rounded_rect(ctx, box_x, box_y, box_width, box_height, box_height / 2)
            set_fill(self.theme.background_color)
            ctx.fill_preserve()
            set_fill(color)
            ctx.set_line_width(max(1.35, outline_width * 0.62))
            ctx.stroke()
            set_fill(color)
            baseline = center_y - (extents.y_bearing + extents.height / 2)
            ctx.move_to(center_x - extents.width / 2 - extents.x_bearing, baseline)
            ctx.show_text(text)

        ctx.save()
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        if kind == "dog":
            draw_floppy_ear(-1.0)
            draw_floppy_ear(1.0)
        elif kind == "cat":
            draw_cat_ear(-1.0)
            draw_cat_ear(1.0)
        else:
            draw_floppy_ear(-1.0)
            draw_cat_ear(1.0)

        ctx.new_path()
        ctx.arc(cx, cy, radius, 0, 2 * math.pi)
        paint_path(face_fill)

        for eye_x in (cx - radius * 0.34, cx + radius * 0.34):
            ctx.new_path()
            ctx.arc(eye_x, cy - radius * 0.08, max(2.2, radius * 0.072), 0, 2 * math.pi)
            fill_path(feature_fill)
            ctx.new_path()
            ctx.arc(
                eye_x - radius * 0.018,
                cy - radius * 0.105,
                max(0.8, radius * 0.018),
                0,
                2 * math.pi,
            )
            fill_path("#FFFFFF")

        # The light, elongated oval establishes the long-snout cue before a
        # cyan highlight makes it explicit for a teaching scene.
        draw_snout(border="#D5A273", width=max(1.2, outline_width * 0.46))

        ctx.new_path()
        ctx.move_to(cx, cy + radius * 0.22)
        ctx.line_to(cx - radius * 0.11, cy + radius * 0.10)
        ctx.line_to(cx + radius * 0.11, cy + radius * 0.10)
        ctx.close_path()
        fill_path(feature_fill)
        ctx.new_path()
        set_fill(feature_fill)
        ctx.set_line_width(max(1.4, outline_width * 0.72))
        ctx.move_to(cx, cy + radius * 0.22)
        ctx.curve_to(
            cx - radius * 0.03,
            cy + radius * 0.40,
            cx - radius * 0.21,
            cy + radius * 0.43,
            cx - radius * 0.26,
            cy + radius * 0.32,
        )
        ctx.move_to(cx, cy + radius * 0.22)
        ctx.curve_to(
            cx + radius * 0.03,
            cy + radius * 0.40,
            cx + radius * 0.21,
            cy + radius * 0.43,
            cx + radius * 0.26,
            cy + radius * 0.32,
        )
        ctx.stroke()

        whisker_sides = (-1.0, 1.0) if kind == "cat" else ((1.0,) if kind == "mystery" else ())
        for direction in whisker_sides:
            for vertical_offset in (-0.03, 0.10):
                ctx.new_path()
                ctx.move_to(
                    cx + direction * radius * 0.53,
                    cy + radius * (0.22 + vertical_offset),
                )
                ctx.line_to(
                    cx + direction * radius * 1.10,
                    cy + radius * (0.16 + vertical_offset * 1.8),
                )
                ctx.stroke()

        if "ears" in highlights:
            coral = self.theme.channel_coral
            ear_width = max(2.4, outline_width * 1.38)
            if kind == "dog":
                for direction in (-1.0, 1.0):
                    floppy_ear_path(direction)
                    outline_current_path(coral, ear_width)
            elif kind == "cat":
                for direction in (-1.0, 1.0):
                    cat_ear_path(direction)
                    outline_current_path(coral, ear_width)
            else:
                # The mystery animal intentionally calls out its floppy dog
                # ear, which is the clue a novice should notice first.
                floppy_ear_path(-1.0)
                outline_current_path(coral, ear_width)

        if "snout" in highlights:
            draw_snout(border=self.theme.channel_cyan, width=max(2.4, outline_width * 1.38))

        if kind == "mystery":
            badge_radius = max(10.0, radius * 0.27)
            badge_x = cx + radius * 0.87
            badge_y = cy - radius * 0.95
            ctx.new_path()
            ctx.arc(badge_x, badge_y, badge_radius, 0, 2 * math.pi)
            paint_path(self.theme.background_color, accent, max(1.5, outline_width * 0.78))
            ctx.select_font_face(
                self.theme.font_family,
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD,
            )
            badge_font_size = max(14.0, radius * 0.52)
            ctx.set_font_size(badge_font_size)
            set_fill(accent)
            extents = ctx.text_extents("?")
            ctx.move_to(
                badge_x - extents.width / 2 - extents.x_bearing,
                badge_y - (extents.y_bearing + extents.height / 2),
            )
            ctx.show_text("?")
        ctx.restore()

        if node.show_feature_labels:
            gutter = max(0.0, (rect.width - side) / 2)
            if "ears" in highlights:
                ears_label = "POINTED EARS" if kind == "cat" else "FLOPPY EARS"
                draw_feature_label(
                    ears_label,
                    self.theme.channel_coral,
                    rect.x + gutter / 2,
                    cy - radius * 0.52,
                    cx - radius * 1.01,
                    cy - radius * 0.36,
                    "right",
                )
            if "snout" in highlights:
                draw_feature_label(
                    "LONG SNOUT",
                    self.theme.channel_cyan,
                    rect.right - gutter / 2,
                    cy + radius * 0.36,
                    cx + radius * 0.50,
                    cy + radius * 0.31,
                    "left",
                )

        if node.content:
            font_size = min(
                float(style.get("font_size", self.theme.font_size_caption)),
                max(13.0, side * 0.12),
            )
            weight = (
                cairo.FONT_WEIGHT_BOLD
                if style.get("font_weight") == "bold"
                else cairo.FONT_WEIGHT_NORMAL
            )
            ctx.select_font_face(self.theme.font_family, cairo.FONT_SLANT_NORMAL, weight)
            ctx.set_font_size(font_size)
            set_fill(self.theme.text_color)
            extents = ctx.text_extents(node.content)
            baseline = (
                rect.y + rect.height - caption_height / 2 - (extents.y_bearing + extents.height / 2)
            )
            ctx.move_to(cx - extents.width / 2 - extents.x_bearing, baseline)
            ctx.show_text(node.content)

    def _draw_semantic_icon(self, ctx: cairo.Context, node: SceneNode) -> None:
        """Draw one of the small, stable illustrations used in explainer diagrams.

        These intentionally use simple, flat geometry rather than an external icon
        font or image asset. That makes the illustration available in a headless
        render container and keeps Canvas/Cairo parity reviewable.
        """
        rect = node.rect
        caption_height = min(30.0, rect.height * 0.22) if node.content else 0.0
        side = min(rect.width, max(1.0, rect.height - caption_height))
        x = rect.x + (rect.width - side) / 2
        y = rect.y
        scale = side / 100.0
        style = node.style_props
        ink = self.theme.primary
        paper = self.theme.background_color
        cyan = style.get("border", self.theme.channel_cyan)
        coral = self.theme.channel_coral
        gold = self.theme.channel_gold
        line_width = max(1.5, self.theme.box_border_width * self._stroke_scale(node))

        def set_fill(color: str) -> None:
            red, green, blue, _ = hex_to_rgba(color)
            ctx.set_source_rgba(red, green, blue, node.opacity)

        def rounded(x0: float, y0: float, width: float, height: float, radius: float) -> None:
            self._rounded_rect(ctx, x0, y0, width, height, radius)

        def fill_stroke(fill: str, stroke: str = ink, width: float = line_width) -> None:
            set_fill(fill)
            ctx.fill_preserve()
            set_fill(stroke)
            ctx.set_line_width(width)
            ctx.stroke()

        def stroke(color: str = ink, width: float = line_width) -> None:
            set_fill(color)
            ctx.set_line_width(width)
            ctx.stroke()

        def arrow(x0: float, y0: float, x1: float, y1: float, color: str) -> None:
            set_fill(color)
            ctx.set_line_width(line_width * 1.1)
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.stroke()
            angle = math.atan2(y1 - y0, x1 - x0)
            head = 8.0
            for direction in (math.pi * 0.82, -math.pi * 0.82):
                ctx.move_to(x1, y1)
                ctx.line_to(
                    x1 + math.cos(angle + direction) * head, y1 + math.sin(angle + direction) * head
                )
            ctx.stroke()

        def draw_cash_note(x0: float, y0: float, width: float, height: float) -> None:
            rounded(x0, y0, width, height, 5.0)
            fill_stroke("#E4F0ED", cyan)
            ctx.new_path()
            ctx.arc(x0 + width / 2, y0 + height / 2, height * 0.22, 0, 2 * math.pi)
            fill_stroke(paper, cyan, line_width * 0.7)
            ctx.select_font_face(
                self.theme.font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            ctx.set_font_size(height * 0.50)
            set_fill(ink)
            extents = ctx.text_extents("$")
            ctx.move_to(
                x0 + width / 2 - extents.width / 2 - extents.x_bearing,
                y0 + height / 2 - (extents.y_bearing + extents.height / 2),
            )
            ctx.show_text("$")

        icon = node.icon_name or "fund"
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(scale, scale)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        if icon == "fund":
            for base_x, height, color in ((18, 22, coral), (39, 38, cyan), (63, 29, gold)):
                rounded(base_x, 79 - height, 18, height, 4)
                fill_stroke(paper, color)
                for line_y in range(int(79 - height + 7), 79, 8):
                    ctx.move_to(base_x + 3, line_y)
                    ctx.line_to(base_x + 15, line_y)
                    stroke(color, line_width * 0.55)
            ctx.new_path()
            ctx.arc(50, 24, 15, 0, 2 * math.pi)
            fill_stroke("#F6E9C9", gold)
            ctx.select_font_face(
                self.theme.font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            ctx.set_font_size(17)
            set_fill(ink)
            extents = ctx.text_extents("$")
            ctx.move_to(
                50 - extents.width / 2 - extents.x_bearing,
                24 - (extents.y_bearing + extents.height / 2),
            )
            ctx.show_text("$")
        elif icon == "storefront":
            rounded(17, 46, 66, 37, 3)
            fill_stroke(paper, ink)
            ctx.new_path()
            ctx.move_to(12, 45)
            ctx.line_to(88, 45)
            ctx.line_to(80, 28)
            ctx.line_to(20, 28)
            ctx.close_path()
            fill_stroke("#E4F0ED", cyan)
            for stripe_x in (23, 38, 53, 68):
                ctx.new_path()
                ctx.move_to(stripe_x, 29)
                ctx.line_to(stripe_x + 9, 29)
                ctx.line_to(stripe_x + 3, 45)
                ctx.line_to(stripe_x - 6, 45)
                ctx.close_path()
                set_fill(coral if stripe_x in (23, 53) else paper)
                ctx.fill()
            rounded(43, 59, 14, 24, 2)
            fill_stroke("#F6E9C9", gold)
            ctx.new_path()
            ctx.arc(53, 70, 1.5, 0, 2 * math.pi)
            set_fill(ink)
            ctx.fill()
        elif icon == "warehouse":
            ctx.new_path()
            ctx.move_to(12, 43)
            ctx.line_to(50, 18)
            ctx.line_to(88, 43)
            ctx.line_to(88, 82)
            ctx.line_to(12, 82)
            ctx.close_path()
            fill_stroke("#E4F0ED", cyan)
            for door_x in (24, 45, 66):
                rounded(door_x, 55, 12, 27, 1)
                fill_stroke(paper, ink, line_width * 0.7)
            ctx.move_to(12, 43)
            ctx.line_to(88, 43)
            stroke(ink)
        elif icon == "cash":
            draw_cash_note(14, 28, 72, 44)
        elif icon == "shares":
            for offset_x, offset_y, color in ((12, 34, coral), (20, 25, gold), (28, 16, cyan)):
                rounded(offset_x, offset_y, 58, 48, 4)
                fill_stroke(paper, color)
                ctx.move_to(offset_x + 13, offset_y + 17)
                ctx.line_to(offset_x + 45, offset_y + 17)
                ctx.move_to(offset_x + 13, offset_y + 28)
                ctx.line_to(offset_x + 35, offset_y + 28)
                stroke(ink, line_width * 0.62)
                ctx.new_path()
                ctx.arc(offset_x + 45, offset_y + 32, 5, 0, 2 * math.pi)
                set_fill(color)
                ctx.fill()
        elif icon == "borrow":
            draw_cash_note(18, 18, 48, 30)
            arrow(45, 53, 45, 76, coral)
            ctx.new_path()
            ctx.move_to(20, 78)
            ctx.line_to(30, 67)
            ctx.line_to(60, 67)
            ctx.line_to(74, 78)
            ctx.line_to(70, 84)
            ctx.line_to(25, 84)
            ctx.close_path()
            fill_stroke("#F6E9C9", gold)
        elif icon == "loan":
            ctx.new_path()
            ctx.arc(29, 48, 18, 0, 2 * math.pi)
            fill_stroke("#F6E9C9", gold)
            ctx.select_font_face(
                self.theme.font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            ctx.set_font_size(20)
            set_fill(ink)
            extents = ctx.text_extents("$")
            ctx.move_to(
                29 - extents.width / 2 - extents.x_bearing,
                48 - (extents.y_bearing + extents.height / 2),
            )
            ctx.show_text("$")
            arrow(49, 48, 77, 48, cyan)
            rounded(63, 59, 24, 23, 3)
            fill_stroke(paper, ink)
            ctx.move_to(69, 67)
            ctx.line_to(81, 67)
            ctx.move_to(69, 74)
            ctx.line_to(78, 74)
            stroke(ink, line_width * 0.65)
        elif icon == "handshake":
            ctx.new_path()
            ctx.move_to(13, 44)
            ctx.line_to(32, 30)
            ctx.line_to(53, 47)
            ctx.line_to(44, 59)
            ctx.close_path()
            fill_stroke("#E4F0ED", cyan)
            ctx.new_path()
            ctx.move_to(87, 44)
            ctx.line_to(68, 30)
            ctx.line_to(46, 49)
            ctx.line_to(55, 61)
            ctx.close_path()
            fill_stroke("#F6E9C9", gold)
            for index in range(3):
                x0 = 46 + index * 5
                y0 = 53 + index * 5
                ctx.move_to(x0, y0)
                ctx.line_to(x0 + 9, y0 + 7)
                stroke(coral, line_width * 0.75)

        ctx.restore()

        if node.content:
            font_size = min(
                float(style.get("font_size", self.theme.font_size_caption)),
                max(12.0, side * 0.15),
            )
            weight = (
                cairo.FONT_WEIGHT_BOLD
                if style.get("font_weight") == "bold"
                else cairo.FONT_WEIGHT_NORMAL
            )
            ctx.select_font_face(self.theme.font_family, cairo.FONT_SLANT_NORMAL, weight)
            ctx.set_font_size(font_size)
            set_fill(self.theme.text_color)
            extents = ctx.text_extents(node.content)
            baseline = (
                rect.y + rect.height - caption_height / 2 - (extents.y_bearing + extents.height / 2)
            )
            ctx.move_to(rect.x + rect.width / 2 - extents.width / 2 - extents.x_bearing, baseline)
            ctx.show_text(node.content)

    def _stroke_scale(self, node: SceneNode) -> float:
        variant = node.style_props.get("size_variant")
        if variant == "compact":
            return 0.72
        if variant == "hero":
            return 1.15
        return 1.0

    def _draw_highlight(self, ctx: cairo.Context, node: SceneNode) -> None:
        """Draw a glow/highlight overlay on the node."""
        color = node.highlight_color or "accent"
        hex_color = self.theme.resolve_color(color)
        hr, hg, hb, _ = hex_to_rgba(hex_color)

        intensity = node.highlight_intensity * node.opacity

        if node.obj_type in {ObjectType.CIRCLE, ObjectType.PET_PORTRAIT, ObjectType.PET}:
            # Circular radial gradient glow halo for circles
            cx, cy = node.rect.center.x, node.rect.center.y
            radius = min(node.rect.width, node.rect.height) / 2
            glow_radius = radius * 2.8
            pattern = cairo.RadialGradient(cx, cy, radius * 0.8, cx, cy, glow_radius)
            pattern.add_color_stop_rgba(0.0, hr, hg, hb, intensity * 0.55)
            pattern.add_color_stop_rgba(0.35, hr, hg, hb, intensity * 0.25)
            pattern.add_color_stop_rgba(1.0, hr, hg, hb, 0.0)
            ctx.new_path()
            ctx.arc(cx, cy, glow_radius, 0, 2 * math.pi)
            ctx.set_source(pattern)
            ctx.fill()
        else:
            r = node.rect
            ctx.set_source_rgba(hr, hg, hb, intensity * 0.3)
            self._rounded_rect(
                ctx, r.x - 4, r.y - 4, r.width + 8, r.height + 8, self.theme.box_corner_radius + 4
            )
            ctx.fill()

    def _draw_narration(
        self,
        ctx: cairo.Context,
        text: str,
        scene_time: float,
        scene_duration: float,
        w: int,
        h: int,
    ) -> None:
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

    def _draw_progress_bar(
        self, ctx: cairo.Context, scene_time: float, scene_duration: float, w: int
    ) -> None:
        """Draw a thin accent-colored progress bar at the top of the screen."""
        if scene_duration <= 0:
            return
        progress = scene_time / scene_duration
        bar_h = 3
        ar, ag, ab, _ = hex_to_rgba(self.theme.accent)
        ctx.set_source_rgba(ar, ag, ab, 0.6)
        ctx.rectangle(0, 0, w * progress, bar_h)
        ctx.fill()

    def _draw_callout(
        self, ctx: cairo.Context, node: SceneNode, node_map: dict[str, SceneNode]
    ) -> None:
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

    def _rounded_rect(
        self, ctx: cairo.Context, x: float, y: float, w: float, h: float, r: float
    ) -> None:
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
                    x1 + rng.gauss(0, roughness),
                    y1 + rng.gauss(0, roughness),
                    x2 + rng.gauss(0, roughness),
                    y2 + rng.gauss(0, roughness),
                    x3 + rng.gauss(0, roughness),
                    y3 + rng.gauss(0, roughness),
                )
            elif seg_type == 3:  # CLOSE_PATH
                ctx.close_path()

        ctx.stroke()
