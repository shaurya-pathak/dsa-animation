"""Build a SceneGraph from a validated DocumentSpec."""

from __future__ import annotations

from dsa_anim.dsl.schema import (
    DocumentSpec, SceneSpec, ObjectSpec, LayoutSpec, LayoutType,
    AnimSpec, CameraAnimSpec, parse_duration,
)
from dsa_anim.scene_graph.models import (
    SceneGraph, ResolvedScene, SceneNode, AnimationKeyframe,
    CameraState, CameraKeyframe, TransitionInfo,
)
from dsa_anim.layout.engine import LayoutEngine
from dsa_anim.themes.base import ThemeSpec
from dsa_anim.utils.geometry import Rect


def build_scene_graph(doc: DocumentSpec, theme: ThemeSpec) -> SceneGraph:
    """Convert a DocumentSpec into a fully resolved SceneGraph."""
    width, height = doc.meta.resolution
    layout_engine = LayoutEngine(theme)
    canvas = Rect(theme.margin, theme.margin, width - theme.margin * 2, height - theme.margin * 2)

    scenes = []
    for scene_spec in doc.scenes:
        resolved = _build_scene(scene_spec, layout_engine, theme, canvas)
        scenes.append(resolved)

    return SceneGraph(
        width=width,
        height=height,
        fps=doc.meta.fps,
        theme_name=doc.meta.theme,
        scenes=scenes,
    )


def _build_scene(
    spec: SceneSpec,
    layout_engine: LayoutEngine,
    theme: ThemeSpec,
    canvas: Rect,
) -> ResolvedScene:
    # Resolve layout
    layout = spec.layout if isinstance(spec.layout, LayoutSpec) else LayoutSpec(type=LayoutType.CENTER)

    # Filter out objects with special positions (they're placed separately)
    layout_objects = [o for o in spec.objects if not o.position]
    special_objects = [o for o in spec.objects if o.position]

    # Place special-position objects and compute how much space they consume
    # so the main layout can avoid overlapping them.
    from dsa_anim.layout.strategies._sizing import estimate_object_size
    positions = {}
    top_used = 0
    bottom_used = 0
    for obj in special_objects:
        pos = _resolve_special_position(obj, canvas, theme)
        positions[obj.id or f"special_{id(obj)}"] = pos
        size = estimate_object_size(obj, theme)
        if obj.position == "top":
            top_used = max(top_used, pos.y - canvas.y + size.height + theme.resolve_gap("medium"))
        elif obj.position == "bottom":
            bottom_used = max(bottom_used, canvas.bottom - pos.y + theme.resolve_gap("medium"))

    # Shrink the layout canvas so centered content doesn't overlap pinned objects
    layout_canvas = Rect(
        canvas.x,
        canvas.y + top_used,
        canvas.width,
        canvas.height - top_used - bottom_used,
    )

    # Compute layout positions within the adjusted bounds
    layout_positions = layout_engine.compute(layout_objects, layout, layout_canvas)
    positions.update(layout_positions)

    # Recursively compute positions for group children
    for obj in spec.objects:
        _compute_group_children(obj, positions, layout_engine, theme)

    # Build scene nodes
    nodes = []
    node_map = {}
    for obj in spec.objects:
        node = _build_node(obj, positions, theme)
        nodes.append(node)
        node_map[node.id] = node
        # Also register children
        _register_children(node, node_map)

    # Resolve timeline
    duration = parse_duration(spec.duration) if spec.duration != "auto" else _auto_duration(spec)
    timeline = _resolve_animations(spec.animations, duration)

    # Camera
    camera_initial = CameraState()
    if spec.camera and spec.camera.initial:
        camera_initial = CameraState(zoom=spec.camera.initial.zoom)

    camera_keyframes = [
        CameraKeyframe(
            action=ca.action.value,
            start_time=parse_duration(ca.at),
            duration=parse_duration(ca.duration),
            easing=ca.easing.value,
            to_zoom=ca.to,
            focus_id=ca.focus,
        )
        for ca in spec.camera_animations
    ]

    # Transition
    transition = None
    if spec.transition:
        transition = TransitionInfo(
            type=spec.transition.type.value,
            duration=parse_duration(spec.transition.duration),
            target_id=spec.transition.target,
            direction=spec.transition.direction,
        )

    return ResolvedScene(
        id=spec.id or "scene",
        duration=duration,
        nodes=nodes,
        node_map=node_map,
        timeline=timeline,
        camera_initial=camera_initial,
        camera_keyframes=camera_keyframes,
        transition=transition,
        narration=spec.narration,
    )


def _compute_group_children(
    obj: ObjectSpec,
    positions: dict[str, Rect],
    layout_engine: LayoutEngine,
    theme: ThemeSpec,
) -> None:
    """Recursively lay out children of group objects using the group's computed rect as bounds."""
    if obj.type.value != "group" or not obj.children:
        return
    obj_id = obj.id or f"obj_{id(obj)}"
    group_rect = positions.get(obj_id)
    if group_rect is None:
        return
    child_layout = obj.layout if isinstance(obj.layout, LayoutSpec) else LayoutSpec(type=LayoutType.FLOW)
    child_positions = layout_engine.compute(obj.children, child_layout, group_rect)
    positions.update(child_positions)
    # Recurse into nested groups
    for child in obj.children:
        _compute_group_children(child, positions, layout_engine, theme)


def _build_node(obj: ObjectSpec, positions: dict[str, Rect], theme: ThemeSpec) -> SceneNode:
    obj_id = obj.id or f"obj_{id(obj)}"
    rect = positions.get(obj_id, Rect(0, 0, 100, 50))

    children = []
    if obj.children:
        for child in obj.children:
            child_node = _build_node(child, positions, theme)
            children.append(child_node)

    node = SceneNode(
        id=obj_id,
        obj_type=obj.type,
        rect=rect,
        content=obj.content,
        style=obj.style,
        style_props=theme.resolve_style(obj.style),
        children=children,
        position=obj.position,
        label=obj.label,
        from_id=obj.from_id,
        to_id=obj.to,
        token_id=obj.token_id,
        matrix_rows=obj.rows,
        matrix_cols=obj.cols,
        matrix_labels=obj.labels.model_dump() if obj.labels else None,
        matrix_data=obj.data,
        tokens=obj.tokens,
        highlight_pairs=[p.model_dump(by_alias=True) for p in obj.highlight_pairs] if obj.highlight_pairs else None,
        prob_items=[p.model_dump() for p in obj.items] if obj.items else None,
    )
    return node


def _register_children(node: SceneNode, node_map: dict[str, SceneNode]) -> None:
    for child in node.children:
        node_map[child.id] = child
        _register_children(child, node_map)


def _resolve_animations(anims: list[AnimSpec], scene_duration: float) -> list[AnimationKeyframe]:
    keyframes = []
    for anim in anims:
        start = parse_duration(anim.at) if anim.at else 0.0
        duration = parse_duration(anim.duration)
        stagger = parse_duration(anim.stagger) if anim.stagger else 0.0

        targets = anim.target if isinstance(anim.target, list) else [anim.target] if anim.target else []

        for i, target_id in enumerate(targets):
            kf = AnimationKeyframe(
                target_id=target_id,
                action=anim.action,
                start_time=start + i * stagger,
                duration=duration,
                easing=anim.easing.value,
                source_id=anim.source,
                targets=targets if len(targets) > 1 else None,
                to_value=anim.to,
                style=anim.style,
                color=anim.color,
                content=anim.content,
                direction=anim.direction,
                stagger=stagger,
                phases=[p.model_dump() for p in anim.phases] if anim.phases else None,
                offset_x=anim.offset_x,
                offset_y=anim.offset_y,
            )
            keyframes.append(kf)

    keyframes.sort(key=lambda k: k.start_time)
    return keyframes


def _resolve_special_position(obj: ObjectSpec, canvas: Rect, theme: ThemeSpec) -> Rect:
    from dsa_anim.layout.strategies._sizing import estimate_object_size
    size = estimate_object_size(obj, theme)

    match obj.position:
        case "top":
            return Rect(canvas.x + (canvas.width - size.width) / 2, canvas.y, size.width, size.height)
        case "above-layout":
            return Rect(canvas.x + (canvas.width - size.width) / 2, canvas.y - size.height - 10, size.width, size.height)
        case "bottom":
            return Rect(canvas.x + (canvas.width - size.width) / 2, canvas.bottom - size.height, size.width, size.height)
        case _:
            return Rect(canvas.x, canvas.y, size.width, size.height)


def _auto_duration(spec: SceneSpec) -> float:
    """Estimate duration from animations."""
    max_end = 5.0  # default
    for anim in spec.animations:
        start = parse_duration(anim.at) if anim.at else 0.0
        dur = parse_duration(anim.duration)
        stagger = parse_duration(anim.stagger) if anim.stagger else 0.0
        n_targets = len(anim.target) if isinstance(anim.target, list) else 1
        end = start + dur + stagger * (n_targets - 1)
        max_end = max(max_end, end + 1.0)  # 1s buffer
    return max_end
