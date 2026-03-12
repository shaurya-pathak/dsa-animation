"""Timeline utilities for animation state computation."""

from __future__ import annotations

from dsa_anim.scene_graph.models import SceneNode, AnimationKeyframe, CameraState, CameraKeyframe
from dsa_anim.dsl.schema import AnimAction
from dsa_anim.utils.easing import get_easing


def apply_animations_at_time(
    nodes: dict[str, SceneNode],
    keyframes: list[AnimationKeyframe],
    t: float,
) -> None:
    """Mutate scene nodes to reflect animation state at time t."""
    # Reset all nodes to default hidden state
    for node in nodes.values():
        node.visible = False
        node.opacity = 0.0
        node.scale_x = 1.0
        node.scale_y = 1.0
        node.translate_x = 0.0
        node.translate_y = 0.0
        node.draw_progress = 0.0
        node.highlight_intensity = 0.0
        node.highlight_color = None

    for kf in keyframes:
        node = nodes.get(kf.target_id)
        if node is None:
            continue

        progress = _get_progress(kf, t)

        match kf.action:
            case AnimAction.APPEAR:
                if t >= kf.start_time:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0

            case AnimAction.DISAPPEAR:
                if t < kf.start_time:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                else:
                    node.visible = False
                    node.opacity = 0.0

            case AnimAction.FADE_IN:
                if progress is not None:
                    node.visible = True
                    node.opacity = max(node.opacity, progress)
                    node.draw_progress = 1.0
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0

            case AnimAction.FADE_OUT:
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0 - progress
                    node.draw_progress = 1.0
                elif t >= kf.start_time + kf.duration:
                    node.visible = False
                    node.opacity = 0.0

            case AnimAction.TYPE | AnimAction.DRAW:
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = progress
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0

            case AnimAction.SCALE:
                to_val = kf.to_value or 1.0
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                    s = 1.0 + (to_val - 1.0) * progress
                    node.scale_x = s
                    node.scale_y = s
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                    node.scale_x = to_val
                    node.scale_y = to_val

            case AnimAction.MOVE:
                dx = kf.offset_x or 0.0
                dy = kf.offset_y or 0.0
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                    node.translate_x = dx * progress
                    node.translate_y = dy * progress
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                    node.translate_x = dx
                    node.translate_y = dy

            case AnimAction.HIGHLIGHT | AnimAction.GLOW | AnimAction.PULSE:
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                    # Envelope: fade-in (0–25%), hold (25–75%), fade-out (75–100%)
                    if progress < 0.25:
                        intensity = progress / 0.25
                    elif progress > 0.75:
                        intensity = (1.0 - progress) / 0.25
                    else:
                        intensity = 1.0
                    node.highlight_intensity = intensity
                    node.highlight_color = kf.color
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                    # intensity stays at reset value (0) — glow has faded out

            case AnimAction.ENCLOSE:
                # Box/token border draws around existing content
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = progress  # controls border draw progress
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0

            case AnimAction.SPLIT_FROM | AnimAction.FLOW_INTO | AnimAction.REVEAL_CELLS | AnimAction.GROW_BARS:
                # These are treated similarly to fade-in for basic rendering
                if progress is not None:
                    node.visible = True
                    node.opacity = progress
                    node.draw_progress = progress
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0

            case AnimAction.ANNOTATE:
                if progress is not None:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0
                elif t >= kf.start_time + kf.duration:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0

            case AnimAction.BUILD:
                # Multi-phase: treat overall as progressive reveal
                if kf.phases:
                    for phase in kf.phases:
                        phase_start = float(phase["at"].rstrip("s"))
                        phase_dur = float(phase["duration"].rstrip("s"))
                        if t >= phase_start:
                            node.visible = True
                            node.opacity = 1.0
                            phase_progress = min(1.0, (t - phase_start) / phase_dur) if phase_dur > 0 else 1.0
                            node.draw_progress = phase_progress

            case _:
                # Default: make visible if animation has started
                if t >= kf.start_time:
                    node.visible = True
                    node.opacity = 1.0
                    node.draw_progress = 1.0


def compute_camera_at_time(
    initial: CameraState,
    keyframes: list[CameraKeyframe],
    t: float,
    canvas_width: int,
    canvas_height: int,
) -> CameraState:
    """Compute camera state at time t."""
    state = CameraState(
        zoom=initial.zoom,
        center_x=canvas_width / 2,
        center_y=canvas_height / 2,
    )

    for kf in keyframes:
        if t < kf.start_time:
            continue

        raw_progress = (t - kf.start_time) / kf.duration if kf.duration > 0 else 1.0
        raw_progress = min(1.0, max(0.0, raw_progress))
        easing_fn = get_easing(kf.easing)
        progress = easing_fn(raw_progress)

        if kf.action == "zoom" and kf.to_zoom is not None:
            prev_zoom = initial.zoom
            # Find the zoom value just before this keyframe
            for prev_kf in keyframes:
                if prev_kf is kf:
                    break
                if prev_kf.action == "zoom" and prev_kf.to_zoom is not None and t >= prev_kf.start_time + prev_kf.duration:
                    prev_zoom = prev_kf.to_zoom
            state.zoom = prev_zoom + (kf.to_zoom - prev_zoom) * progress

    return state


def _get_progress(kf: AnimationKeyframe, t: float) -> float | None:
    """Get eased progress for a keyframe at time t. Returns None if not active."""
    if t < kf.start_time:
        return None
    if kf.duration <= 0:
        return 1.0 if t >= kf.start_time else None

    raw = (t - kf.start_time) / kf.duration
    if raw > 1.0:
        return None  # animation complete, handled separately

    easing_fn = get_easing(kf.easing)
    return easing_fn(max(0.0, min(1.0, raw)))
