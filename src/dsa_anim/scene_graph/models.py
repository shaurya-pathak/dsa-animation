"""Internal representation — the resolved scene graph.

This is the rendering-backend-agnostic IR that sits between the DSL and renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dsa_anim.dsl.schema import ObjectType, AnimAction, EasingType
from dsa_anim.utils.geometry import Rect, Point


@dataclass
class SceneNode:
    """A resolved visual object with computed position."""

    id: str
    obj_type: ObjectType
    rect: Rect
    content: str | None = None
    style: str | None = None
    style_props: dict[str, Any] = field(default_factory=dict)
    children: list[SceneNode] = field(default_factory=list)
    position: str | None = None  # "above-layout", "top", etc.
    label: str | None = None

    # Connector-specific
    from_id: str | None = None
    to_id: str | None = None

    # Token-specific
    token_id: int | None = None

    # Matrix-specific
    matrix_rows: int | None = None
    matrix_cols: int | None = None
    matrix_labels: dict | None = None
    matrix_data: Any = None

    # Attention map
    tokens: list[str] | None = None
    highlight_pairs: list[dict] | None = None

    # Probability bar
    prob_items: list[dict] | None = None

    # Animation state (mutated during rendering)
    opacity: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    visible: bool = False  # starts hidden, animations reveal
    draw_progress: float = 1.0  # for draw/type animations (0-1)
    highlight_intensity: float = 0.0
    highlight_color: str | None = None


@dataclass
class AnimationKeyframe:
    """A resolved animation keyframe with absolute timing."""

    target_id: str
    action: AnimAction
    start_time: float  # absolute seconds within the scene
    duration: float
    easing: str = "ease-in-out"

    # Action-specific params
    source_id: str | None = None
    targets: list[str] | None = None  # for multi-target animations
    to_value: float | None = None  # for scale
    style: str | None = None
    color: str | None = None
    content: str | None = None
    direction: str | None = None
    stagger: float = 0.0
    phases: list[dict] | None = None
    offset_x: float | None = None  # for move (pixels)
    offset_y: float | None = None  # for move (pixels)


@dataclass
class CameraState:
    """Camera viewport state."""

    zoom: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass
class CameraKeyframe:
    """A camera animation keyframe."""

    action: str
    start_time: float
    duration: float
    easing: str = "ease-in-out"
    to_zoom: float | None = None
    focus_id: str | None = None


@dataclass
class TransitionInfo:
    """Scene transition info."""

    type: str
    duration: float
    target_id: str | None = None
    direction: str | None = None


@dataclass
class ResolvedScene:
    """A fully resolved scene ready for rendering."""

    id: str
    duration: float
    nodes: list[SceneNode]
    node_map: dict[str, SceneNode]  # id -> node for quick lookup
    timeline: list[AnimationKeyframe]
    camera_initial: CameraState = field(default_factory=CameraState)
    camera_keyframes: list[CameraKeyframe] = field(default_factory=list)
    transition: TransitionInfo | None = None
    narration: str | None = None


@dataclass
class SceneGraph:
    """The complete resolved animation."""

    width: int
    height: int
    fps: int
    theme_name: str
    scenes: list[ResolvedScene]

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.scenes)
