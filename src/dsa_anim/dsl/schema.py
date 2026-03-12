"""Pydantic v2 models defining the entire DSL schema.

This is THE core file — it defines what LLMs can generate.
JSON Schema is auto-exported via DocumentSpec.model_json_schema().
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ObjectType(str, Enum):
    # Core
    TEXT = "text"
    BOX = "box"
    CIRCLE = "circle"
    ICON = "icon"
    IMAGE = "image"
    GROUP = "group"
    CONNECTOR = "connector"
    LIST = "list"
    CODE = "code"
    # Domain-specific
    TOKEN = "token"
    MATRIX = "matrix"
    ATTENTION_MAP = "attention-map"
    PROBABILITY_BAR = "probability-bar"
    CHART = "chart"
    ARRAY = "array"
    TREE = "tree"
    GRAPH = "graph"
    CALLOUT = "callout"


class LayoutType(str, Enum):
    CENTER = "center"
    GRID = "grid"
    FLOW = "flow"
    STACK = "stack"
    SPLIT = "split"
    ABSOLUTE = "absolute"


class AnimAction(str, Enum):
    # Basic
    FADE_IN = "fade-in"
    FADE_OUT = "fade-out"
    APPEAR = "appear"
    DISAPPEAR = "disappear"
    SCALE = "scale"
    MOVE = "move"
    # Drawing
    DRAW = "draw"
    TYPE = "type"
    WIPE = "wipe"
    # Data flow
    SPLIT_FROM = "split-from"
    FLOW_INTO = "flow-into"
    REVEAL_CELLS = "reveal-cells"
    GROW_BARS = "grow-bars"
    # Emphasis
    HIGHLIGHT = "highlight"
    PULSE = "pulse"
    GLOW = "glow"
    ANNOTATE = "annotate"
    # Complex
    BUILD = "build"
    MORPH = "morph"
    ENCLOSE = "enclose"


class EasingType(str, Enum):
    LINEAR = "linear"
    EASE_IN = "ease-in"
    EASE_OUT = "ease-out"
    EASE_IN_OUT = "ease-in-out"
    SPRING = "spring"
    BOUNCE = "bounce"


class TransitionType(str, Enum):
    FADE = "fade"
    WIPE = "wipe"
    ZOOM_INTO = "zoom-into"
    SLIDE = "slide"
    DISSOLVE = "dissolve"


class CameraAction(str, Enum):
    ZOOM = "zoom"
    PAN = "pan"
    FOCUS = "focus"
    SHAKE = "shake"


class GapSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|ms)$")


def parse_duration(value: str) -> float:
    """Parse a duration string like '2s' or '500ms' into seconds."""
    if value == "auto":
        return -1.0  # sentinel for auto-duration
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise ValueError(f"Invalid duration: {value!r}. Use e.g. '2s' or '500ms'.")
    num, unit = float(m.group(1)), m.group(2)
    return num if unit == "s" else num / 1000.0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class LayoutSpec(BaseModel):
    """Full layout specification."""

    type: LayoutType = LayoutType.CENTER
    columns: int | None = None
    rows: int | None = None
    gap: GapSize | str = GapSize.MEDIUM
    direction: str = "horizontal"
    align: str = "center"
    ratio: str | None = None  # e.g. "1:1", "1:3" for split layouts

    model_config = {"extra": "allow"}


# Union type: layout can be a string shorthand or full spec
Layout = LayoutSpec | str


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


class AttentionPair(BaseModel):
    """A weighted connection in an attention map."""

    from_token: str = Field(alias="from")
    to: str
    weight: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class ProbabilityItem(BaseModel):
    """An item in a probability bar chart."""

    label: str
    value: float = Field(ge=0.0, le=1.0)


class MatrixLabels(BaseModel):
    """Labels for matrix rows/columns."""

    rows: list[str] | None = None
    cols: list[str] | None = None


class ObjectSpec(BaseModel):
    """Specification for any visual object in a scene."""

    type: ObjectType
    id: str | None = None
    content: str | None = None
    style: str | None = None
    position: str | None = None  # e.g. "above-layout", "top"
    label: str | None = None

    # Group children
    children: list[ObjectSpec] | None = None
    layout: Layout | None = None  # for groups

    # Connector
    from_id: str | None = Field(None, alias="from")
    to: str | None = None

    # Token
    token_id: int | None = None

    # Matrix
    rows: int | None = None
    cols: int | None = None
    labels: MatrixLabels | None = None
    data: str | list[list[float]] | None = None  # "random" or actual data

    # Attention map
    tokens: list[str] | None = None
    highlight_pairs: list[AttentionPair] | None = None

    # Probability bar
    items: list[ProbabilityItem] | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------


class BuildPhase(BaseModel):
    """A phase in a multi-step build animation."""

    step: str
    at: str
    duration: str = "1s"
    stagger: str | None = None


class AnimSpec(BaseModel):
    """Specification for an animation action."""

    action: AnimAction
    target: str | list[str] | None = None
    source: str | None = None  # for split-from, flow-into

    # Timing
    at: str | None = None
    after: str | None = None
    duration: str = "0.5s"
    stagger: str | None = None
    easing: EasingType = EasingType.EASE_IN_OUT

    # Action-specific
    to: float | None = None  # for scale
    style: str | None = None
    color: str | None = None
    content: str | None = None  # for annotate
    direction: str | None = None  # for reveal-cells, wipe
    phases: list[BuildPhase] | None = None  # for build
    offset_x: float | None = None  # for move (pixels right = positive)
    offset_y: float | None = None  # for move (pixels down = positive)

    model_config = {"extra": "allow"}

    @field_validator("at", "duration", "stagger", mode="before")
    @classmethod
    def validate_duration_format(cls, v: str | None) -> str | None:
        if v is not None:
            parse_duration(v)  # validates format
        return v


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class CameraInitial(BaseModel):
    """Initial camera state for a scene."""

    zoom: float = 1.0
    focus: str = "center"  # object id or "center"


class CameraSpec(BaseModel):
    """Camera configuration for a scene."""

    initial: CameraInitial | None = None


class CameraAnimSpec(BaseModel):
    """A camera animation (viewport-level, not object-level)."""

    action: CameraAction
    to: float | None = None  # for zoom
    focus: str | None = None  # for pan/focus — object id or "center"
    at: str
    duration: str = "1s"
    easing: EasingType = EasingType.EASE_IN_OUT


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class TransitionSpec(BaseModel):
    """Transition between scenes."""

    type: TransitionType
    target: str | None = None  # for zoom-into
    direction: str | None = None  # for wipe, slide
    duration: str = "0.5s"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class SceneSpec(BaseModel):
    """A single scene in the animation."""

    id: str | None = None
    duration: str = "auto"
    layout: Layout = "center"
    narration: str | None = None

    objects: list[ObjectSpec] = Field(default_factory=list)
    animations: list[AnimSpec] = Field(default_factory=list)

    camera: CameraSpec | None = None
    camera_animations: list[CameraAnimSpec] = Field(default_factory=list)
    transition: TransitionSpec | None = None

    @field_validator("layout", mode="before")
    @classmethod
    def parse_layout_shorthand(cls, v: Any) -> Any:
        if isinstance(v, str):
            return LayoutSpec(type=LayoutType(v))
        return v


# ---------------------------------------------------------------------------
# Document (top-level)
# ---------------------------------------------------------------------------


class MetaSpec(BaseModel):
    """Top-level metadata."""

    title: str = "Untitled Animation"
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    theme: str = "whiteboard"


class DocumentSpec(BaseModel):
    """The top-level document — this is what the LLM generates."""

    version: str = "1.0"
    meta: MetaSpec = Field(default_factory=MetaSpec)
    scenes: list[SceneSpec] = Field(default_factory=list)
