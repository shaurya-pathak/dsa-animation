"""Pydantic v2 models defining the entire DSL schema.

This is THE core file — it defines what LLMs can generate.
JSON Schema is auto-exported via DocumentSpec.model_json_schema().
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from kaivra.version import CURRENT_DSL_VERSION

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ObjectType(str, Enum):
    TEXT = "text"
    BOX = "box"
    CIRCLE = "circle"
    LINEAR_METER = "linear_meter"
    SIGMOID_PLOT = "sigmoid_plot"
    PET_PORTRAIT = "pet_portrait"
    # Short authoring alias for ``pet_portrait``. Both values intentionally
    # render through the same primitive so a document remains explicit when it
    # is serialized again.
    PET = "pet"
    GROUP = "group"
    CONNECTOR = "connector"
    TOKEN = "token"
    CALLOUT = "callout"
    SEMANTIC_ICON = "semantic_icon"


class LayoutType(str, Enum):
    CENTER = "center"
    GRID = "grid"
    FLOW = "flow"
    STACK = "stack"
    SPLIT = "split"
    CAROUSEL = "carousel"


class AnimAction(str, Enum):
    # Visibility
    APPEAR = "appear"
    DISAPPEAR = "disappear"
    FADE_IN = "fade-in"
    FADE_OUT = "fade-out"
    # Motion
    MOVE = "move"
    MOVE_TO = "move-to"
    SWAP = "swap"
    SCALE = "scale"
    METER_TO = "meter-to"
    # Drawing
    DRAW = "draw"
    FLOW = "flow"
    TYPE = "type"
    REVEAL = "reveal"
    REVEAL_CHILDREN = "reveal-children"
    # Emphasis
    HIGHLIGHT = "highlight"
    PULSE = "pulse"
    # Complex
    BUILD = "build"
    REPLACE = "replace"


class EasingType(str, Enum):
    LINEAR = "linear"
    EASE_IN = "ease-in"
    EASE_OUT = "ease-out"
    EASE_IN_OUT = "ease-in-out"
    SPRING = "spring"
    BOUNCE = "bounce"


class TransitionType(str, Enum):
    FADE = "fade"


class GapSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class PacingPreset(str, Enum):
    QUICK_DEMO = "quick-demo"
    BALANCED = "balanced"
    EDUCATIONAL = "educational"


class ContinuityMode(str, Enum):
    STRICT = "strict"
    EVOLVING = "evolving"
    POSITION_ONLY = "position_only"


class SizeVariant(str, Enum):
    COMPACT = "compact"
    DEFAULT = "default"
    HERO = "hero"


class AudienceLevel(str, Enum):
    LAYPERSON = "layperson"
    MIXED = "mixed"
    TECHNICAL = "technical"


class PetKind(str, Enum):
    """The familiar animal cues used by a native pet portrait."""

    MYSTERY = "mystery"
    DOG = "dog"
    CAT = "cat"


class PetFeature(str, Enum):
    """The explainable visual cues a pet portrait can call out."""

    EARS = "ears"
    SNOUT = "snout"


class SemanticIconName(str, Enum):
    """Small deterministic illustrations for story-first explainer diagrams."""

    FUND = "fund"
    STOREFRONT = "storefront"
    WAREHOUSE = "warehouse"
    CASH = "cash"
    SHARES = "shares"
    BORROW = "borrow"
    LOAN = "loan"
    HANDSHAKE = "handshake"


class RelativeBasis(str, Enum):
    SELF = "self"
    TARGET = "target"


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


def _validate_timing_value(value: str | None) -> str | None:
    """Accept duration literals and semantic timing expressions."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Timing values must be strings.")
    stripped = value.strip()
    if not stripped:
        raise ValueError("Timing values must not be empty.")
    if stripped == "auto" or _DURATION_RE.match(stripped):
        parse_duration(stripped)
    return stripped


class RelativePositionSpec(BaseModel):
    """Relative translation normalized to an object or target bounds."""

    x: float | None = Field(None, description="Horizontal translation multiplier")
    y: float | None = Field(None, description="Vertical translation multiplier")
    basis: RelativeBasis = Field(
        RelativeBasis.SELF,
        description="Bounds used to resolve the translation: self or target",
    )

    @model_validator(mode="after")
    def validate_axes(self) -> "RelativePositionSpec":
        if self.x is None and self.y is None:
            raise ValueError("Relative translations require at least one of `x` or `y`.")
        return self


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class GridRegionSpec(BaseModel):
    """A named grid region (Bootstrap-style column span)."""

    row: int = Field(1, description="1-based row index")
    row_span: int = Field(1, description="Number of rows to span")
    col: int = Field(1, description="1-based column index")
    span: int = Field(1, description="Number of columns to span")
    align: Literal["center", "top", "bottom", "left", "right"] = Field(
        "center",
        description="How objects placed in this region are aligned within its bounds.",
    )


class GridPositionSpec(BaseModel):
    """Explicit grid placement for an object."""

    row: int | None = Field(None, description="1-based row index")
    col: int | None = Field(None, description="1-based column index")
    span: int | None = Field(None, description="Number of columns to span")
    row_span: int | None = Field(None, description="Number of rows to span")
    region: str | None = Field(
        None,
        description=(
            "Named region defined in layout.regions (e.g. 'main', 'sidebar'). "
            "The 'one-column' template also supports semantic regions like "
            "'problem_solution', 'request_pipeline', 'fan_out', "
            "'system_architecture', and 'timeline_steps'."
        ),
    )


class LayoutSpec(BaseModel):
    """Full layout specification for arranging child objects."""

    type: LayoutType = Field(
        LayoutType.CENTER,
        description="Layout algorithm: center, grid, flow, stack, split, carousel",
    )
    columns: int | None = Field(None, description="Number of grid columns (for grid layout)")
    rows: int | None = Field(None, description="Number of grid rows (for grid layout)")
    gap: GapSize | str = Field(
        GapSize.MEDIUM, description="Spacing between objects: 'small', 'medium', or 'large'"
    )
    direction: Literal["horizontal", "vertical"] = Field(
        "horizontal", description="Flow direction for flow/stack layouts"
    )
    align: Literal["center", "top", "bottom", "left", "right"] = Field(
        "center", description="Alignment of objects within the layout"
    )
    ratio: str | None = Field(None, description="Size ratio for split layouts, e.g. '1:1', '1:3'")
    regions: dict[str, GridRegionSpec] | None = Field(
        None, description="Named grid regions for Bootstrap-style placement"
    )
    # Carousel-specific options
    curve: float | None = Field(
        None, description="Carousel arc height in pixels (positive = upward arc)"
    )
    active: str | None = Field(None, description="Active item ID for carousel emphasis")
    active_scale: float | None = Field(None, description="Scale for active carousel item")
    inactive_scale: float | None = Field(None, description="Scale for inactive carousel items")

    model_config = {"extra": "allow"}


# Union type: layout can be a string shorthand or full spec
Layout = LayoutSpec | str


# ---------------------------------------------------------------------------
# Motion presets
# ---------------------------------------------------------------------------


class MotionSpec(BaseModel):
    """High-level motion preset for enter/exit/idle."""

    preset: str = Field(
        ..., description="Motion preset name (e.g. 'fade', 'pop', 'slide-up', 'breathe')"
    )
    at: str | None = Field(None, description="Start time for the motion (optional)")
    duration: str = Field("0.6s", description="Motion duration")
    easing: EasingType = Field(EasingType.EASE_OUT, description="Easing function")

    # Optional overrides
    translate: RelativePositionSpec | None = Field(
        None,
        description=(
            "Relative target translation as normalized deltas. "
            "`basis='self'` scales by the animated object's size, "
            "`basis='target'` scales by the destination object's size."
        ),
    )
    from_translate: RelativePositionSpec | None = Field(
        None,
        description="Starting relative translation for move presets (animates to `translate`).",
    )
    scale: float | None = Field(None, description="Target scale for pop/scale presets")
    from_scale: float | None = Field(None, description="Starting scale for pop/scale presets")
    intensity: float | None = Field(None, description="Idle intensity (pixels or scale delta)")
    speed: float | None = Field(None, description="Idle speed")
    axis: Literal["x", "y", "both"] | None = Field("both", description="Idle motion axis")
    # Absolute pixel offsets — kept for backwards compatibility.
    # Prefer `translate`/`from_translate` for new animations.
    offset_x: float | None = Field(
        None, description="Absolute horizontal pixel offset (legacy; prefer translate)"
    )
    offset_y: float | None = Field(
        None, description="Absolute vertical pixel offset (legacy; prefer translate)"
    )
    from_offset_x: float | None = Field(
        None,
        description="Absolute starting horizontal pixel offset (legacy; prefer from_translate)",
    )
    from_offset_y: float | None = Field(
        None, description="Absolute starting vertical pixel offset (legacy; prefer from_translate)"
    )

    @field_validator("at", "duration", mode="before")
    @classmethod
    def validate_motion_durations(cls, v: str | None) -> str | None:
        return _validate_timing_value(v)


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


class ObjectSpec(BaseModel):
    """Specification for any visual object in a scene."""

    type: ObjectType = Field(
        description=(
            "Object type: text, box, circle, linear_meter, sigmoid_plot, pet_portrait "
            "(or pet), semantic_icon, group, connector, token, or callout"
        )
    )
    id: str | None = Field(
        None, description="Unique identifier for this object (auto-generated if omitted)"
    )
    content: str | None = Field(None, description="Text content displayed inside the object")
    spoken_forms: list[str] | None = Field(
        None,
        description=(
            "Optional narration/pronunciation aliases used for voice-sync matching, "
            "for example ['co pilot', 'cobalt'] for on-screen text 'Copilot'."
        ),
    )
    style: str | None = Field(
        None,
        description=(
            "Visual style preset. Typography roles include 'hero-heading', 'heading', "
            "'section-heading', 'body', 'metric', 'operator', and 'annotation'; semantic color "
            "variants can be applied to metrics and connectors."
        ),
    )
    position: Literal["top", "bottom", "left", "right", "above-layout"] | None = Field(
        None, description="Pin object to a canvas edge instead of participating in layout"
    )
    grid: GridPositionSpec | None = Field(
        None, description="Explicit grid placement (row, col, span, or named region)"
    )
    label: str | None = Field(
        None, description="Small label displayed on the object (e.g. badge text)"
    )
    actor_id: str | None = Field(
        None,
        description=(
            "Stable actor identity used for continuity across scenes. "
            "When omitted, continuity falls back to the object's id."
        ),
    )
    continuity_mode: ContinuityMode = Field(
        ContinuityMode.STRICT,
        description=(
            "Continuity matching policy. `strict` preserves current behavior, "
            "`evolving` allows moderate content evolution, and "
            "`position_only` keeps motion continuity for abstract actors."
        ),
    )
    size_variant: SizeVariant = Field(
        SizeVariant.DEFAULT,
        description="Visual size preset: compact, default, or hero.",
    )
    visible: bool | None = Field(
        None, description="Default visibility for this object (overrides scene auto_visible)"
    )
    scale_text: bool | None = Field(
        None,
        description="Whether content text should scale with the object transform. Defaults to false for boxes/tokens and true otherwise.",
    )
    align_equals: bool = Field(
        False,
        description=(
            "Whether equation-like text should align its equals sign with sibling objects. "
            "Defaults to false so alignment is opt-in."
        ),
    )
    # Motion presets
    enter: "MotionSpec | None" = Field(None, description="Enter animation preset for this object")
    exit: "MotionSpec | None" = Field(None, description="Exit animation preset for this object")
    idle: "MotionSpec | None" = Field(None, description="Idle motion preset for this object")

    # Group children
    children: list[ObjectSpec] | None = Field(
        None, description="Child objects (only for type='group')"
    )
    layout: Layout | None = Field(
        None, description="Layout for arranging children (only for type='group')"
    )

    # Connector
    from_id: str | None = Field(None, alias="from", description="Source object ID (for connectors)")
    to_id: str | None = Field(
        None, alias="to", description="Destination object ID (for connectors)"
    )
    target: str | None = Field(
        None, description="Target object ID that this callout points to (for callouts)"
    )

    # Token
    token_id: int | None = Field(
        None, description="Numeric token ID displayed as a badge (for tokens)"
    )

    # Pet portrait
    pet_kind: PetKind = Field(
        PetKind.MYSTERY,
        description=(
            "Friendly pet portrait variant for type='pet_portrait' or type='pet': "
            "'dog', 'cat', or 'mystery'. The mystery variant deliberately combines "
            "dog and cat cues for a dog-or-cat question."
        ),
    )
    pet_highlights: list[PetFeature] = Field(
        default_factory=list,
        description=(
            "Visible semantic cues to outline on a pet portrait. Supported values are 'ears' "
            "and 'snout'. Ears use the explanatory coral channel; snout uses cyan."
        ),
    )
    show_feature_labels: bool = Field(
        False,
        description=(
            "Whether a pet portrait should render its selected feature labels with short leader "
            "lines. The layout reserves annotation gutters when true."
        ),
    )

    # Semantic icon
    icon_name: SemanticIconName | None = Field(
        None,
        description=(
            "Named flat illustration for type='semantic_icon'. Supported values are "
            "'fund', 'storefront', 'warehouse', 'cash', 'shares', 'borrow', 'loan', "
            "and 'handshake'. Use content only as a short human-readable caption."
        ),
    )

    # Linear meter
    meter_value: float = Field(
        0.0,
        description=(
            "Current numeric value for type='linear_meter'. The rendered fill and pointer are "
            "bounded to meter_min through meter_max."
        ),
    )
    meter_min: float = Field(
        0.0,
        description="Inclusive low end for type='linear_meter'. Must be smaller than meter_max.",
    )
    meter_max: float = Field(
        100.0,
        description="Inclusive high end for type='linear_meter'. Must be larger than meter_min.",
    )
    meter_left_label: str | None = Field(
        None,
        description="Optional label anchored to the low end of a linear meter.",
    )
    meter_center_label: str | None = Field(
        None,
        description="Optional label anchored to the midpoint of a linear meter.",
    )
    meter_right_label: str | None = Field(
        None,
        description="Optional label anchored to the high end of a linear meter.",
    )
    meter_value_label: str | None = Field(
        None,
        description="Optional readable label attached to a linear meter's current pointer.",
    )
    meter_caption: str | None = Field(
        None,
        description="Optional concise explanatory caption above a linear meter.",
    )

    # Sigmoid plot
    sigmoid_input: float = Field(
        0.0,
        description=(
            "Input score highlighted on type='sigmoid_plot'. The renderer derives the "
            "corresponding probability with the logistic sigmoid."
        ),
    )
    sigmoid_input_label: str | None = Field(
        None,
        description="Optional readable label below the highlighted sigmoid input.",
    )
    sigmoid_output_label: str | None = Field(
        None,
        description="Optional readable probability label beside the highlighted sigmoid output.",
    )
    sigmoid_caption: str | None = Field(
        "SIGMOID",
        description="Optional concise title above a sigmoid plot.",
    )

    # Callout
    callout_side: Literal["left", "right", "top", "bottom"] | None = Field(
        None, description="Which side of the target to place the callout"
    )

    model_config = {"populate_by_name": True, "extra": "allow"}

    @model_validator(mode="after")
    def validate_linear_meter_bounds(self) -> "ObjectSpec":
        """Keep every meter's scale intelligible before it reaches a renderer."""
        if self.type == ObjectType.LINEAR_METER and self.meter_max <= self.meter_min:
            raise ValueError("linear_meter requires meter_max to be greater than meter_min")
        if self.type not in {ObjectType.PET_PORTRAIT, ObjectType.PET} and (
            self.pet_highlights or self.show_feature_labels
        ):
            raise ValueError(
                "pet_highlights and show_feature_labels require type='pet_portrait' or 'pet'"
            )
        if self.type == ObjectType.SEMANTIC_ICON and self.icon_name is None:
            raise ValueError("semantic_icon requires icon_name")
        if self.type != ObjectType.SEMANTIC_ICON and self.icon_name is not None:
            raise ValueError("icon_name requires type='semantic_icon'")
        return self


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------


class BuildPhase(BaseModel):
    """A phase in a multi-step build animation."""

    step: str = Field(description="Description of this build phase")
    at: str = Field(description="Start time for this phase, e.g. '2s'")
    duration: str = Field("1s", description="Duration of this phase")
    stagger: str | None = Field(None, description="Delay between targets in this phase")

    @field_validator("at", "duration", "stagger", mode="before")
    @classmethod
    def validate_phase_durations(cls, v: str | None) -> str | None:
        return _validate_timing_value(v)


class AnimSpec(BaseModel):
    """Specification for an animation action."""

    id: str | None = Field(
        None,
        description="Optional animation identifier used by semantic timing anchors.",
    )
    action: AnimAction = Field(
        description="Animation type: appear, disappear, fade-in, fade-out, move, move-to, swap, scale, meter-to, draw, flow, type, reveal, reveal-children, highlight, pulse, build, replace"
    )
    target: str | list[str] | None = Field(
        None,
        validation_alias=AliasChoices("target", "targets"),
        serialization_alias="target",
        description="Object ID(s) to animate",
    )
    to_id: str | None = Field(None, description="Destination object ID (for move-to)")
    with_id: str | None = Field(
        None,
        validation_alias=AliasChoices("with", "with_id"),
        serialization_alias="with",
        description="Replacement object ID for replace animations",
    )

    # Timing
    at: str | None = Field(None, description="Start time, e.g. '0.5s' or '200ms'")
    anchor: str | None = Field(
        None,
        description="Anchor to another animation ID, object ID, or scene boundary like 'scene_start'/'scene_end'.",
    )
    after: str | None = Field(None, description="Start after another animation completes")
    duration: str = Field("0.5s", description="Animation duration, e.g. '1s' or '500ms'")
    gap: str | None = Field(
        None,
        description="Relative offset token or duration applied after an anchor/cue, e.g. 'short'.",
    )
    step: str | None = Field(
        None,
        description="Per-target reveal delay for high-level reveal actions, e.g. 'short'.",
    )
    stagger: str | None = Field(
        None, description="Delay between targets when animating multiple objects"
    )
    order: Literal["sequential"] | None = Field(
        None,
        description="Ordering mode for high-level reveal actions.",
    )
    cue: str | None = Field(
        None,
        description="Narration cue phrase to anchor against when external cue timings are supplied.",
    )
    easing: EasingType = Field(
        EasingType.EASE_IN_OUT,
        description="Easing function: linear, ease-in, ease-out, ease-in-out, spring, bounce",
    )

    # Action-specific
    scale_factor: float | None = Field(
        None, description="Target scale multiplier (for scale action, e.g. 1.5 = 150%)"
    )
    from_scale: float | None = Field(
        None, description="Starting scale for scale action (defaults to 1.0)"
    )
    meter_value: float | None = Field(
        None,
        description=(
            "Target numeric value for a meter-to action. The value changes continuously from "
            "the meter's current value and is visually bounded to its declared meter range."
        ),
    )
    style: Literal["glow", "outline", "fade-in", "appear"] | None = Field(
        None,
        description="Visual style, such as 'glow'/'outline' for emphasis or 'fade-in'/'appear' for reveal actions.",
    )
    color: str | None = Field(
        None, description="Color name for emphasis animations (e.g. 'accent', 'success', 'error')"
    )
    phases: list[BuildPhase] | None = Field(None, description="Build phases (for build action)")
    translate: RelativePositionSpec | None = Field(
        None,
        description=(
            "Relative translation for move or move-to as normalized deltas. "
            "Use `basis='self'` to scale by the animated object, or `basis='target'` for move-to target bounds."
        ),
    )
    from_translate: RelativePositionSpec | None = Field(
        None,
        description="Starting relative translation for move actions (animates to `translate`).",
    )
    # Absolute pixel offsets — kept for backwards compatibility.
    # Prefer `translate`/`from_translate` for new animations.
    offset_x: float | None = Field(
        None, description="Absolute horizontal pixel offset (legacy; prefer translate)"
    )
    offset_y: float | None = Field(
        None, description="Absolute vertical pixel offset (legacy; prefer translate)"
    )
    from_offset_x: float | None = Field(
        None,
        description="Absolute starting horizontal pixel offset (legacy; prefer from_translate)",
    )
    from_offset_y: float | None = Field(
        None, description="Absolute starting vertical pixel offset (legacy; prefer from_translate)"
    )

    model_config = {"extra": "allow"}

    @field_validator("at", "duration", "gap", "step", "stagger", mode="before")
    @classmethod
    def validate_duration_format(cls, v: str | None) -> str | None:
        return _validate_timing_value(v)

    @model_validator(mode="after")
    def validate_replace_shape(self) -> "AnimSpec":
        if self.action != AnimAction.REPLACE:
            return self._validate_high_level_shape()
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("Replace animations require a single string target ID.")
        if not self.with_id:
            raise ValueError("Replace animations require a `with` object ID.")
        return self._validate_high_level_shape()

    def _validate_high_level_shape(self) -> "AnimSpec":
        selectors = [
            name
            for name, value in (
                ("at", self.at),
                ("anchor", self.anchor),
                ("after", self.after),
                ("cue", self.cue),
            )
            if value
        ]
        if len(selectors) > 1 and set(selectors) != {"at", "cue"}:
            raise ValueError(
                f"Animation {self.id or self.action.value!r} uses multiple timing anchors "
                f"{selectors}; only `at` + `cue` may be combined so `at` can provide a "
                "silent-render fallback."
            )
        if self.meter_value is not None and self.action != AnimAction.METER_TO:
            raise ValueError("`meter_value` is only supported for meter-to animations.")
        if self.action == AnimAction.METER_TO:
            if not isinstance(self.target, str) or not self.target.strip():
                raise ValueError("Meter-to animations require one linear_meter target ID.")
            if self.meter_value is None:
                raise ValueError("Meter-to animations require a numeric `meter_value` target.")
        elif self.action == AnimAction.REVEAL:
            if self.target is None:
                raise ValueError("Reveal animations require `target` or `targets`.")
            if self.style is not None and self.style not in {"fade-in", "appear"}:
                raise ValueError("Reveal animations only support style `fade-in` or `appear`.")
        elif self.action == AnimAction.REVEAL_CHILDREN:
            if not isinstance(self.target, str) or not self.target.strip():
                raise ValueError("Reveal-children animations require one target group ID.")
            if self.style is not None and self.style not in {"fade-in", "appear"}:
                raise ValueError(
                    "Reveal-children animations only support style `fade-in` or `appear`."
                )
        elif self.style is not None and self.action in {AnimAction.HIGHLIGHT, AnimAction.PULSE}:
            if self.style not in {"glow", "outline"}:
                raise ValueError(
                    "Highlight and pulse animations only support style `glow` or `outline`."
                )
        elif self.style is not None:
            raise ValueError(
                "`style` is only supported for reveal, reveal-children, highlight, and pulse animations."
            )
        if self.order is not None and self.action not in {
            AnimAction.REVEAL,
            AnimAction.REVEAL_CHILDREN,
        }:
            raise ValueError("`order` is only supported for reveal and reveal-children animations.")
        if self.step is not None and self.action not in {
            AnimAction.REVEAL,
            AnimAction.REVEAL_CHILDREN,
        }:
            raise ValueError("`step` is only supported for reveal and reveal-children animations.")
        return self


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class TransitionSpec(BaseModel):
    """Transition between scenes."""

    type: TransitionType = Field(description="Transition type: fade")
    duration: str = Field("0.5s", description="Transition duration")


# ---------------------------------------------------------------------------
# Focus helpers
# ---------------------------------------------------------------------------


class FocusStyleSpec(BaseModel):
    """Auto-focus styling for a scene (scale + highlight)."""

    at: str = Field("0s", description="Start time for focus animation")
    duration: str = Field("1.2s", description="Duration for focus animation")
    scale: float = Field(1.15, description="Scale applied to focused targets")
    color: str = Field("accent", description="Highlight color")
    style: Literal["glow", "outline"] = Field("glow", description="Highlight style")

    @field_validator("at", "duration", mode="before")
    @classmethod
    def validate_focus_durations(cls, v: str | None) -> str | None:
        return _validate_timing_value(v)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class SceneSpec(BaseModel):
    """A single scene in the animation."""

    id: str | None = Field(None, description="Unique scene identifier (auto-generated if omitted)")
    duration: str = Field(
        "auto", description="Scene duration, e.g. '5s'. Use 'auto' to infer from animations"
    )
    layout: Layout = "center"
    template: str | None = Field(
        None,
        description=(
            "Optional legacy/document-layout template: 'editorial', 'two-column', "
            "'one-column', or 'storyboard'. Do not use a template as the starting point "
            "for a narrated explainer; derive explicit scene and group layouts from the "
            "subject's causal choreography."
        ),
    )
    narration: str | None = Field(
        None, description="Narration text displayed at the bottom of the scene"
    )

    objects: list[ObjectSpec] = Field(
        default_factory=list, description="Visual objects in this scene"
    )
    animations: list[AnimSpec] = Field(
        default_factory=list, description="Animations to play during this scene"
    )
    auto_visible: bool = Field(
        False, description="If true, objects are visible by default without appear animations"
    )
    focus: str | list[str] | None = Field(None, description="Auto-focus target(s) for this scene")
    focus_style: FocusStyleSpec | None = Field(None, description="Focus styling options")
    continuity: bool | None = Field(
        None, description="If true, inherit positions from previous scene for shared IDs"
    )
    include_persistent_objects: bool = Field(
        True, description="Whether document-level persistent objects should appear in this scene"
    )
    show_progress_bar: bool = Field(
        False,
        description=(
            "Whether to render bottom navigation progress. Disabled by default because "
            "presentation chrome should not occupy narrated explainer frames."
        ),
    )
    transition: TransitionSpec | None = Field(None, description="Transition to next scene")

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

    title: str = Field("Untitled Animation", description="Animation title")
    resolution: tuple[int, int] = Field(
        (1920, 1080), description="Canvas resolution [width, height]"
    )
    fps: int = Field(30, description="Frames per second")
    theme: str = Field("editorial", description="Visual theme name")
    audience: AudienceLevel | None = Field(
        None,
        description="Target audience level: layperson, mixed, or technical.",
    )
    story_contract: str | None = Field(
        None,
        description=(
            "Relative path to the reviewed Markdown story contract for this animation, "
            "typically '<slug>.story.md' beside the JSON file. It guides authoring and "
            "does not affect rendering."
        ),
    )
    show_subtitles: bool = Field(
        False,
        description=(
            "Whether to render scene narration as on-screen subtitles. Disabled by default "
            "so narration does not become duplicated screen copy."
        ),
        validation_alias=AliasChoices("show_subtitles", "show_narration"),
        serialization_alias="show_subtitles",
    )
    pacing: PacingPreset = Field(
        PacingPreset.BALANCED, description="Timing profile: quick-demo, balanced, or educational"
    )
    continuity: bool = Field(True, description="Inherit positions between scenes for shared IDs")
    continuity_duration: str = Field(
        "0.6s", description="Duration for continuity moves between scenes"
    )
    glow_release_padding: str = Field(
        "0.6s",
        description="Minimum tail time at scene end after highlight/pulse effects",
    )
    video_bookends: bool = Field(
        False,
        description="Whether rendered videos should include intro and outro bookend scenes.",
    )

    model_config = {"populate_by_name": True}

    @field_validator("continuity_duration", "glow_release_padding", mode="before")
    @classmethod
    def validate_meta_durations(cls, v: str | None) -> str | None:
        if v is not None:
            parse_duration(v)
        return v

    @property
    def show_narration(self) -> bool:
        """Backward-compatible alias for subtitle rendering."""
        return self.show_subtitles

    @show_narration.setter
    def show_narration(self, value: bool) -> None:
        self.show_subtitles = value

    def subtitles_were_explicitly_set(self) -> bool:
        """Whether subtitle visibility was explicitly authored in the input."""
        return "show_subtitles" in self.model_fields_set


class DocumentSpec(BaseModel):
    """The top-level document — this is what the LLM generates."""

    version: str = Field(CURRENT_DSL_VERSION, description="Schema version")
    meta: MetaSpec = Field(default_factory=MetaSpec, description="Animation metadata")
    objects: list[ObjectSpec] = Field(
        default_factory=list,
        description=(
            "Persistent story actors or values visible across scenes. Do not use this for "
            "automatic headings, legends, chapter rails, or navigation chrome."
        ),
    )
    scenes: list[SceneSpec] = Field(default_factory=list, description="Ordered list of scenes")

    # The exported schema describes the current v1.5 authoring contract. Legacy
    # top-level permissiveness remains a runtime compatibility path below.
    model_config = {"json_schema_extra": {"additionalProperties": False}}

    @model_validator(mode="before")
    @classmethod
    def apply_versioned_defaults_and_validate_top_level(cls, value: Any) -> Any:
        """Preserve legacy rendering defaults while making editorial the v1.5 contract.

        The schema's field defaults describe the current document format.  Older documents
        need their defaults materialized before ``MetaSpec`` validates, otherwise a missing
        field would silently adopt the new rendering behavior.
        """
        if not isinstance(value, dict):
            return value

        raw = dict(value)
        version = raw.get("version")
        legacy = _is_legacy_document_version(version)

        if not legacy:
            unknown_fields = sorted(set(raw) - _DOCUMENT_TOP_LEVEL_FIELDS)
            if unknown_fields:
                unknown = ", ".join(f"`{field}`" for field in unknown_fields)
                misplaced_metadata = [
                    field for field in unknown_fields if field in _MISPLACED_META_FIELDS
                ]
                hint = ""
                if misplaced_metadata:
                    fields = ", ".join(f"`{field}`" for field in misplaced_metadata)
                    hint = f" Move {fields} under `meta`."
                requested_version = version if version is not None else "1.5"
                raise ValueError(
                    f"DSL {requested_version!r} does not allow unknown top-level fields: "
                    f"{unknown}. Allowed fields are `version`, `meta`, `objects`, and `scenes`."
                    f"{hint}"
                )

        meta = raw.get("meta")
        if isinstance(meta, MetaSpec):
            # ``exclude_unset`` retains only fields an API caller explicitly selected, so
            # legacy compatibility defaults can still be applied below.
            meta = meta.model_dump(by_alias=True, exclude_unset=True)

        if meta is None and "meta" not in raw:
            meta = {}
        if isinstance(meta, dict):
            resolved_meta = dict(meta)
            if legacy:
                resolved_meta.setdefault("theme", "whiteboard")
                resolved_meta.setdefault("video_bookends", True)
                if "show_subtitles" not in resolved_meta and "show_narration" not in resolved_meta:
                    resolved_meta["show_subtitles"] = True
            raw["meta"] = resolved_meta

        if legacy and isinstance(raw.get("scenes"), list):
            raw["scenes"] = [
                {**scene, "show_progress_bar": scene.get("show_progress_bar", True)}
                if isinstance(scene, dict)
                else scene
                for scene in raw["scenes"]
            ]

        return raw

    @model_validator(mode="after")
    def validate_animation_targets(self) -> "DocumentSpec":
        """Ensure flow animations have a rendered connector to follow.

        This intentionally relies only on declarative scene data: a flow must either
        explicitly follow its matching draw animation or use literal timestamps that begin
        after that draw completes.  More elaborate semantic timing is resolved later by the
        scene graph and remains outside schema validation.
        """
        persistent_objects = _index_objects(self.objects)
        for scene in self.scenes:
            object_index = persistent_objects | _index_objects(scene.objects)
            draws_by_target: dict[str, list[AnimSpec]] = {}
            for animation in scene.animations:
                if animation.action != AnimAction.DRAW:
                    continue
                for target in _animation_target_ids(animation):
                    draws_by_target.setdefault(target, []).append(animation)

            for animation in scene.animations:
                if animation.action != AnimAction.FLOW:
                    continue

                targets = _animation_target_ids(animation)
                if not targets:
                    raise ValueError(
                        f"Flow animation {_animation_label(animation)!r} in scene "
                        f"{_scene_label(scene)!r} requires a connector target."
                    )

                for target in targets:
                    target_object = object_index.get(target)
                    if target_object is None:
                        raise ValueError(
                            f"Flow animation {_animation_label(animation)!r} in scene "
                            f"{_scene_label(scene)!r} targets unknown object {target!r}."
                        )
                    if target_object.type != ObjectType.CONNECTOR:
                        raise ValueError(
                            f"Flow animation {_animation_label(animation)!r} in scene "
                            f"{_scene_label(scene)!r} must target a connector; {target!r} is "
                            f"a {target_object.type.value!r} object."
                        )

                    matching_draws = draws_by_target.get(target, [])
                    if not matching_draws:
                        raise ValueError(
                            f"Flow animation {_animation_label(animation)!r} in scene "
                            f"{_scene_label(scene)!r} needs a preceding draw animation for "
                            f"connector {target!r}."
                        )
                    if not any(_flow_follows_draw(animation, draw) for draw in matching_draws):
                        raise ValueError(
                            f"Flow animation {_animation_label(animation)!r} in scene "
                            f"{_scene_label(scene)!r} must follow the draw animation for "
                            f"connector {target!r}. Use `after` with that draw animation's ID, "
                            "or literal non-overlapping `at` timestamps."
                        )

            for animation in scene.animations:
                if animation.action != AnimAction.METER_TO:
                    continue
                for target in _animation_target_ids(animation):
                    target_object = object_index.get(target)
                    if target_object is None:
                        raise ValueError(
                            f"Meter-to animation {_animation_label(animation)!r} in scene "
                            f"{_scene_label(scene)!r} targets unknown object {target!r}."
                        )
                    if target_object.type != ObjectType.LINEAR_METER:
                        raise ValueError(
                            f"Meter-to animation {_animation_label(animation)!r} in scene "
                            f"{_scene_label(scene)!r} must target a linear_meter; {target!r} "
                            f"is a {target_object.type.value!r} object."
                        )

        return self


# v1.5 introduces editorial defaults and strict top-level document metadata.
_EDITORIAL_DEFAULTS_VERSION = (1, 5)
_DOCUMENT_TOP_LEVEL_FIELDS = frozenset({"version", "meta", "objects", "scenes"})
_MISPLACED_META_FIELDS = frozenset(
    {
        "title",
        "resolution",
        "fps",
        "theme",
        "audience",
        "story_contract",
        "show_subtitles",
        "show_narration",
        "pacing",
        "continuity",
        "continuity_duration",
        "glow_release_padding",
        "video_bookends",
    }
)


def _is_legacy_document_version(value: object) -> bool:
    """Return whether an explicitly versioned document predates DSL 1.5."""
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?(?:\.\d+)*\s*", value)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) < _EDITORIAL_DEFAULTS_VERSION


def _index_objects(objects: list[ObjectSpec]) -> dict[str, ObjectSpec]:
    """Index scene objects and nested children by explicit ID."""
    indexed: dict[str, ObjectSpec] = {}
    for obj in objects:
        if obj.id:
            indexed[obj.id] = obj
        if obj.children:
            indexed.update(_index_objects(obj.children))
    return indexed


def _animation_target_ids(animation: AnimSpec) -> list[str]:
    """Return concrete target IDs without accepting missing or blank targets."""
    if isinstance(animation.target, str):
        return [animation.target] if animation.target else []
    if isinstance(animation.target, list):
        return [target for target in animation.target if target]
    return []


def _flow_follows_draw(flow: AnimSpec, draw: AnimSpec) -> bool:
    """Check the two timing shapes that can be proven without timeline resolution."""
    if flow.after and draw.id and flow.after == draw.id:
        return True
    if (
        flow.at is None
        or draw.at is None
        or flow.at == "auto"
        or draw.at == "auto"
        or draw.stagger is not None
    ):
        return False
    try:
        return parse_duration(flow.at) >= parse_duration(draw.at) + parse_duration(draw.duration)
    except ValueError:
        return False


def _animation_label(animation: AnimSpec) -> str:
    return animation.id or animation.action.value


def _scene_label(scene: SceneSpec) -> str:
    return scene.id or "(unnamed scene)"
