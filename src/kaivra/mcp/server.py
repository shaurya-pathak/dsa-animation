"""Minimal stdio MCP server for the guided Kaivra workflow."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kaivra.mcp.resources import list_resources, read_resource
from kaivra.mcp.workspace import KaivraWorkspace
from kaivra.version import CURRENT_DSL_VERSION

SERVER_NAME = "kaivra-local-mcp"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

JSONRPC_VERSION = "2.0"


@dataclass(frozen=True)
class ToolDefinition:
    """Static tool metadata exposed through MCP."""

    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]
    handler: Callable[[dict[str, Any], "ToolContext"], dict[str, Any]]


@dataclass
class ToolContext:
    """Execution context passed to each tool handler."""

    workspace: KaivraWorkspace
    emit_progress: Callable[[float, str], None]


class KaivraMCPServer:
    """Guided local MCP server for Kaivra authoring and rendering."""

    def __init__(self, *, workspace_root: str | None = None) -> None:
        self.workspace = KaivraWorkspace(workspace_root)
        self.initialized = False
        self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]
        self._writer: Callable[[dict[str, Any]], None] | None = None
        self.tools = {tool.name: tool for tool in _build_tools()}

    def serve(self) -> None:
        """Serve newline-delimited JSON-RPC messages over stdio."""
        self._writer = self._write_message
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._write_message(_error_response(None, -32700, f"Invalid JSON: {exc.msg}"))
                continue

            responses = self.handle_message(message)
            for response in responses:
                self._write_message(response)

    def handle_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle a single parsed JSON-RPC message."""
        if not isinstance(message, dict):
            return [_error_response(None, -32600, "Messages must be JSON objects.")]

        method = message.get("method")
        if not isinstance(method, str):
            return [_error_response(message.get("id"), -32600, "Missing method name.")]

        request_id = message.get("id")
        params = message.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return [_error_response(request_id, -32602, "params must be an object.")]

        try:
            result = self._dispatch(method, params)
        except MCPError as exc:
            if request_id is None:
                return []
            return [_error_response(request_id, exc.code, exc.message)]
        except Exception as exc:  # pragma: no cover - defensive fallback
            traceback.print_exc(file=sys.stderr)
            if request_id is None:
                return []
            return [_error_response(request_id, -32603, f"Internal error: {exc}")]

        if request_id is None:
            return []
        return [_success_response(request_id, result)]

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "notifications/initialized":
            self.initialized = True
            return {}
        if method == "ping":
            return {}

        if not self.initialized:
            raise MCPError(-32002, "Server not initialized.")

        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "title": tool.title,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                        "annotations": tool.annotations,
                    }
                    for tool in self.tools.values()
                ]
            }

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str):
                raise MCPError(-32602, "tools/call requires a string tool name.")
            if not isinstance(arguments, dict):
                raise MCPError(-32602, "tools/call requires arguments to be an object.")
            meta = params.get("_meta")
            if meta is not None and not isinstance(meta, dict):
                raise MCPError(-32602, "tools/call _meta must be an object when provided.")
            return self._call_tool(name, arguments, meta if isinstance(meta, dict) else None)

        if method == "resources/list":
            return {"resources": list_resources()}

        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise MCPError(-32602, "resources/read requires a string uri.")
            return read_resource(uri)

        raise MCPError(-32601, f"Unknown method: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS:
            self.protocol_version = requested
        else:
            self.protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]

        return {
            "protocolVersion": self.protocol_version,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
            },
            "instructions": (
                f'Kaivra DSL v{CURRENT_DSL_VERSION}. Always set "version": "{CURRENT_DSL_VERSION}". '
                "Workflow: plan_animation → create and read <slug>.story.md → write JSON → check_animation → preview_animation → render_animation. "
                "Start with plan_animation when user preferences are still missing. If the user already gave enough direction, "
                "assume the draft defaults but still create and review the story contract before writing JSON. "
                "For every layperson explainer, read kaivra://story-contract, translate the user's intention, examples, and constraints "
                "into the paired Markdown contract, and set meta.story_contract to its filename. Start with the learner's before-and-after transformation, one familiar mental model, "
                "the earlier-training versus current-prediction boundary, and a misconception map. Do not copy a reference example's numbers, claims, or scene sequence directly into DSL. "
                "Use create_story_contract to write the sidecar; file-backed layperson checks, previews, and renders block until it is complete. "
                "Teach everyday action and visible causal change before technical vocabulary or formal notation, then place the correct technical name beside the actor it names. A learned multiplier is a weight; bias is a separate added baseline. When a named function's shape teaches the mapping, show it: use sigmoid_plot for score-to-probability instead of an opaque converter box. Require sound-off, teach-back, counterfactual, and scale-change review evidence; "
                "correct labels or arithmetic do not make a layperson explainer releasable on their own. "
                "Reference examples are syntax demonstrations, never visual quality bars or composition templates. "
                "Agree on one story question and a choreography path before authoring. One creative director must own the complete timeline; specialists review it, but do not independently compose isolated scenes. "
                "If the approved concept needs a missing reusable visualization primitive, read kaivra://capability-escalation, keep the creative approval intact, and record a capability request through plan_animation and the story contract. "
                "Creative approval is separate from implementation readiness. The host orchestrator—not Kaivra—assigns one bounded implementation agent per missing reusable primitive, supplies its exact acceptance tests and context, then integrates and reviews the result. "
                "Do not author JSON that depends on the missing primitive until it is implemented, or an explicitly accepted fallback and product risk are recorded. "
                "Default concept-led explainers to pattern: motion_explainer. Treat a theme as palette and typography only. Derive the spatial metaphor, framing, and motion language from the subject. "
                "Scene boundaries are render and edit segments inside one evolving visual world, not new slides. A repeated heading-stage-footer composition, a row of labelled cards, or a new static diagram per beat is a blocked draft. "
                "Do not begin from editorial, storyboard, one-column, or two-column templates for a narrated explainer. Build a custom scene or group layout around the causal actors. Use large direct-on-canvas values only when they are actors in the transformation. Keep screen copy to fragments, values, and symbols; let narration carry full sentences. "
                "Wrap related actors in groups only to control their spatial relationship; do not use boxes or panels as automatic containers. "
                "Keep connected nodes adjacent within groups so connectors don't cross unrelated nodes. "
                "Available layout types: center, grid, flow, stack, split, carousel. "
                "Use move-to, replace, draw, flow, and scale as explanatory verbs. Fade-in is only for a true entrance; it is not the main teaching action. "
                "Do not use pulse, glow, bounce, or idle motion as decoration. Emphasis must come from composition, timing, and a visible state change. "
                "Persist causal actors and values, not labels, chapter rails, sidebars, legends, or navigation chrome. "
                "Reuse the same object id and content across consecutive scenes for smooth continuity morphs. "
                "Add chapter navigation only when the viewer would otherwise lose their place. "
                "Template vs layout: templates remain available for legacy documents and intentionally document-like material. New narrated explainers should use explicit scene/group layouts so composition follows the subject instead of a preset frame. "
                "Connector overlap: connectors route as straight lines between anchors. To avoid crossings, "
                "order objects in the group so that connected nodes are adjacent — the engine does not auto-route around obstacles. "
                "If a connector must span non-adjacent nodes, split into a separate group or use an intermediate waypoint node. "
                "Write narration as conversational spoken English that adds interpretation instead of reading the screen. Speak directly about the subject; keep production reasoning out of the output. Never announce 'I'm going to show you', 'we'll walk through', 'let's slow this down', 'we're about to see', or 'first I'll explain'. Shared assumptions such as 'let's assume' and genuine questions may use 'we' when they actively involve the learner. Read it aloud; avoid 'Welcome', 'In this video', and documentation-style prose. Reject generic editorial slogans such as 'ONE INPUT · ONE ANSWER' and meta labels such as 'TECHNICAL NAME' or 'KEY TAKEAWAY'; screen copy must name a real concept, value, or state. "
                "Keep authored scene duration and at values as readable silent-render fallbacks. For voice, synthesize first and fit each scene to measured audio plus the configured lead and hold; never guess voice duration from word count. Aim for about one second of combined tail and lead-in between adjacent narrated movements; silence longer than two seconds needs a deliberate prediction, comparison, or teach-back job. "
                "Narration sync: use an explicit cue phrase plus an authored at fallback for each spoken visual beat. The engine case-folds and removes punctuation, then matches the full cue as contiguous words. "
                "Use the same words in narration that appear on screen — e.g., if an object has content 'Server', "
                "say 'the server boots' in narration so the reveal lands on that word. "
                "For tricky names, add object.spoken_forms aliases such as ['co pilot', 'cobalt']. "
                "ElevenLabs can provide native word timing; OpenAI, local Sherpa, and local Qwen receive deterministic estimated word timing when native cues are absent. The same cue contract applies to every provider, and local providers also add a short narration lead-in."
            ),
        }

    def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise MCPError(-32602, f"Unknown tool: {name}")

        progress_token = meta.get("progressToken") if meta else None

        def emit_progress(progress: float, message: str) -> None:
            if progress_token is None or self._writer is None:
                return
            self._writer(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": progress_token,
                        "progress": round(max(0.0, min(progress, 1.0)), 3),
                        "total": 1.0,
                        "message": message,
                    },
                }
            )

        context = ToolContext(workspace=self.workspace, emit_progress=emit_progress)
        try:
            result = tool.handler(arguments, context)
            return _tool_success(tool.name, result)
        except Exception as exc:
            return _tool_error(tool.name, str(exc))

    @staticmethod
    def _write_message(message: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()


def _build_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="doctor_kaivra",
            title="Doctor Kaivra",
            description="Check local Kaivra dependencies, workspace access, the resolved kaivra-mcp command path, the default cloud voice provider, and local voice model defaults.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            annotations={
                "title": "Doctor Kaivra",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_doctor_tool,
        ),
        ToolDefinition(
            name="add_theme",
            title="Add Theme",
            description="Create or update a custom Kaivra theme file inside the local workspace. Starts from a base theme and applies overrides.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the custom theme (used as filename and reference).",
                    },
                    "base_theme": {
                        "type": "string",
                        "enum": [
                            "editorial",
                            "material",
                            "whiteboard",
                            "modern",
                            "storyboard_dark",
                        ],
                        "description": "Built-in theme to start from. Defaults to 'editorial'.",
                    },
                    "overrides": {
                        "type": "object",
                        "description": (
                            "Theme fields to override. Valid keys include: "
                            "background_color, primary, accent, success, warning, error, muted, "
                            "text_color, text_light, font_family, font_size_heading, font_size_body, "
                            "box_fill, box_border, box_border_width, box_corner_radius, box_padding, "
                            "token_fill, token_border, connector_color, connector_width, "
                            "gap_small, gap_medium, gap_large, margin, "
                            "shadow (bool), sketch_effect (bool)."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            annotations={
                "title": "Add Theme",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_add_theme_tool,
        ),
        ToolDefinition(
            name="plan_animation",
            title="Plan Animation",
            description="Story-and-choreography planning step — returns a story contract path/template, motion-first draft defaults, a continuous beat outline, review briefs for timeline segments, and capability escalation requests for missing reusable visualization primitives. A creative director can approve the concept while implementation readiness is pending; the host orchestrator (not Kaivra) assigns one bounded implementation agent per primitive, supplies exact acceptance tests/context, and integrates/reviews it. JSON authoring waits for implementation or an explicitly accepted fallback with product risk. Translate user intent into one evolving visual world before authoring layperson JSON. One creative director owns the complete timeline; specialists review rather than independently composing scenes. Default concept-led explainers to motion_explainer, derive composition from the subject, and treat themes as palette only. Write narration for the ear and anchor spoken visual changes with explicit cue phrases plus authored timing fallbacks.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "capability_requests": {
                        "type": "array",
                        "description": "Missing reusable visualization primitives needed by an approved creative concept. These are recorded for the host orchestrator; Kaivra does not spawn agents.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "requested_primitive": {"type": "string"},
                                "creative_intent": {"type": "string"},
                                "story_moment": {"type": "string"},
                                "reusable_scope": {"type": "string"},
                                "visual_behavior": {"type": "string"},
                                "acceptance_tests": {"type": "array", "items": {"type": "string"}},
                                "implementation_context": {"type": "string"},
                                "fallback": {"type": "string"},
                                "product_risk": {"type": "string"},
                                "authorization": {
                                    "type": "string",
                                    "enum": ["authorized", "not authorized"],
                                },
                                "resolution_status": {
                                    "type": "string",
                                    "enum": ["pending", "implemented", "fallback accepted"],
                                },
                            },
                            "required": [
                                "requested_primitive",
                                "creative_intent",
                                "story_moment",
                                "reusable_scope",
                                "visual_behavior",
                                "acceptance_tests",
                                "implementation_context",
                                "fallback",
                                "product_risk",
                                "authorization",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
            annotations={
                "title": "Plan Animation",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_plan_tool,
        ),
        ToolDefinition(
            name="create_story_contract",
            title="Create Story Contract",
            description="Create or update the paired <slug>.story.md contract before authoring a layperson explainer. With no markdown it writes the required template; complete and review every section before previewing or rendering.",
            input_schema={
                "type": "object",
                "properties": {
                    "animation_path": {
                        "type": "string",
                        "description": "Planned animation JSON/YAML path; the sidecar is written beside it.",
                    },
                    "markdown": {
                        "type": "string",
                        "description": (
                            "Reviewed story-contract Markdown. Omit it to create the required template."
                        ),
                    },
                },
                "required": ["animation_path"],
                "additionalProperties": False,
            },
            annotations={
                "title": "Create Story Contract",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_create_story_contract_tool,
        ),
        ToolDefinition(
            name="check_animation",
            title="Check Animation",
            description="Validate and audit a Kaivra JSON file or raw JSON string, including the required paired story-contract gate for file-backed layperson explainers, with optional normalization write-back, narration timing guidance, and provider-aware voice sync guidance.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "dsl_json": {"type": "string"},
                    "write_back": {"type": "boolean"},
                    "voice": {"type": "boolean"},
                    "voice_provider": {
                        "type": "string",
                        "enum": ["openai", "elevenlabs", "local", "qwen"],
                        "description": (
                            "Optional voice provider hint for sync auditing. "
                            "Use this to tailor the warning text. All providers benefit "
                            "from keyword-overlap checks; native cues and deterministic estimated "
                            "word cues both support the same explicit cue phrases."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            annotations={
                "title": "Check Animation",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_check_tool,
        ),
        ToolDefinition(
            name="preview_animation",
            title="Preview Animation",
            description="Verify the paired story contract for layperson explainers, then write a self-contained HTML preview and representative nonblank PNG into artifacts/previews.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "output_name": {"type": "string"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            annotations={
                "title": "Preview Animation",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_preview_tool,
        ),
        ToolDefinition(
            name="render_animation",
            title="Render Animation",
            description="Verify the paired story contract for layperson explainers, then render a Kaivra animation to PNG, MP4, or WebM inside artifacts/renders.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["png", "mp4", "webm"],
                    },
                    "output_name": {"type": "string"},
                    "audio_path": {"type": "string"},
                    "audio_timings_path": {"type": "string"},
                    "voice": {"type": "boolean"},
                    "voice_provider": {
                        "type": "string",
                        "enum": ["openai", "elevenlabs", "local", "qwen"],
                        "description": (
                            "Voice synthesis provider. 'openai' is the default lower-cost cloud narration path "
                            "with deterministic estimated word timing when native cues are absent; "
                            "'elevenlabs' can provide precise native word alignment; 'local' uses lightweight "
                            "offline Sherpa TTS; 'qwen' uses the resident local Qwen3-TTS CoreML worker. "
                            "Both local providers use the same cue contract and a short narration lead-in."
                        ),
                    },
                    "voice_id": {"type": "string"},
                },
                "required": ["file_path", "format"],
                "additionalProperties": False,
            },
            annotations={
                "title": "Render Animation",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            handler=_render_tool,
        ),
    ]


def _doctor_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    del arguments
    return context.workspace.run_doctor()


def _plan_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return context.workspace.plan_animation(
        topic=arguments.get("topic"),
        capability_requests=arguments.get("capability_requests"),
    )


def _create_story_contract_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    context.emit_progress(0.2, "Writing the paired story contract.")
    result = context.workspace.create_story_contract(
        animation_path=arguments["animation_path"],
        markdown=arguments.get("markdown"),
    )
    context.emit_progress(1.0, "Story contract is ready for review.")
    return result


def _add_theme_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    context.emit_progress(0.2, "Building the custom theme.")
    result = context.workspace.add_theme(
        name=arguments["name"],
        base_theme=arguments.get("base_theme"),
        overrides=arguments.get("overrides"),
    )
    context.emit_progress(1.0, "Theme file written to the workspace.")
    return result


def _check_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    context.emit_progress(0.2, "Validating the Kaivra document.")
    result = context.workspace.check_animation(
        file_path=arguments.get("file_path"),
        dsl_json=arguments.get("dsl_json"),
        write_back=bool(arguments.get("write_back", False)),
        voice=bool(arguments.get("voice", False)),
        voice_provider=arguments.get("voice_provider"),
    )
    context.emit_progress(1.0, "Validation and audit complete.")
    return result


def _preview_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    context.emit_progress(0.2, "Building preview artifacts.")
    result = context.workspace.preview_animation(
        file_path=arguments["file_path"],
        output_name=arguments.get("output_name"),
    )
    context.emit_progress(1.0, "Preview artifacts are ready.")
    return result


def _render_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return context.workspace.render_animation(
        file_path=arguments["file_path"],
        format=arguments["format"],
        output_name=arguments.get("output_name"),
        audio_path=arguments.get("audio_path"),
        audio_timings_path=arguments.get("audio_timings_path"),
        voice=bool(arguments.get("voice", False)),
        voice_provider=arguments.get("voice_provider"),
        voice_id=arguments.get("voice_id"),
        progress=context.emit_progress,
    )


def _success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _tool_success(name: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = _summarize_tool_result(name, result)
    return {
        "content": [
            {
                "type": "text",
                "text": summary,
            }
        ],
        "structuredContent": result,
        "isError": False,
    }


def _tool_error(name: str, message: str) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": f"{name} failed: {message}",
            }
        ],
        "structuredContent": {
            "status": "error",
            "error": message,
        },
        "isError": True,
    }


def _summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    if name == "doctor_kaivra":
        return (
            "Kaivra doctor passed."
            if result.get("ok")
            else "Kaivra doctor found local setup issues."
        )
    if name == "plan_animation":
        questions = [
            question
            for question in (result.get("questions") or [])
            if isinstance(question, dict) and question.get("id")
        ]
        suggested_meta = result.get("suggested_meta") or {}
        draft_defaults = result.get("draft_defaults") or {}
        lines = [
            "Animation plan ready. Translate the user's intent and examples into the paired story contract before writing JSON.",
            "",
            "Suggested meta:",
            f"- title: {suggested_meta.get('title', 'Untitled Animation')}",
            f"- theme: {suggested_meta.get('theme', 'editorial')}",
            f"- pacing: {suggested_meta.get('pacing', 'balanced')}",
            f"- audience: {suggested_meta.get('audience', 'mixed')}",
            f"- continuity: {suggested_meta.get('continuity', True)}",
            f"- show_subtitles: {suggested_meta.get('show_subtitles', False)}",
            "",
            "Draft defaults:",
            f"- audience: {draft_defaults.get('audience', 'mixed')}",
            f"- detail_level: {draft_defaults.get('detail_level', 'balanced')}",
            f"- voice_mode: {draft_defaults.get('voice_mode', 'captions')}",
            f"- pattern: {draft_defaults.get('pattern', 'motion_explainer')}",
            f"- theme: {draft_defaults.get('theme', 'editorial')}",
            f"- num_beats: {draft_defaults.get('num_beats', 'auto')}",
            "",
            "Choreography outline:",
        ]
        for outline_item in result.get("choreography_outline") or []:
            if isinstance(outline_item, dict):
                lines.append(
                    f"- {outline_item.get('movement_id', 'movement')}: {outline_item.get('suggested_title', '')}"
                )
        story_contract = result.get("story_contract") or {}
        if story_contract:
            lines.extend(
                [
                    "",
                    "Story checkpoint:",
                    f"- {story_contract.get('story_question', '')}",
                    f"- {story_contract.get('human_checkpoint', '')}",
                    f"- paired sidecar: {story_contract.get('sidecar_path', '<slug>.story.md')}",
                ]
            )
        choreography_briefs = result.get("choreography_review_briefs") or []
        if choreography_briefs:
            lines.extend(["", "Choreography review segments:"])
            for brief in choreography_briefs:
                if isinstance(brief, dict):
                    lines.append(
                        f"- {brief.get('segment_id', 'segment')}: {brief.get('narrative_job', '')} — {brief.get('dominant_visual', '')}"
                    )
        capability_escalation = result.get("capability_escalation") or {}
        capability_requests = capability_escalation.get("requests") or []
        if capability_escalation:
            lines.extend(
                [
                    "",
                    "Capability escalation:",
                    "- Creative approval is separate from implementation readiness; do not weaken the approved concept because a reusable primitive is missing.",
                    "- The host orchestrator—not Kaivra—assigns one bounded implementation agent per missing reusable primitive, gives it the exact acceptance tests/context, then integrates and reviews the result.",
                    "- Wait to author dependent JSON until every request is implemented or an explicitly accepted fallback and product risk are recorded.",
                ]
            )
            for request in capability_requests:
                if isinstance(request, dict):
                    lines.append(
                        f"- request {request.get('id', 'capability')}: {request.get('requested_primitive', '')} ({request.get('resolution_status', 'pending')})"
                    )
        lines.extend(
            [
                "",
                "Questions to collect:",
            ]
        )
        for question in questions:
            default = question.get("default")
            default_suffix = f" (default: {default})" if default is not None else ""
            lines.append(f"- {question['id']}: {question.get('question', '')}{default_suffix}")
        lines.extend(
            [
                "",
                "If voice is enabled, mirror on-screen keywords in narration and add spoken_forms for tricky names.",
                "Default to clear spoken English and avoid file paths, repo-internal names, and implementation inventory unless the user explicitly wants technical detail.",
                "If audience is layperson, strip file paths, repo-internal names, and jargon from narration completely.",
                "For layperson explainers, create and review the paired .story.md contract, then set meta.story_contract to its filename before previewing or rendering.",
                "Prefer motion explainers: one evolving visual world, causal choreography, and state that transforms instead of resetting between beats.",
                "Persist story actors and values, never presentation chrome or repeated navigation.",
                "Read kaivra://story-contract before turning an intention or example into DSL.",
                "Treat reference examples as syntax demonstrations, not visual or compositional quality bars.",
            ]
        )
        return "\n".join(lines)
    if name == "add_theme":
        return f"Theme saved at {result['file_path']}."
    if name == "create_story_contract":
        status = "ready" if result.get("status") == "ok" else "saved as a draft"
        return (
            f"Story contract {status} at {result['story_contract_path']}. "
            f"Set meta.story_contract to {result['meta_value']!r} and review it before JSON authoring."
        )
    if name == "check_animation":
        grouped = result.get("finding_groups") or {}
        blocking = grouped.get("blocking") or result.get("blocking_issues") or []
        quality = grouped.get("quality") or result.get("warnings") or []
        voice_sync = grouped.get("voice_sync") or []
        continuity = grouped.get("continuity") or []
        recommended = result.get("recommended_edits") or []
        warning_count = len(quality) + len(voice_sync) + len(continuity)
        blocking_count = len(blocking)

        parts: list[str] = []

        # Header
        if result.get("valid") and not warning_count:
            parts.append("Animation validated cleanly.")
        elif result.get("valid"):
            parts.append(f"Animation validated with {warning_count} warning(s).")
        else:
            parts.append(f"Animation check found {blocking_count} blocking issue(s).")

        # Blocking issues
        if blocking:
            parts.append("\n**Blocking issues:**")
            for issue in blocking:
                parts.append(f"- {issue}")

        if quality:
            parts.append("\n**Quality checks:**")
            for w in quality:
                parts.append(f"- {w}")

        if voice_sync:
            parts.append("\n**Voice sync:**")
            for warning in voice_sync:
                parts.append(f"- {warning}")

        if continuity:
            parts.append("\n**Continuity:**")
            for warning in continuity:
                parts.append(f"- {warning}")

        # Recommended edits
        if recommended:
            parts.append("\n**Recommended edits:**")
            for edit in recommended:
                if isinstance(edit, dict):
                    action = edit.get("action", "edit")
                    field = edit.get("field")
                    reason = edit.get("reason")
                    field_part = f" on {field}" if field else ""
                    reason_part = f" ({reason})" if reason else ""
                    parts.append(f"- {action}{field_part}{reason_part}")
                else:
                    parts.append(f"- {edit}")

        return "\n".join(parts)
    if name == "preview_animation":
        return f"Preview HTML written to {result['html_path']}."
    if name == "render_animation":
        return f"Render written to {result['artifact_path']}."
    return f"{name} completed."


class MCPError(RuntimeError):
    """Protocol-level JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
