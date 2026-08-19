from __future__ import annotations

import json
from pathlib import Path

from kaivra.mcp.resources import read_resource
from kaivra.mcp.server import KaivraMCPServer, _summarize_tool_result


def test_server_initialization_and_tool_call(tmp_path: Path) -> None:
    server = KaivraMCPServer(workspace_root=str(tmp_path))

    init_response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )[0]
    assert init_response["result"]["serverInfo"]["name"] == "kaivra-local-mcp"
    instructions = init_response["result"]["instructions"]
    assert "create and read <slug>.story.md" in instructions
    assert "kaivra://story-contract" in instructions
    assert "create_story_contract" in instructions
    assert "move-to" in instructions
    assert "draw" in instructions
    assert "continuity" in instructions
    assert "conversational spoken English" in instructions
    assert "Template vs layout" in instructions
    assert "Connector overlap" in instructions
    assert "Narration sync" in instructions
    assert "spoken_forms" in instructions
    assert "deterministic estimated word timing" in instructions
    assert "OpenAI" in instructions
    assert "Persist causal actors and values" in instructions
    assert "assume the draft defaults" in instructions
    assert "Do not use pulse, glow" in instructions
    assert "one evolving visual world" in instructions
    assert "blocked draft" in instructions
    assert "production reasoning out of the output" in instructions
    assert "ONE INPUT · ONE ANSWER" in instructions

    assert (
        server.handle_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        == []
    )

    tools_response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )[0]
    tool_names = {tool["name"] for tool in tools_response["result"]["tools"]}
    assert "add_theme" in tool_names
    assert "create_story_contract" in tool_names
    assert "render_animation" in tool_names
    assert "start_animation" not in tool_names
    assert "quick_render" not in tool_names

    resources_response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
            "params": {},
        }
    )[0]
    resource_names = {resource["name"] for resource in resources_response["result"]["resources"]}
    assert "authoring_profile" in resource_names
    assert "story_contract" in resource_names
    assert "capability_escalation" in resource_names
    assert "example_api_how_it_works" in resource_names
    assert "example_forward_propagation" in resource_names
    assert "example_perspectiv_medcase_process_explainer" in resource_names


def test_resource_guidance_promotes_story_first_motion_authoring() -> None:
    authoring = read_resource("kaivra://authoring-profile")["contents"][0]["text"]
    story_contract = read_resource("kaivra://story-contract")["contents"][0]["text"]
    capability_escalation = read_resource("kaivra://capability-escalation")["contents"][0]["text"]
    pattern_catalog = read_resource("kaivra://pattern-catalog")["contents"][0]["text"]
    examples = read_resource("kaivra://example-catalog")["contents"][0]["text"]
    api_example = read_resource("kaivra://example/api_how_it_works")["contents"][0]["text"]
    medcase_example = read_resource("kaivra://example/perspectiv_medcase_process_explainer")[
        "contents"
    ][0]["text"]

    assert "motion_explainer" in authoring
    assert "visual_explainer" not in authoring
    assert "process_explainer" not in authoring
    assert "educational" in authoring
    assert "fade-in" in authoring
    assert "same `id` and `content`" in authoring
    assert "draw" in authoring
    assert "document-like" in authoring
    assert "visible: true" in authoring
    assert "Persist story actors" in authoring
    assert "Voice Sync Checklist" in authoring
    assert "Common Mistakes" in authoring
    assert "150 WPM" in authoring
    assert "positional matching" in authoring
    assert "understandable explainer voice" in authoring
    assert "contiguous spoken words" in authoring
    assert "metric-coral" in authoring
    assert "Reserve `success`, `warning`, and `error`" in authoring
    assert "Do not narrate the teaching plan anywhere" in authoring
    assert "ONE X · ONE Y" in authoring
    assert "v1.5 rejects unknown top-level fields" in authoring
    assert "meta.story_contract" in authoring
    assert "Examples demonstrate DSL syntax only" in story_contract
    assert "Causal ledger" in story_contract
    assert "Causal reveal plan" in story_contract
    assert "Acceptance checks" in story_contract
    assert "Creative capability requests" in story_contract
    assert "Implementation status" in capability_escalation
    assert "lower-cost" in capability_escalation
    assert "motion_explainer" in pattern_catalog
    assert "visual_explainer" not in pattern_catalog
    assert "process_explainer" not in pattern_catalog
    assert "algorithm_walkthrough" in pattern_catalog
    assert "behavioral starting points, not visual scaffolds" in pattern_catalog
    assert "process-first story arc" not in pattern_catalog
    assert "Blocked composition grammar" in examples
    assert "heading, stage" in examples
    assert "Actor Continuity" in examples
    assert "perspectiv_medcase_process_explainer.json" in examples
    assert '"version": "1.5"' in examples
    assert '"template": "editorial"' not in examples
    assert '"action": "flow"' in examples
    assert '"title": "How an API Works"' in api_example
    assert '"title": "How Perspectiv MedCase Works"' in medcase_example


def test_document_schema_advertises_strict_v15_top_level_fields() -> None:
    schema_text = read_resource("kaivra://document-schema")["contents"][0]["text"]
    schema = json.loads(schema_text)

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"version", "meta", "objects", "scenes"}


def test_plan_animation_defaults_are_supported_and_story_first(tmp_path: Path) -> None:
    server = KaivraMCPServer(workspace_root=str(tmp_path))

    plan = server.workspace.plan_animation(topic="Queueing")
    pattern_question = next(
        question for question in plan["questions"] if question["id"] == "pattern"
    )
    patterns = {option["value"] for option in pattern_question["options"]}

    assert plan["draft_defaults"]["pattern"] == "motion_explainer"
    assert plan["suggested_meta"]["theme"] == "editorial"
    assert "motion_explainer" in patterns
    assert "visual_explainer" not in patterns
    assert "process_explainer" not in patterns
    assert "reference_examples" not in plan
    voice_question = next(
        question for question in plan["questions"] if question["id"] == "voice_mode"
    )
    assert "qwen" in {option["value"] for option in voice_question["options"]}
    assert plan["story_contract"]["sidecar_path"] == "animations/queueing.story.md"
    assert "Causal ledger" in plan["story_contract"]["template"]


def test_create_story_contract_tool_writes_paired_markdown(tmp_path: Path) -> None:
    server = KaivraMCPServer(workspace_root=str(tmp_path))
    server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "create_story_contract",
                "arguments": {"animation_path": "animations/dog-or-cat.json"},
            },
        }
    )[0]
    result = response["result"]["structuredContent"]

    assert result["status"] == "draft"
    assert Path(result["story_contract_path"]).exists()
    assert result["meta_value"] == "dog-or-cat.story.md"


def test_motion_resource_avoids_copyable_slideshow_composition() -> None:
    examples = read_resource("kaivra://example-catalog")["contents"][0]["text"]
    sample = examples.split("## Continuous Motion Fragment", maxsplit=1)[1]

    assert '"style": "coral"' in sample
    assert '"style": "cyan"' in sample
    assert "metric-gold" in sample
    assert '"id": "draw_path"' in sample
    assert '"after": "draw_path"' in sample
    assert '"template"' not in sample
    assert '"style": "heading"' not in sample
    assert '"action": "pulse"' not in sample
    assert '"action": "highlight"' not in sample


def test_check_animation_summary_mentions_warning_count() -> None:
    summary = _summarize_tool_result(
        "check_animation",
        {
            "valid": True,
            "finding_groups": {
                "blocking": [],
                "quality": ["warning one"],
                "voice_sync": ["voice warning"],
                "continuity": ["continuity warning"],
            },
            "recommended_edits": [
                {
                    "action": "enable_layout_group_visibility",
                    "field": "scenes[0].objects",
                    "reason": "Set visible on layout-only group.",
                }
            ],
        },
    )
    assert "Animation validated with 3 warning(s)." in summary
    assert "- warning one" in summary
    assert "**Voice sync:**" in summary
    assert "- voice warning" in summary
    assert "**Continuity:**" in summary
    assert "- continuity warning" in summary
    assert "enable_layout_group_visibility on scenes[0].objects" in summary


def test_plan_animation_summary_mentions_voice_sync_guidance() -> None:
    summary = _summarize_tool_result(
        "plan_animation",
        {
            "status": "ok",
            "suggested_meta": {
                "title": "Queues",
                "theme": "editorial",
                "pacing": "balanced",
                "audience": "mixed",
                "continuity": True,
                "show_subtitles": False,
            },
            "draft_defaults": {
                "audience": "mixed",
                "detail_level": "balanced",
                "voice_mode": "captions",
                "pattern": "motion_explainer",
                "theme": "editorial",
                "num_beats": "auto",
            },
            "choreography_outline": [
                {"movement_id": "arrival", "suggested_title": "A queue arrives"},
                {"movement_id": "change", "suggested_title": "The queue changes"},
            ],
            "questions": [
                {"id": "audience"},
                {"id": "detail_level"},
                {"id": "voice_mode"},
            ],
        },
    )

    assert "mirror on-screen keywords" in summary
    assert "spoken_forms" in summary
    assert "Questions to collect:" in summary
    assert "- audience:" in summary
    assert "- detail_level:" in summary
    assert "- voice_mode:" in summary
    assert "Draft defaults:" in summary
    assert "Choreography outline:" in summary
    assert "- arrival: A queue arrives" in summary
    assert "Embedded reference examples:" not in summary
    assert "Prefer motion explainers" in summary
    assert "Persist story actors and values" in summary
    assert "story contract before writing JSON" in summary
    assert "If audience is layperson" in summary


def test_render_tool_exposes_voice_fields_and_emits_progress(tmp_path: Path) -> None:
    server = KaivraMCPServer(workspace_root=str(tmp_path))
    server._writer = lambda message: emitted.append(message)
    emitted: list[dict] = []

    server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    server.handle_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
    )

    tools_response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )[0]
    render_tool = next(
        tool for tool in tools_response["result"]["tools"] if tool["name"] == "render_animation"
    )
    check_tool = next(
        tool for tool in tools_response["result"]["tools"] if tool["name"] == "check_animation"
    )
    plan_tool = next(
        tool for tool in tools_response["result"]["tools"] if tool["name"] == "plan_animation"
    )
    assert "voice" in render_tool["inputSchema"]["properties"]
    assert "voice_provider" in render_tool["inputSchema"]["properties"]
    assert "voice_id" in render_tool["inputSchema"]["properties"]
    assert "voice_provider" in check_tool["inputSchema"]["properties"]
    assert "openai" in render_tool["inputSchema"]["properties"]["voice_provider"]["enum"]
    assert "openai" in check_tool["inputSchema"]["properties"]["voice_provider"]["enum"]
    assert "qwen" in render_tool["inputSchema"]["properties"]["voice_provider"]["enum"]
    assert "qwen" in check_tool["inputSchema"]["properties"]["voice_provider"]["enum"]
    add_theme_tool = next(
        tool for tool in tools_response["result"]["tools"] if tool["name"] == "add_theme"
    )
    preview_tool = next(
        tool for tool in tools_response["result"]["tools"] if tool["name"] == "preview_animation"
    )
    assert "storyboard_dark" in add_theme_tool["inputSchema"]["properties"]["base_theme"]["enum"]
    assert "representative nonblank PNG" in preview_tool["description"]
    assert "narration timing guidance" in check_tool["description"]
    assert "story contract" in plan_tool["description"]
    assert "review briefs for timeline segments" in plan_tool["description"]
    assert "one evolving visual world" in plan_tool["description"]
    assert "themes as palette only" in plan_tool["description"]
    assert "Write narration for the ear" in plan_tool["description"]

    captured: dict[str, object] = {}

    def fake_render_animation(**kwargs):
        captured.update(kwargs)
        progress = kwargs["progress"]
        progress(0.1, "Discovering voice provider: local.")
        progress(1.0, "Narrated render complete.")
        return {
            "status": "ok",
            "artifact_path": str(tmp_path / "out.mp4"),
            "duration_seconds": 1.2,
            "warnings": [],
            "source_file_path": str(tmp_path / "demo.json"),
        }

    server.workspace.render_animation = fake_render_animation
    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "render_animation",
                "arguments": {
                    "file_path": "animations/demo.json",
                    "format": "mp4",
                    "voice": True,
                    "voice_provider": "local",
                    "voice_id": "amy",
                },
                "_meta": {"progressToken": "voice-render"},
            },
        }
    )[0]

    assert response["result"]["isError"] is False
    assert captured["voice"] is True
    assert captured["voice_provider"] == "local"
    assert captured["voice_id"] == "amy"
    progress_messages = [
        message["params"]["message"]
        for message in emitted
        if message.get("method") == "notifications/progress"
    ]
    assert "Discovering voice provider: local." in progress_messages
    assert "Narrated render complete." in progress_messages
