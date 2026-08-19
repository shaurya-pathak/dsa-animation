from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaivra.mcp.server import KaivraMCPServer, _summarize_tool_result
from kaivra.mcp.workspace import KaivraWorkspace


def test_plan_records_capability_request_and_emits_agent_ready_brief(tmp_path) -> None:
    plan = KaivraWorkspace(tmp_path).plan_animation(
        topic="Database replication",
        capability_requests=[
            {
                "id": "replication_wave",
                "requested_primitive": "replication-wave connector",
                "creative_intent": "Make one write visibly fan out to replicas.",
                "story_moment": "The causal transformation beat.",
                "reusable_scope": "Any one-to-many propagation explainer.",
                "visual_behavior": "A source pulse branches into ordered traveling waves.",
                "acceptance_tests": [
                    "Renders one source and three destinations without crossed connectors.",
                    "The web preview and Cairo render show the same branch order.",
                ],
                "implementation_context": "Implement as a reusable DSL primitive, not a one-off scene asset.",
                "fallback": "Reject a static diagram because it hides propagation order.",
                "product_risk": "Without the primitive, one-to-many causality remains ambiguous.",
                "authorization": "authorized",
            }
        ],
    )

    escalation = plan["capability_escalation"]
    assert escalation["creative_approval"]["separate_from_implementation_readiness"] is True
    assert "does not weaken the concept" in escalation["creative_approval"]["rule"]
    assert "json_authoring_gate" in escalation["implementation_readiness"]
    assert set(escalation["request_fields"]) >= {
        "requested_primitive",
        "acceptance_tests",
        "implementation_context",
        "fallback",
        "product_risk",
        "authorization",
        "resolution_status",
    }
    assert escalation["requests"][0]["resolution_status"] == "pending"
    assert escalation["ready_for_json_authoring"] is False
    brief = escalation["implementation_briefs"][0]
    assert brief["requested_primitive"] == "replication-wave connector"
    assert brief["acceptance_tests"] == escalation["requests"][0]["acceptance_tests"]
    assert "One bounded lower-cost implementation agent" in brief["assignment"]
    assert any(
        "host orchestrator, not Kaivra" in item for item in escalation["host_orchestrator_guidance"]
    )


def test_server_advertises_host_orchestrator_capability_workflow(tmp_path) -> None:
    server = KaivraMCPServer(workspace_root=str(tmp_path))
    initialized = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )[0]["result"]

    instructions = initialized["instructions"]
    assert "host orchestrator—not Kaivra" in instructions
    assert "explicitly accepted fallback and product risk" in instructions

    server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})[0]["result"][
        "tools"
    ]
    plan_tool = next(tool for tool in tools if tool["name"] == "plan_animation")
    assert "host orchestrator (not Kaivra)" in plan_tool["description"]
    assert "capability_requests" in plan_tool["inputSchema"]["properties"]


def test_plan_rejects_an_agent_brief_without_acceptance_context(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing required context"):
        KaivraWorkspace(tmp_path).plan_animation(
            topic="Database replication",
            capability_requests=[{"requested_primitive": "replication wave"}],
        )


def test_plan_summary_keeps_capability_gate_visible() -> None:
    summary = _summarize_tool_result(
        "plan_animation",
        {
            "suggested_meta": {},
            "draft_defaults": {},
            "questions": [],
            "capability_escalation": {
                "requests": [
                    {
                        "id": "wave",
                        "requested_primitive": "replication-wave connector",
                        "resolution_status": "pending",
                    }
                ]
            },
        },
    )

    assert "Capability escalation:" in summary
    assert "host orchestrator—not Kaivra" in summary
    assert "accepted fallback and product risk" in summary
    assert "request wave: replication-wave connector (pending)" in summary


def test_pending_story_capability_blocks_json_until_resolved(tmp_path: Path) -> None:
    workspace = KaivraWorkspace(tmp_path)
    source_path = tmp_path / "animations" / "capability-gate.json"
    canonical_story = (
        Path(__file__).parents[1] / "examples" / "reference" / "forward_propagation.story.md"
    ).read_text(encoding="utf-8")
    pending_table = """| ID | Desired visualization | Story need | Missing reusable capability | Rejected/acceptable fallback | Authorization | Implementation status |
| --- | --- | --- | --- | --- | --- | --- |
| CAP-01 | A visible pile that grows item by item | Make accumulation physically obvious | Accumulating pile primitive | Reject static cards because they hide accumulation | authorized | pending |"""
    pending_story = canonical_story.replace("No missing capabilities.", pending_table)

    created = workspace.create_story_contract(
        animation_path=str(source_path), markdown=pending_story
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "version": "1.5",
                "meta": {
                    "title": "Capability gate",
                    "audience": "layperson",
                    "story_contract": "capability-gate.story.md",
                },
                "scenes": [
                    {
                        "id": "one",
                        "duration": "3s",
                        "objects": [{"id": "item", "type": "text", "content": "One item"}],
                        "animations": [{"action": "fade-in", "target": "item", "at": "0s"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert created["status"] == "capability_pending"
    assert created["creative_verdict"] == "approved"
    assert created["ready_for_authoring"] is False
    checked = workspace.check_animation(file_path=str(source_path))
    assert checked["valid"] is False
    assert any(
        "CAP-01" in issue and "still pending" in issue for issue in checked["blocking_issues"]
    )

    story_path = source_path.with_suffix(".story.md")
    story_path.write_text(pending_story.replace("| pending |", "| implemented |"), encoding="utf-8")
    resolved = workspace.check_animation(file_path=str(source_path))

    assert resolved["valid"] is True
    assert resolved["story_contract"]["ready_for_authoring"] is True
