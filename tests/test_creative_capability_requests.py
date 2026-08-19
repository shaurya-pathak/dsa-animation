from __future__ import annotations

from pathlib import Path

from kaivra.mcp.story_contract import (
    CreativeCapabilityRequest,
    StoryContractReport,
    parse_creative_capability_requests,
    validate_story_contract_markdown,
)

TABLE = """## Creative capability requests

| ID | Desired visualization | Story need | Missing reusable capability | Rejected/acceptable fallback | Authorization | Implementation status |
| --- | --- | --- | --- | --- | --- | --- |
| CAP-01 | A jar filling with individual examples | Make prior examples visibly accumulate into a learned rule | Example-collection animation | Accept a labeled stack of cards; reject a generic progress bar | authorized | {status} |
"""


def test_no_missing_capabilities_is_an_explicit_ready_approval() -> None:
    review = parse_creative_capability_requests(
        "## Creative capability requests\n\n- No missing capabilities.\n"
    )

    assert review.errors == ()
    assert review.requests == ()
    assert review.verdict == "approved"
    assert review.readiness == "ready"


def test_authorized_pending_request_preserves_creative_approval() -> None:
    review = parse_creative_capability_requests(TABLE.format(status="pending"))

    assert review.errors == ()
    assert review.requests[0].id == "CAP-01"
    assert review.requests[0].implementation_status == "pending"
    assert review.verdict == "approved"
    assert review.readiness == "approved_with_pending_capabilities"


def test_malformed_request_table_is_a_contract_error() -> None:
    markdown = """## Creative capability requests

| ID | Desired visualization |
| --- | --- |
| CAP-01 | A jar |
"""

    review = parse_creative_capability_requests(markdown)

    assert review.requests == ()
    assert any("must use columns" in error for error in review.errors)
    assert any(
        "Creative capability requests table must use columns" in error
        for error in validate_story_contract_markdown(markdown)
    )


def test_implemented_request_is_ready() -> None:
    review = parse_creative_capability_requests(TABLE.format(status="implemented"))

    assert review.errors == ()
    assert review.requests[0].implementation_status == "implemented"
    assert review.readiness == "ready"


def test_explicitly_accepted_fallback_is_ready() -> None:
    review = parse_creative_capability_requests(TABLE.format(status="fallback accepted"))

    assert review.errors == ()
    assert review.requests[0].implementation_status == "fallback accepted"
    assert review.readiness == "ready"


def test_report_serializes_creative_requests_without_making_pending_invalid() -> None:
    request = CreativeCapabilityRequest(
        id="CAP-01",
        desired_visualization="A jar filling with individual examples",
        story_need="Make learning visible",
        missing_reusable_capability="Example-collection animation",
        fallback="Accept cards; reject a progress bar",
        authorization="authorized",
        implementation_status="pending",
    )
    report = StoryContractReport(
        Path("animation.json"),
        Path("animation.story.md"),
        creative_capability_requests=(request,),
        creative_verdict="approved",
        creative_readiness="approved_with_pending_capabilities",
    )

    assert report.valid is True
    assert report.to_dict() == {
        "animation_path": "animation.json",
        "story_contract_path": "animation.story.md",
        "valid": True,
        "errors": [],
        "creative_capability_requests": [
            {
                "id": "CAP-01",
                "desired_visualization": "A jar filling with individual examples",
                "story_need": "Make learning visible",
                "missing_reusable_capability": "Example-collection animation",
                "fallback": "Accept cards; reject a progress bar",
                "authorization": "authorized",
                "implementation_status": "pending",
            }
        ],
        "creative_verdict": "approved",
        "creative_readiness": "approved_with_pending_capabilities",
        "ready_for_authoring": False,
    }
