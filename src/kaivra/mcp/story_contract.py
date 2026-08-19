"""Story-contract helpers for narrated beginner explainers.

The DSL stays portable: it records only a relative sidecar pointer.  Workspace
tools use this module when they have a file path and can verify the companion
Markdown contract before producing an artifact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

STORY_CONTRACT_SUFFIX = ".story.md"
REQUIRED_STORY_CONTRACT_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Intent translation and learning transformation",
        ("Intent translation and learning transformation", "Intent translation"),
    ),
    ("Viewer question and promise", ("Viewer question and promise",)),
    ("Audience starting point", ("Audience starting point", "Audience and scope")),
    ("Concrete entry point", ("Concrete entry point",)),
    ("Mental model and analogy map", ("Mental model and analogy map", "Mental model")),
    (
        "Training, prediction, and outcome boundary",
        ("Training, prediction, and outcome boundary", "Training versus prediction"),
    ),
    ("Prerequisite staircase", ("Prerequisite staircase",)),
    ("Causal ledger", ("Causal ledger",)),
    ("Causal reveal plan", ("Causal reveal plan", "Causal-reveal plan")),
    ("Beat sheet", ("Beat sheet",)),
    ("Misconception map", ("Misconception map",)),
    ("Confusion traps", ("Confusion traps", "Confusion traps to prevent")),
    ("Acceptance checks", ("Acceptance checks", "Acceptance checks before render")),
    ("Creative capability requests", ("Creative capability requests",)),
)
SECTION_EVIDENCE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Intent translation and learning transformation",
        ("before", "after", "goal"),
    ),
    (
        "Mental model and analogy map",
        ("everyday", "machine", "analogy"),
    ),
    (
        "Training, prediction, and outcome boundary",
        ("earlier", "now", "after"),
    ),
    (
        "Prerequisite staircase",
        ("learner question", "visual", "teach-back"),
    ),
    (
        "Causal reveal plan",
        (
            "source",
            "learned rule",
            "rationale before operation/output",
            "visible action",
            "persistent changed state",
            "sound-off",
        ),
    ),
    (
        "Beat sheet",
        ("incoming belief", "learner question", "visual", "outgoing belief"),
    ),
    (
        "Misconception map",
        ("wrong", "repair"),
    ),
    (
        "Acceptance checks",
        ("sound-off", "teach-back", "counterfactual"),
    ),
)
_PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD)\b|\[fill (?:this|in)\]", re.IGNORECASE)
_CAPABILITY_SECTION = "Creative capability requests"
_CAPABILITY_COLUMNS = (
    "ID",
    "Desired visualization",
    "Story need",
    "Missing reusable capability",
    "Rejected/acceptable fallback",
    "Authorization",
    "Implementation status",
)
_CAPABILITY_AUTHORIZATIONS = {"authorized", "not authorized"}
_CAPABILITY_STATUSES = {"pending", "implemented", "fallback accepted"}


@dataclass(frozen=True)
class CreativeCapabilityRequest:
    """A director-approved visual need not supplied by the current component set."""

    id: str
    desired_visualization: str
    story_need: str
    missing_reusable_capability: str
    fallback: str
    authorization: str
    implementation_status: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "desired_visualization": self.desired_visualization,
            "story_need": self.story_need,
            "missing_reusable_capability": self.missing_reusable_capability,
            "fallback": self.fallback,
            "authorization": self.authorization,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class CreativeCapabilityReview:
    """Parsed capability decisions, separate from the structural story review."""

    requests: tuple[CreativeCapabilityRequest, ...] = ()
    verdict: str = "not_recorded"
    readiness: str = "not_recorded"
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoryContractReport:
    """The result of resolving and reviewing a paired Markdown sidecar."""

    animation_path: Path | None
    story_contract_path: Path | None
    errors: tuple[str, ...] = ()
    creative_capability_requests: tuple[CreativeCapabilityRequest, ...] = ()
    creative_verdict: str = "not_recorded"
    creative_readiness: str = "not_recorded"

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ready_for_authoring(self) -> bool:
        """Return whether creative review and capability implementation are complete."""
        return self.valid and self.creative_readiness == "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "animation_path": str(self.animation_path) if self.animation_path else None,
            "story_contract_path": (
                str(self.story_contract_path) if self.story_contract_path else None
            ),
            "valid": self.valid,
            "errors": list(self.errors),
            "creative_capability_requests": [
                request.to_dict() for request in self.creative_capability_requests
            ],
            "creative_verdict": self.creative_verdict,
            "creative_readiness": self.creative_readiness,
            "ready_for_authoring": self.ready_for_authoring,
        }


def paired_story_contract_path(animation_path: str | Path) -> Path:
    """Return the one allowed sidecar location for an animation file."""
    return Path(animation_path).with_suffix(STORY_CONTRACT_SUFFIX)


def story_contract_template(title: str) -> str:
    """Return a deliberate, incomplete template for a new explainer contract."""
    display_title = title.replace("_", " ").replace("-", " ").strip().title()
    display_title = display_title or "Untitled Explainer"
    return f"""# {display_title} — Story Contract

## Intent translation and learning transformation

TODO: Separate the user's underlying learning goal from examples or details
that are merely illustrative. State the learner's Before and After state, the
chosen teaching strategy, and mechanics deliberately deferred.

## Viewer question and promise

TODO: State the concrete question the opening asks and what a first-time viewer
will understand by the end.

## Audience starting point

TODO: State what the viewer already knows and define every term before it is
used as a label, number, or formula.

## Concrete entry point

TODO: Choose a familiar situation, object, or tension that makes the abstract
topic matter before introducing mechanics.

## Mental model and analogy map

TODO: State one transferable plain-language mental model before technical
vocabulary. Add a table that maps everyday element, machine-learning or domain
meaning, and where the analogy stops.

## Training, prediction, and outcome boundary

TODO: Say what happened Earlier, what happens Now, and what happens After the
moment being explained. Make clear what is fixed versus changing.

## Prerequisite staircase

TODO: For every unfamiliar idea, record the viewer's starting belief, learner
question, everyday action, visual proof, technical term introduced afterward,
and a Teach-back sentence.

## Causal ledger

TODO: For every important number or formula, name what it represents, where it
comes from, why the operation happens, and what the result means.

## Causal reveal plan

TODO: For every important transition, record this revealed order: Source,
Learned rule or fixed context, Rationale before operation/output, Visible
action, Persistent changed state, and Sound-off causal evidence. The operation
and its output must start hidden until the viewer has seen why they happen.

## Beat sheet

TODO: Give every beat one new idea, one dominant visual proof, a spoken intent,
and the question it answers. Use columns for Incoming belief, Learner question,
Everyday action, visual proof, Narration's job, Outgoing belief, and a
five-second comprehension prompt.

## Misconception map

TODO: List likely wrong inferences, the frame that could cause each one, and
the positive visual or narrative Repair.

## Confusion traps

TODO: List the transitions, undefined terms, false implications, or premature
payoffs that this explainer must avoid.

## Acceptance checks

TODO: Write observable checks a reviewer can use before the first render.
Include sound-off, teach-back, and counterfactual checks for a layperson.

## Creative capability requests

TODO: Inventory the visualization primitives required by the approved
choreography. If the current library supports all of them, replace this text
with `No missing capabilities.` Otherwise use this exact table and keep the
story approved while implementation is pending:

| ID | Desired visualization | Story need | Missing reusable capability | Rejected/acceptable fallback | Authorization | Implementation status |
| --- | --- | --- | --- | --- | --- | --- |
| CAP-01 | Describe the learner-visible action | Explain why this visual proof matters | Name the smallest reusable primitive | State what a fallback would weaken or explicitly accept | authorized | pending |
"""


def validate_story_contract_markdown(markdown: str) -> tuple[str, ...]:
    """Check that a reviewed contract has the required, non-placeholder sections."""
    if not markdown.strip():
        return ("Story contract is empty.",)

    errors: list[str] = []
    for label, aliases in REQUIRED_STORY_CONTRACT_SECTIONS:
        body = _section_body(markdown, aliases)
        if body is None:
            alternatives = " or ".join(f"## {alias}" for alias in aliases)
            errors.append(f"Story contract is missing required section: {alternatives}.")
            continue
        if not re.search(r"[A-Za-z0-9]", body):
            errors.append(f"Story contract section '{label}' is empty.")

    for label, markers in SECTION_EVIDENCE_MARKERS:
        aliases = next(
            aliases
            for section_label, aliases in REQUIRED_STORY_CONTRACT_SECTIONS
            if section_label == label
        )
        body = _section_body(markdown, aliases)
        if body is None:
            continue
        normalized_body = body.casefold()
        missing = [marker for marker in markers if marker not in normalized_body]
        if missing:
            quoted_markers = ", ".join(repr(marker) for marker in missing)
            errors.append(
                f"Story contract section '{label}' is missing required review evidence: "
                f"{quoted_markers}."
            )

    if _PLACEHOLDER_RE.search(markdown):
        errors.append("Story contract still contains a TODO or other placeholder.")
    errors.extend(parse_creative_capability_requests(markdown).errors)
    return tuple(dict.fromkeys(errors))


def parse_creative_capability_requests(markdown: str) -> CreativeCapabilityReview:
    """Parse the creative director's reusable-capability decision from a contract.

    The section deliberately records unsupported ideas without making a pending,
    authorized implementation a structural validation failure. It must contain
    either ``No missing capabilities.`` or the exact seven-column Markdown table
    described by :data:`_CAPABILITY_COLUMNS`.
    """
    body = _section_body(markdown, (_CAPABILITY_SECTION,))
    if body is None:
        return CreativeCapabilityReview(
            errors=(f"Story contract is missing required section: ## {_CAPABILITY_SECTION}.",),
        )

    nonempty_lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(nonempty_lines) == 1 and re.fullmatch(
        r"(?:[-*]\s+)?no missing capabilities\.?", nonempty_lines[0], re.IGNORECASE
    ):
        return CreativeCapabilityReview(verdict="approved", readiness="ready")

    if len(nonempty_lines) < 3:
        return CreativeCapabilityReview(
            errors=(
                f"Creative capability requests must say 'No missing capabilities.' "
                f"or provide the {_CAPABILITY_COLUMNS!r} Markdown table.",
            ),
        )

    # Keep the section human-readable: the creative director may put a short
    # rationale before the machine-readable table. Find the exact header row
    # instead of treating that rationale as malformed table data.
    header_index = next(
        (
            index
            for index, line in enumerate(nonempty_lines)
            if _parse_markdown_row(line) == list(_CAPABILITY_COLUMNS)
        ),
        None,
    )
    header = _parse_markdown_row(nonempty_lines[header_index]) if header_index is not None else None
    separator = (
        _parse_markdown_row(nonempty_lines[header_index + 1])
        if header_index is not None and header_index + 1 < len(nonempty_lines)
        else None
    )
    if (
        header != list(_CAPABILITY_COLUMNS)
        or separator is None
        or len(separator) != len(header)
        or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
    ):
        return CreativeCapabilityReview(
            errors=(
                "Creative capability requests table must use columns: "
                + ", ".join(_CAPABILITY_COLUMNS)
                + ".",
            ),
        )

    requests: list[CreativeCapabilityRequest] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    rows_start = (header_index or 0) + 2
    for number, line in enumerate(nonempty_lines[rows_start:], start=rows_start + 1):
        row = _parse_markdown_row(line)
        if row is None or len(row) != len(_CAPABILITY_COLUMNS):
            errors.append(f"Creative capability requests row {number} must have seven cells.")
            continue
        if any(not cell for cell in row):
            errors.append(f"Creative capability requests row {number} has an empty required cell.")
            continue
        request_id, *values = row
        if request_id in seen_ids:
            errors.append(f"Creative capability request ID {request_id!r} is duplicated.")
            continue
        seen_ids.add(request_id)
        authorization = values[4].casefold()
        implementation_status = values[5].casefold()
        if authorization not in _CAPABILITY_AUTHORIZATIONS:
            errors.append(
                f"Creative capability request {request_id!r} authorization must be "
                "'authorized' or 'not authorized'."
            )
        if implementation_status not in _CAPABILITY_STATUSES:
            errors.append(
                f"Creative capability request {request_id!r} implementation status must be "
                "'pending', 'implemented', or 'fallback accepted'."
            )
        requests.append(
            CreativeCapabilityRequest(
                id=request_id,
                desired_visualization=values[0],
                story_need=values[1],
                missing_reusable_capability=values[2],
                fallback=values[3],
                authorization=authorization,
                implementation_status=implementation_status,
            )
        )

    parsed_requests = tuple(requests)
    if errors:
        return CreativeCapabilityReview(requests=parsed_requests, errors=tuple(errors))
    pending_requests = tuple(
        request for request in parsed_requests if request.implementation_status == "pending"
    )
    if any(request.authorization == "not authorized" for request in pending_requests):
        return CreativeCapabilityReview(
            parsed_requests, verdict="not_approved", readiness="needs_authorization"
        )
    if pending_requests:
        return CreativeCapabilityReview(
            parsed_requests,
            verdict="approved",
            readiness="approved_with_pending_capabilities",
        )
    return CreativeCapabilityReview(parsed_requests, verdict="approved", readiness="ready")


def validate_paired_story_contract(
    animation_path: str | Path,
    declared_path: str | None,
    *,
    require_declaration: bool,
) -> StoryContractReport:
    """Resolve and validate the contract declared by a file-backed animation."""
    resolved_animation = Path(animation_path).resolve()
    expected_path = paired_story_contract_path(resolved_animation)
    errors: list[str] = []

    if not declared_path or not declared_path.strip():
        if require_declaration:
            errors.append(
                "Beginner explainers require meta.story_contract set to "
                f"'{expected_path.name}' before previewing or rendering."
            )
        return StoryContractReport(resolved_animation, expected_path, tuple(errors))

    candidate = Path(declared_path.strip())
    if candidate.is_absolute():
        errors.append("meta.story_contract must be a relative path beside the animation JSON.")
        return StoryContractReport(resolved_animation, None, tuple(errors))

    resolved_contract = (resolved_animation.parent / candidate).resolve()
    if resolved_contract != expected_path:
        errors.append(
            "meta.story_contract must point to the paired sidecar "
            f"'{expected_path.name}', not '{declared_path}'."
        )
        return StoryContractReport(resolved_animation, resolved_contract, tuple(errors))

    if not resolved_contract.exists():
        errors.append(f"Story contract file not found: {resolved_contract}")
        return StoryContractReport(resolved_animation, resolved_contract, tuple(errors))
    if not resolved_contract.is_file():
        errors.append(f"Story contract path is not a file: {resolved_contract}")
        return StoryContractReport(resolved_animation, resolved_contract, tuple(errors))

    markdown = resolved_contract.read_text(encoding="utf-8")
    creative_review = parse_creative_capability_requests(markdown)
    errors.extend(validate_story_contract_markdown(markdown))
    return StoryContractReport(
        resolved_animation,
        resolved_contract,
        tuple(errors),
        creative_review.requests,
        creative_review.verdict,
        creative_review.readiness,
    )


def _section_body(markdown: str, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        heading = re.compile(rf"(?im)^##\s+{re.escape(alias)}\s*$")
        match = heading.search(markdown)
        if match is None:
            continue
        next_heading = re.search(r"(?m)^##\s+", markdown[match.end() :])
        end = match.end() + next_heading.start() if next_heading else len(markdown)
        return markdown[match.end() : end].strip()
    return None


def _parse_markdown_row(line: str) -> list[str] | None:
    """Return a simple pipe-table row, rejecting prose and escaped pipe cells."""
    if "|" not in line:
        return None
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]
