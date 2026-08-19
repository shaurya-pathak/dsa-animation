"""Static MCP resources for guided Kaivra authoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaivra.dsl.schema import DocumentSpec
from kaivra.mcp.story_contract import (
    REQUIRED_STORY_CONTRACT_SECTIONS,
    story_contract_template,
)

RESOURCE_DEFINITIONS = [
    {
        "uri": "kaivra://authoring-profile",
        "name": "authoring_profile",
        "title": "Kaivra Authoring Profile",
        "description": "The recommended subset of the Kaivra DSL for local MCP-guided authoring.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "kaivra://story-contract",
        "name": "story_contract",
        "title": "Story-First Explainer Contract",
        "description": "Required Markdown contract and template for beginner explainers.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "kaivra://capability-escalation",
        "name": "capability_escalation",
        "title": "Creative Capability Escalation",
        "description": "How to request an approved visual primitive that Kaivra does not yet support.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "kaivra://pattern-catalog",
        "name": "pattern_catalog",
        "title": "Starter Pattern Catalog",
        "description": "When to use each supported starter blueprint.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "kaivra://theme-catalog",
        "name": "theme_catalog",
        "title": "Theme Catalog",
        "description": "Built-in theme guidance for the local MCP workflow.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "kaivra://example-catalog",
        "name": "example_catalog",
        "title": "Example Catalog",
        "description": "Curated examples and snippets that show the supported shape.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "kaivra://example/api_how_it_works",
        "name": "example_api_how_it_works",
        "title": "Reference Example: How an API Works",
        "description": "Full reference example JSON for a narrated API explainer.",
        "mimeType": "application/json",
    },
    {
        "uri": "kaivra://example/forward_propagation",
        "name": "example_forward_propagation",
        "title": "Reference Example: Forward Propagation",
        "description": "Full reference example JSON for a narrated neural-network explainer.",
        "mimeType": "application/json",
    },
    {
        "uri": "kaivra://example/perspectiv_medcase_process_explainer",
        "name": "example_perspectiv_medcase_process_explainer",
        "title": "Reference Example: Perspectiv MedCase Process Explainer",
        "description": "Full reference example JSON for a narrated system process explainer.",
        "mimeType": "application/json",
    },
    {
        "uri": "kaivra://example/system_storyboard_demo",
        "name": "example_system_storyboard_demo",
        "title": "Reference Example: System Storyboard Demo",
        "description": "General-purpose reference example JSON for the system_storyboard pattern.",
        "mimeType": "application/json",
    },
    {
        "uri": "kaivra://example/qa_copilot_storyboard",
        "name": "example_qa_copilot_storyboard",
        "title": "Reference Example: QA Copilot Storyboard",
        "description": "Acceptance example JSON for a dense operational storyboard.",
        "mimeType": "application/json",
    },
    {
        "uri": "kaivra://document-schema",
        "name": "document_schema",
        "title": "Document Schema",
        "description": "The full JSON Schema for the Kaivra document format.",
        "mimeType": "application/json",
    },
]


def list_resources() -> list[dict[str, Any]]:
    """Return the MCP resource descriptors."""
    return RESOURCE_DEFINITIONS


def read_resource(uri: str) -> dict[str, Any]:
    """Return the contents for a Kaivra MCP resource."""
    content_map = {
        "kaivra://authoring-profile": _authoring_profile(),
        "kaivra://story-contract": _story_contract_resource(),
        "kaivra://capability-escalation": _capability_escalation_resource(),
        "kaivra://pattern-catalog": _pattern_catalog(),
        "kaivra://theme-catalog": _theme_catalog(),
        "kaivra://example-catalog": _example_catalog(),
        "kaivra://example/api_how_it_works": _reference_example_text("api_how_it_works.json"),
        "kaivra://example/forward_propagation": _reference_example_text("forward_propagation.json"),
        "kaivra://example/perspectiv_medcase_process_explainer": _reference_example_text(
            "perspectiv_medcase_process_explainer.json"
        ),
        "kaivra://example/system_storyboard_demo": _reference_example_text(
            "system_storyboard_demo.json"
        ),
        "kaivra://example/qa_copilot_storyboard": _reference_example_text(
            "qa_copilot_storyboard.json"
        ),
        "kaivra://document-schema": json.dumps(DocumentSpec.model_json_schema(), indent=2),
    }
    if uri not in content_map:
        raise ValueError(f"Unknown resource URI: {uri}")

    resource = next(item for item in RESOURCE_DEFINITIONS if item["uri"] == uri)
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": resource["mimeType"],
                "text": content_map[uri],
            }
        ]
    }


def _story_contract_resource() -> str:
    sections = "\n".join(f"- **{label}**" for label, _aliases in REQUIRED_STORY_CONTRACT_SECTIONS)
    return f"""# Story-First Explainer Contract

For every layperson explainer, create and review `<slug>.story.md` before
writing `<slug>.json`. Set `meta.story_contract` to that filename. The MCP
blocks file-backed layperson checks, previews, and renders when the paired
contract is missing or incomplete.

## Source of truth

Translate the user's intention, examples, and constraints into this document.
Examples demonstrate DSL syntax only: never copy their composition, numbers,
claims, objects, motion language, or scene order into a new animation. The
approved brief and choreography—not a nearby JSON file—are the golden contract
for the creative director, specialist reviewers, and renderers.

## Required sections

{sections}

## Hard rules

- Translate the user's underlying intention into a before-and-after learner
  transformation; examples, values, and proposed scenes are illustrative, not
  a script to reproduce.
- Start with a familiar question. Introduce the technical name only after the
  viewer has a reason to care.
- State one transferable mental model, map every everyday element to the real
  mechanism, and say where the analogy stops.
- Show what happened earlier, what is fixed now, and what appears afterward.
  Do not make learned settings look live or arbitrary.
- Never show a number, formula, or conclusion before its everyday meaning and
  causal visual proof are visible.
- Explain why an operation happens and what visibly changes afterward. A label,
  equation, or arrow alone is not an explanation.
- Give a result a stable name before it changes; when a value moves to a new
  scale, show what stays the same and why the new scale exists.
- Keep one continuous causal world. Reuse the case, learned rules, and running
  state instead of resetting into disconnected slides.
- Include a counterfactual or contrast the viewer can predict before reveal.
- Introduce only mechanisms needed to answer the viewer's question. Omit or
  explicitly defer extra layers, activations, or domain detail.
- Give each movement one dominant visual argument and one visible state change.
- Keep the authoring strategy in this contract, not in the finished narration.
  Do not announce that the video will teach, explain, unpack, slow down, or
  walk through the topic.
- Reject generic slogan copy made from symmetrical fragments such as “ONE
  INPUT · ONE ANSWER.” Do not put meta labels such as “TECHNICAL NAME,” “KEY
  TAKEAWAY,” or “WHAT WE LEARNED” on screen; show the real concept or state.

## Required review gates

- **Sound-off:** a novice can identify the real-world case, current
  observation, stored rule, changing state, and answer in any paused frame.
- **Teach-back:** they can explain the causal path without technical labels.
- **Counterfactual:** they can predict how a changed input moves the result.
- **Scale-change:** they can explain why a raw value and a readable estimate
  differ without claiming new evidence appeared.

## Template

```markdown
{story_contract_template("Example Explainer").rstrip()}
```
"""


def _authoring_profile() -> str:
    return """# Kaivra Authoring Profile

## Defaults

- `motion_explainer` for narrated explainers. It builds one evolving visual world instead of one composition per beat.
- `pacing: educational` for narrated, `balanced` for silent.
- `audience: mixed` by default, but still write for clarity first. Use plain spoken English and avoid file paths, repo names, or module inventories in narration unless the user explicitly wants implementation detail.
- Use `layperson` when you want the checker to push back even harder on jargon, repo names, file paths, and code identifiers in narration.
- The selected theme supplies color and typography only. It must not determine scene composition.
- New or unversioned v1.5 documents default to `editorial` with bookends, subtitles, and scene progress bars disabled. Put metadata inside `meta`; v1.5 rejects unknown top-level fields. Version 1.4 and older preserve the prior `whiteboard` and enabled chrome defaults when those fields are omitted.

## Story Before Scenes

- Translate the user's underlying learning goal before drafting scenes. Their
  examples, values, and requested beats are evidence of intent, not a script to
  copy.
- Write the learner's Before and After state, one transferable mental model,
  the training-versus-prediction boundary, and a misconception map before JSON
  authoring.
- Agree on one viewer question, one familiar causal model, and a short
  belief-changing choreography path before JSON authoring begins.
- Let everyday action and a visible transformation teach the mechanism before
  technical vocabulary and equations appear.
- After the action is clear, put the correct technical name beside the actor.
  A learned multiplier is a weight; a bias is a separate added baseline.
- Show a named function when its shape teaches the mapping. Use
  `sigmoid_plot` for score-to-probability explanations instead of hiding the
  transformation inside a generic box.
- Run sound-off, teach-back, counterfactual, and scale-change checks before
  rendering.

- For every layperson explainer, create and read `<slug>.story.md` before writing `<slug>.json`; set `meta.story_contract` to the sidecar filename.
- Translate the user's intention, examples, and constraints into the story contract first. Examples demonstrate syntax only; never copy their composition, equation sequence, or claims into DSL.
- Use `kaivra://story-contract` for the required headings and hard rules. A file-backed layperson explainer cannot pass `check_animation`, `preview_animation`, or `render_animation` without a complete paired contract.
- Agree on one viewer question and a short choreography path before JSON authoring begins.
- Open with the question, tension, visible change, or surprising number. Do not open with a welcome screen or agenda.
- Give each movement one narrative job and one visible state change.
- One creative director owns the complete spatial and motion continuum. Specialist agents review focused concerns; they do not compose isolated scenes.
- Verify the beat path and choreography map before JSON authoring. Preserve actor identity and spatial causality across edit boundaries.
- If the approved teaching move needs an unsupported reusable primitive, stop
  JSON authoring and file a structured capability escalation. Creative approval
  approves the story, not implementation readiness.

## Scene Construction

- Do not begin a narrated explainer from `editorial`, `storyboard`, `one-column`, or `two-column` templates. Those remain compatibility tools for intentionally document-like material.
- Build the composition from the subject's actors, forces, transformations, and reading path.
- Treat scene boundaries as edit and render segments, not permission to reset into a title, body, and footer slide.
- A repeated heading, stage label, row of cards, or chapter rail is a blocked draft unless that object is part of the subject itself.
- Start with `text`, large `metric` text, `circle`, and `connector`. Add `box` only when a visible boundary is part of the concept.
- Use `linear_meter` for a bounded quantity or signed lean. Use `sigmoid_plot`
  when a learner needs to see how an internal score maps onto a probability;
  set `sigmoid_input` and readable input/output labels, and let the renderer
  derive the plotted point.
- Label a learned multiplier as `WEIGHT` only after its scaling action is
  understandable. Never label a multiplier as bias; bias is a distinct value
  added to the combined score.
- Use `hero-heading`, `metric`, `metric-coral`, `metric-cyan`, `metric-gold`, and `annotation` as direct-on-canvas typography roles. Reserve `success`, `warning`, and `error` variants for actual status, not explanatory channels.
- Keep on-screen copy to fragments, labels, values, and symbols. Narration carries complete sentences; the frame should not transcribe them.
- Screen fragments must belong to the subject. Do not manufacture editorial
  taglines, eyebrow headings, or “ONE X · ONE Y” slogans to fill empty space.
  Empty space is preferable to a caption a human author would not naturally
  choose.
- Author signed metrics as one natural value such as `-0.40`; the engine automatically hangs a compact sign beside the aligned magnitude. Use separate `operator` objects only for standalone operations that need to move or disappear.
- Show arithmetic when possible: reveal an operator, move it into the destination node with `move-to`, then fade it out as the result changes.
- Use `draw` to establish a connector, then `flow` to carry a visible signal from source to destination.
- Use the fewest objects that make the idea immediately understandable. Empty space is useful when it strengthens the reading path.

## Layout Essentials

**This is critical.** Flat object lists with the default `center` layout stack everything on the same point, producing massive overlaps.

- Build a small number of subject-specific groups whose spatial relationship explains the idea.
- Use `flow`, `stack`, `grid`, and `split` as low-level geometry tools, not as page-layout recipes.
- Available layout types: `center`, `grid`, `flow`, `stack`, `split`, `carousel`.
- Use `gap: "small" | "medium" | "large"` on groups to control spacing.
- Use `direction: "horizontal" | "vertical"` on flow/stack layouts.
- If a legacy template is required, treat it as an explicit product choice and keep it out of the default narrated workflow.

**Connector overlap:** The engine does not auto-route connectors. If `check_animation` flags crossover warnings, reorder objects within their group so connected nodes are adjacent, or split objects into smaller groups to keep connector paths clear.

## Document-Level Objects

- Persist story actors, values, and physical context only when the viewer must see them change over time.
- Do not auto-create headings, labels, legends, chapter rails, progress dots, or navigation chrome.
- Persistent objects appear in every scene when `include_persistent_objects: true`; use that capability for the causal world, not presentation furniture.

## Animation and Reveals

- Choreograph transformations first: `move-to`, `replace`, `draw`, `flow`, and meaningful scale changes should show what happened.
- Use `fade-in` or `appear` only for genuine entrances. Opacity changes are not an explanation.
- Never add pulse, glow, bounce, repeated highlighting, or idle motion merely to make a static composition feel animated.
- For narrated reveals, add an explicit `cue` phrase and an authored `at` fallback. Voice renders use the cue; silent previews use `at`.
- Chain dependent movement with animation IDs plus `after`; one speech cue should anchor one visual beat.
- Do not use `reveal-children` for multiple independently spoken text items. Reveal each item in spoken order.
- Use `draw` on connectors to animate them in. Connectors without `draw` appear instantly.
- Follow `draw` with `flow` when the viewer needs to see direction or causality along an established path.
- `fade-in` on a group ID reveals the group and all its children. You don't need separate animations for children unless you want them staggered.
- Layout-only container groups under `auto_visible: false` should usually set `visible: true` unless you plan to animate the group itself.

## Narration

- Write narration as conversational spoken English with contractions and direct address. Not "Title. Definition."
- Read every line aloud. Rewrite anything that sounds like a heading, caption, documentation paragraph, or list.
- Speak like a teacher beside the learner: use familiar invitations such as "let's assume", "let's look at", and "what do you think should happen?" when they help the viewer participate.
- State assumptions as shared setup. Prefer "Let's assume we already have a trained model, and let's say it only looks for two clues" over detached shorthand such as "the learning already happened."
- Do not begin with "Welcome", "In this video", or "Today we will". Start with the idea.
- Do not narrate the teaching plan anywhere: no "I'm going to show you", "we'll
  walk through", "let's slow this down", "we're about to see", or "first I'll
  explain". Shared assumptions and genuine questions can use "we"; production
  reasoning cannot leak into the spoken output.
- Narration should interpret what is visible, not read every label or equation verbatim.
- Default to an understandable explainer voice even for `mixed` audiences. Explain the user-facing process, not the repo structure.
- Mention labels and values in the order you want reveals to land.
- Let the explanation determine scene length. Authored duration is the silent-render fallback; a voice render synthesizes first and fits each scene to measured audio plus the configured lead and hold. Never guess voice duration from word count.
- `check_animation` reports estimated read time at roughly 150 WPM. Treat that as a comprehension warning for silent fallback timing, not a source of truth for voice duration. Voice renders measure the generated audio directly; verify the resulting pauses by listening.
- Aim for about one second of combined tail and lead-in between adjacent narrated movements. Silence longer than two seconds needs a deliberate prediction, comparison, or teach-back job.

## Voice Sync Checklist

When authoring for voice:

- **Mirror on-screen labels in narration.** If a box says "Load Balancer", say "the load balancer distributes" in narration — not "the distributor sends". The engine matches spoken words to animation targets by content overlap.
- **Order narration to match reveal order.** Mention concepts in the same sequence as their `at` timings so reveals land naturally.
- **Run `check_animation` with `voice: true`** to catch targets with no keyword overlap before rendering.
- **Add `spoken_forms` for tricky names.** If a label is often spoken or transcribed differently, give the object aliases like `"spoken_forms": ["co pilot", "cobalt"]` so checks and cue matching still recognize it.

### Good
Object: `{ "id": "server", "content": "Server" }`
Narration: "First, the server receives the request..."
→ Engine matches "server" in speech to the Server object reveal.

### Bad
Object: `{ "id": "server", "content": "Server" }`
Narration: "First, the backend component handles incoming traffic..."
→ No word overlap — reveal falls back to positional matching (less precise).

**Note:** Object-content checks still accept useful semantic overlap — "failure" can match "fail", and "servers" can match "server". Explicit animation `cue` phrases are stricter: Kaivra case-folds text, removes punctuation, and matches the complete phrase as contiguous spoken words across either native phrase cues or deterministic estimated word cues. Pair every cue with `at` so silent previews retain their authored timing.

## Continuity

- Prefer persistent story actors and changing values, then continuity morphs for local objects that truly evolve.
- Reuse the same `id` and `content` across consecutive scenes when a value carries forward. The engine morphs it into its new position automatically.
- Use `actor_id` when the same visual actor should carry across scenes even if local object IDs change by slot or region.
- Use `continuity_mode: "evolving"` for moderate copy changes on the same actor, or `continuity_mode: "position_only"` for abstract dense actors where motion matters more than text identity.
- When a data structure spans scenes (array, graph, pipeline), keep the same object IDs. Recreating with new IDs each scene kills the smooth morph.
- When a concept repeats the same operation, show one concrete worked example, then generalize.

## Common Mistakes

- Reusing the same `id` for a different label in the next scene. Keep the content close if you want a morph; otherwise rename the object.
- Forgetting to add `actor_id` when the same actor moves between different slots in a storyboard.
- Leaving top-level objects flat under the default center layout. Wrap rows and columns in `group` containers with `flow` or `stack`.
- Drawing connectors across unrelated nodes. Keep connected objects adjacent in their group so straight-line connectors stay legible.
- Forgetting `spoken_forms` on names the TTS or aligner may hear differently.
- Hiding a layout-only parent group under `auto_visible: false` without either `visible: true` or a group-level reveal.

## Workflow

1. Transform the learning goal into a reviewed story contract.
2. Define the persistent visual cast, then assign one motion verb and visible
   state change to every beat.
3. Inventory the available capabilities against that choreography. If a
   necessary reusable primitive is absent, submit `kaivra://capability-escalation`.
4. Build and review an animatic; do not treat creative approval as proof that
   the requested primitive is implemented.
5. The host/root orchestrator delegates each approved primitive to a bounded,
   lower-cost implementation agent with acceptance tests, integrates the
   result personally, and resumes JSON only after implementation or an
   explicitly accepted fallback and product risk.
6. Write topic-specific JSON with `meta.story_contract: "<slug>.story.md"`,
   then `check_animation` → `preview_animation` → `render_animation`.
"""


def _capability_escalation_resource() -> str:
    return """# Creative Capability Escalation

Use this when the creative director can say: “This visualization best teaches
the idea, but the reusable component does not exist yet; I approve the story
and request the component.” This is a product request, not JSON to improvise.

## Story-contract format

```markdown
## Creative capability requests

| ID | Desired visualization | Story need | Missing reusable capability | Rejected/acceptable fallback | Authorization | Implementation status |
| --- | --- | --- | --- | --- | --- | --- |
| CAP-01 | The learner-visible action and persistent actor | The belief or causal relationship this makes understandable | The smallest reusable DSL/renderer primitive, not a one-off scene | What an existing workaround would hide, weaken, or explicitly accept | authorized | pending |
```

If the capability inventory finds no gap, write exactly `No missing
capabilities.` instead of the table. Allowed implementation statuses are
`pending`, `implemented`, and `fallback accepted`. A pending request may remain
creatively approved, but it is not ready for dependent JSON authoring.

## Handoff rule

Creative approval means the story is approved. It does **not** mean the
primitive exists or that JSON authoring may proceed. The host/root orchestrator
turns every approved primitive into a bounded task for a lower-cost
implementation agent, provides acceptance tests, personally integrates and
verifies the result, then resumes JSON authoring. JSON may resume earlier only
when the creative director explicitly accepts a fallback and its product risk.

Keep the escalation attached to the story contract and capability inventory so
the animatic records whether it uses the implemented primitive or the accepted
fallback.
"""


def _pattern_catalog() -> str:
    return """# Starter Pattern Catalog

## `motion_explainer`

Default for narrated explainers. Establish the subject once, then let its actors move, combine, split, transform, and hand state forward through one continuous visual world.

## `system_storyboard`

Legacy/intentionally operational format for dense dashboards, support lanes, and many simultaneous actors. It is not a default explainer composition.

## `algorithm_walkthrough`

Sequence with a clear active step and surrounding context (compare/swap/progress beats).

## `architecture_explainer`

Systems or pipeline explanation with visible stages and connections. Use this only when the viewer truly needs a component map, not as the default for narrated explainers.

## `before_after_comparison`

Contrasting states, revisions, or outcomes.

Patterns are behavioral starting points, not visual scaffolds. The subject's causal choreography determines the composition.
"""


def _theme_catalog() -> str:
    return """# Theme Catalog

Themes supply palette, typography, and primitive styling. They do not supply a composition, card grammar, scene rhythm, or motion language.

## `editorial`

- Flat warm palette and direct typography
- Flat warm canvas, direct typography, large metric roles, crisp circles and connectors, no shadows
- Pair `draw` with `flow` for directional motion

## `material`

- Material UI inspired sample for future custom-theme prompts
- Clean light surfaces, blue accent, generous radius, subtle elevation

## `modern`

- Card-based product presentation style
- Soft depth and UI-like surfaces; use only when the subject benefits from visible containers

## `storyboard_dark`

- Best for dense operational storyboards and high-contrast engineering explainers
- Dark canvas, compact actor treatment, and stronger state-color separation

## `whiteboard`

- Best for teaching, sketches, and conceptual walkthroughs
- Hand-drawn feel with stronger borders and lighter background

Recommendation:

- Pick a theme after the story contract and choreography map exist
- Reach for `storyboard_dark` only when the subject genuinely needs a dense operational field
- Reach for `material` when the user wants a product-UI feel or asks for a theme example to customize
- Switch to `whiteboard` only when the user explicitly wants a sketch or classroom feel
- Use `add_theme` when the user wants a reusable custom palette or card treatment
"""


def _example_catalog() -> str:
    return """# Example Catalog

These files demonstrate syntax and renderer capabilities only. Do not borrow their composition, scene boundaries, card grammar, navigation, or motion language. Derive those from the current subject and its choreography map.

## Full Reference Examples

Read a reference only when you need a concrete DSL syntax example:

- **`examples/reference/forward_propagation.json`** — Story-contract, cue timing, meter, replace, draw, and flow syntax.
- **`examples/reference/perspectiv_medcase_process_explainer.json`** — Carousel and persistent-state syntax for legacy or intentionally dashboard-like work.
- **`examples/reference/system_storyboard_demo.json`** — Dense-grid and actor-identity syntax.
- **`examples/reference/api_how_it_works.json`** — Connector and continuity syntax.
- **`examples/demos/semantic_one_column_regions.json`** — Legacy document-region syntax; do not use it as a narrated-explainer composition model.

The current story contract and choreography map—not any repository example—are the quality bar.

## Blocked composition grammar

Reject a narrated draft when each beat resets into the same heading, stage
label, row of cards, and footer tracker. Also reject animations whose primary
action is staggered fades, pulse, glow, or a new static diagram per beat. Do not
include those structures as copyable JSON examples, even when labeling them as
bad; author from the choreography map instead.

## Actor Continuity

Reuse the same `id` and `content` in consecutive scenes — the engine glides the object to its new position.

```json
{
  "scenes": [
    {
      "id": "weighted_sum",
      "objects": [{ "type": "box", "id": "result_val", "content": "0.36" }]
    },
    {
      "id": "activation",
      "objects": [{ "type": "box", "id": "result_val", "content": "0.36" }]
    }
  ]
}
```

## Continuous Motion Fragment

This fragment demonstrates causal motion without prescribing a page composition. One actor remains on screen while a path draws, a signal travels, and the actor's state changes.

```json
{
  "version": "1.5",
  "meta": {"title": "A Tiny Probability", "theme": "editorial", "video_bookends": false},
  "scenes": [
    {
      "id": "continuous_motion", "duration": "8s", "auto_visible": false,
      "narration": "This clue travels into the model, changes its running evidence, and nudges the prediction toward dog.",
      "objects": [
        {"type": "circle", "id": "clue", "content": "floppy ears", "style": "coral", "size_variant": "large"},
        {"type": "circle", "id": "evidence", "content": "evidence", "style": "cyan", "size_variant": "hero"},
        {"type": "text", "id": "prediction", "content": "more likely dog", "style": "metric-gold"},
        {"type": "connector", "id": "path", "from": "clue", "to": "evidence", "style": "coral"}
      ],
      "animations": [
        {"action": "appear", "target": ["clue", "evidence"], "at": "0s"},
        {"id": "draw_path", "action": "draw", "target": "path", "at": "2s", "duration": "0.8s"},
        {"action": "flow", "target": "path", "after": "draw_path", "duration": "1s"},
        {"action": "replace", "target": "evidence", "content": "evidence rises", "at": "4s", "duration": "0.8s"},
        {"action": "appear", "target": "prediction", "at": "5.2s"}
      ]
    }
  ]
}
```

The important relationship is `draw` → `flow` → persistent state change. The example intentionally does not define a reusable heading, card, or footer composition.
"""


def _reference_example_text(filename: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "examples" / "reference" / filename).read_text(encoding="utf-8")
