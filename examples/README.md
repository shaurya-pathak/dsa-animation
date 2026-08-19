# Examples

For layperson work, translate the user's intention into a learner
transformation—not a literal sequence of requested numbers or scenes. A
reviewed brief needs one familiar mental model, the earlier-learning versus
current-prediction boundary, visible causal proof for every operation, and
sound-off, teach-back, counterfactual, and scale-change checks.

For every narrated layperson explainer, write and review `<slug>.story.md` before authoring `<slug>.json`. The brief is the golden contract for the viewer question, definitions, causal ledger, beat sheet, and confusion traps; JSON implements that approved story. Follow the [story-first explainer guide](../docs/STORY_FIRST_EXPLAINERS.md).

Use the full authoring path: learning transformation → persistent visual cast →
motion verb per beat → capability inventory → animatic → final JSON. If an
approved visualization needs a reusable primitive that does not exist, record a
capability escalation in the story work: desired visualization, learning
purpose, missing primitive, fallback impact, authorization, and status. Story
approval is not implementation readiness; JSON waits for the primitive to be
implemented and integrated, or for an explicitly accepted fallback and product
risk.

The canonical story-contract example is [forward_propagation.story.md](reference/forward_propagation.story.md), paired with [forward_propagation.json](reference/forward_propagation.json). Read the story first. The JSON demonstrates DSL syntax such as cues, meters, replacement, and connector flow; it is not a composition, scene rhythm, or motion template to copy.

Signed metrics require no layout workaround: author `content: "-0.40"` with a metric style. Kaivra keeps the magnitude aligned with unsigned peers and renders a compact hanging sign automatically.

- `examples/algorithms/`: compact algorithm walkthroughs and sanity-check examples.
- `examples/demos/`: polished demo-ready animations.
- `examples/explainers/`: longer narrated explainers and architecture walkthroughs.
- `examples/reference/`: reviewed story contracts and focused DSL syntax references.
- `examples/themes/`: reference JSON theme specs for MCP and custom theme authoring.
- `examples/archived/`: older reference files kept for historical context.
- `examples/local/`: local-only scratch variants and draft rewrites that should not be committed.

Rendered media belongs in local `artifacts/` folders, not in `examples/`.
If you are experimenting with alternate explainer/reference files, put them under `examples/local/` so `git status` stays focused on the curated example set.
