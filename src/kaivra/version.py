"""Kaivra version constants — single source of truth for DSL and server versions."""

from __future__ import annotations

# Bump this when the DSL schema changes in ways LLMs should know about.
CURRENT_DSL_VERSION = "1.5"

# Short changelog an LLM can scan to understand what changed.
# Keep entries terse — this may be injected into tool responses.
DSL_CHANGELOG: list[tuple[str, str]] = [
    (
        "1.5",
        "Motion-first narrated authoring with required story contracts, one continuous choreography, draw-to-flow validation, reliable multiword narration cues, representative MCP previews, explicit equation alignment, and accessible neutral channel roles. New documents disable bookends, duplicated subtitles, and scene progress chrome by default; v1.4 documents retain their legacy rendering defaults.",
    ),
    (
        "1.4",
        "Added actor_id-based continuity matching, continuity_mode (strict/evolving/position_only), and size_variant (compact/default/hero) for dense storyboard scenes.; Added the storyboard scene template, the system_storyboard starter pattern, and the storyboard_dark built-in theme.",
    ),
    (
        "1.3",
        "Added semantic named regions to the one-column template: problem_solution, request_pipeline, fan_out, system_architecture, and timeline_steps; each maps to a dedicated vertical band in the 12-row grid.; Added relative motion translations (translate/from_translate with normalized multipliers) as the preferred approach for move and move-to animations; absolute pixel offset fields (offset_x/y, from_offset_x/y) are retained for backwards compatibility; the unused absolute layout mode is removed from the DSL surface.; Added semantic timing tokens (short/medium/long) and high-level reveal/reveal-children animation actions; introduced repo-level kaivra.config.json for timing customization; added anchor, gap, cue, step, and order fields to AnimSpec for declarative animation sequencing.",
    ),
    (
        "1.2",
        "Added: plan_animation questionnaire guidance, spoken_forms aliases, "
        "provider-aware voice sync diagnostics, scene voice lead-in, "
        "and invisible parent group audit in check_animation.",
    ),
    (
        "1.1",
        "Added: carousel object type, one-column scene template, "
        "persistent document-level objects, show_subtitles flag, "
        "narration field on scenes, pacing presets (quick-demo / balanced / educational), "
        "continuity morphs via reused object IDs across scenes.",
    ),
    ("1.0", "Initial schema."),
]


def version_drift_warning(doc_version: str) -> str | None:
    """Return an actionable compatibility warning for a non-current schema version."""
    current_key = _version_key(CURRENT_DSL_VERSION)
    document_key = _version_key(doc_version)
    if document_key is not None and document_key == current_key:
        return None

    if document_key is not None and current_key is not None and document_key > current_key:
        return (
            f"This document targets newer schema {doc_version!r}, but this Kaivra runtime "
            f"supports {CURRENT_DSL_VERSION!r}. Upgrade Kaivra before relying on newer-only "
            "fields or behavior."
        )

    current_entry = next((msg for ver, msg in DSL_CHANGELOG if ver == CURRENT_DSL_VERSION), None)
    return (
        f"This document targets schema {doc_version!r}, but the current version is "
        f"{CURRENT_DSL_VERSION!r}. {current_entry or ''} "
        f'Update the "version" field to "{CURRENT_DSL_VERSION}" and use the newer features.'
    )


def _version_key(value: str) -> tuple[int, ...] | None:
    """Parse a dotted numeric DSL version for safe ordering comparisons."""
    parts = value.strip().split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers.pop()
    return tuple(numbers)
