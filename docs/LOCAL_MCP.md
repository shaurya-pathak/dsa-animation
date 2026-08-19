# Local Kaivra MCP

Kaivra includes a local stdio MCP server for guided animation authoring.

## Install

Run these commands from the repo root.

Kaivra targets Python 3.12+. If you use `uv`, the repo's [`.python-version`](/Users/shauryapathak/Desktop/Development/dsa-animation/.python-version) file gives it a supported default interpreter for the local virtualenv.

### 1. Install `uv`

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install system dependencies

macOS:

```bash
brew install cairo pkg-config ffmpeg
```

Ubuntu / Debian:

```bash
sudo apt install libcairo2-dev pkg-config ffmpeg
```

### 3. Install Kaivra core

```bash
make install
source .venv/bin/activate
```

### 4. Optional: install voice support

If you want local or provider-backed narration, install the editable voice package too:

```bash
make install-voice-local
```

That install exposes the built-in `openai`, `local`, and `elevenlabs` providers through Kaivra's provider discovery hooks. Use `--voice-provider` or `KAIVRA_VOICE_PROVIDER` when you want to force one. If you omit `voice_provider`, Kaivra defaults to `openai` for cloud narration.

If you want the default local Sherpa bundle after that:

```bash
kaivra download-model
```

### 5. Verify the local setup

```bash
kaivra doctor
```

If `doctor` is green, point your MCP client at `.venv/bin/kaivra-mcp`. The doctor report prints the exact resolved command path plus the default local voice model name and destination directory.

## MCP Client Setup

### Automatic install

```bash
kaivra mcp-install --client auto
```

`auto` prefers Claude Code when `~/.claude.json` already exists, then Cursor.

### Generic stdio config

```json
{
  "mcpServers": {
    "kaivra": {
      "type": "stdio",
      "command": "/absolute/path/to/this/repo/.venv/bin/kaivra-mcp",
      "args": []
    }
  }
}
```

Use that same command path pattern for Cursor or any other local stdio MCP client.

After any editable install or package update that changes Kaivra code, restart the MCP client so it reloads the `kaivra-mcp` process instead of serving stale imports.
Use `kaivra doctor` if you want to verify the exact binary path your MCP client should be launching.

## Workflow

The MCP is intentionally small and opinionated:

1. `add_theme` creates a reusable custom theme JSON in the workspace.
2. For a narrated layperson explainer, create and review `animations/<slug>.story.md`.
3. `plan_animation` gathers topic, audience, theme, structure, and voice choices from that brief.
4. Write the animation JSON in `animations/` only after the story brief is approved.
5. `check_animation` validates, normalizes, and audits the result.
6. `preview_animation` writes an HTML preview and a PNG still.
7. `render_animation` writes the final PNG, MP4, or WebM artifact.

`animations/` is a local workspace and is gitignored by default. If a draft graduates into a curated repo example, move it into `examples/` intentionally. Keep throwaway or alternate example variants under `examples/local/`.

For every narrated layperson explainer, the Markdown brief is required before JSON authoring: read it, review it for causal gaps, then make the JSON serve it. See the [story-first explainer guide](STORY_FIRST_EXPLAINERS.md). The canonical pair is [the forward-propagation story](../examples/reference/forward_propagation.story.md) and its [reference JSON](../examples/reference/forward_propagation.json); the story is the source of truth, and the JSON is an implementation reference rather than a sequence or number set to copy.

Use `motion_explainer` for narrated work. Write one choreography map for an evolving visual world, then let the same actors move, combine, split, and change state. A single creative director owns the full timeline; specialist agents review it rather than building disconnected scenes. Themes supply palette and typography, not composition. Do not start with a title/body/footer template, repeated cards, chapter rails, or ornamental pulse and glow. Use `move-to`, `replace`, `draw`, `flow`, and meaningful scale changes to explain cause; reserve `fade-in` and `appear` for genuine entrances. Reuse stable IDs and `actor_id` when an actor persists across an edit boundary. Write narration in natural spoken English and avoid filenames, repo paths, or internal inventories unless explicitly requested.

`add_theme` accepts a theme name, an optional `base_theme`, and an `overrides` object whose keys match the `ThemeSpec` fields, such as `accent`, `background_color`, `box_fill`, or `box_border`.

If you edit JSON directly, use `meta.show_subtitles` when you want narration text rendered on screen. Older `meta.show_narration` files still load, but `show_subtitles` is the preferred field name now.

For default cloud narration, the recommended loop is:

1. `export OPENAI_API_KEY=your-key`
2. `kaivra quick-render <file> --voice`

For offline local narration, the recommended loop is:

1. `kaivra doctor`
2. `kaivra download-model`
3. `KAIVRA_VOICE_PROVIDER=local kaivra quick-render <file> --voice`

## Patterns

- `motion_explainer`
- `algorithm_walkthrough`
- `architecture_explainer`
- `before_after_comparison`

## Local Paths

- Source files: `animations/`
- Theme files: `themes/`
- Preview artifacts: `artifacts/previews/`
- Final renders: `artifacts/renders/`
