# Editorial Release Council

Use this review protocol before promoting an editorial change to the private MCP beta.

## Shared contract

Every reviewer receives the target audience, private-beta constraints, the DSL
compatibility policy, the exact change under review, and only the artifacts
needed for its specialty. Reviewers should challenge the change rather than
approve it by default.

## Specialist roles

| Reviewer | Focused context | Single reward | Must push back on |
| --- | --- | --- | --- |
| Story Editor | Viewer brief, narration, storyboard, canonical example | Viewer comprehension and factual correctness | Weak hooks, repetitive beats, incorrect math, and narration that merely reads the screen |
| Art Director | Theme tokens, captures, typography roles, layout | Immediate visual hierarchy | Low contrast, ambiguous color meaning, inert whitespace, weak payoff, and illegible annotations |
| Renderer Parity Engineer | Cairo and web code, fonts, keyframes | Cross-renderer visual consistency | Font drift, clipping, equation jumps, sign overflow, and mismatched flow motion |
| DSL Steward | Schema, parser, versioning, examples, MCP guidance | Safe and coherent authoring contract | Silent ignored fields, undocumented default changes, and invalid animation semantics |
| Release Sentinel | Audits, tests, package output, beta limits, deployment setup | Evidence-backed deployability | Audit failures, missing regression coverage, oversized payloads, and unverified container behavior |

## Required review response

Each reviewer returns only:

- One score from 0 to 100 for its stated reward.
- Concrete blockers with evidence.
- The smallest corrective change for each blocker.
- One verdict: `block`, `revise`, or `approve`.

A blocker cannot be averaged away. The release coordinator may accept a risk
only with a written product decision that names the risk, owner, and follow-up.

## Review loop

1. The implementation owner supplies deterministic render captures and test evidence.
2. Specialists review independently against their single reward.
3. The coordinator groups findings into blockers, revisions, and accepted risks
   without rewriting specialist conclusions.
4. Re-run only the specialists whose blocking area changed, then ask the Release
   Sentinel for the final verdict.
5. Promote only when every specialist approves or every remaining risk has an
   explicit product decision attached to the release.

## Editorial v1.5 evidence

- Full tests, formatting, lint, compilation, clean wheel install, and MCP initialization.
- Legacy 1.4 and current 1.5 default-behavior fixtures.
- Cue matching for multiword phrases, punctuation, repeated words, fallbacks,
  and dependent timing chains.
- Cairo and browser captures at 1920x1080 and 1280x720 for metrics, equations,
  connectors, and `draw` followed by `flow`.
- A clean audit and MP4 smoke render for the forward-propagation reference.
- A nonblank MCP preview and confirmation that the curated reference fits the
  private-beta payload limit.
