# Story-First Explainers

An animation JSON is an implementation artifact, not the first expression of
an idea. A narrated layperson explainer begins with a reviewed Markdown story
contract. The contract is the golden source of truth for authors, specialist
reviewers, and renderers.

The goal is not to make a user's example technically present on screen. The
goal is to translate their intention into a learner transformation: what a
person cannot yet explain, what they should be able to explain afterward, and
the visible causal story that gets them there.

## Required artifact

Create one sidecar beside the animation:

    animations/<slug>.story.md
    examples/reference/<slug>.story.md

Set meta.story_contract to that sidecar's filename. File-backed layperson
checks, previews, and renders reject missing, incomplete, or placeholder
contracts.

## Required sections

Every story contract must include these reviewed sections:

1. Intent translation and learning transformation
2. Viewer question and promise
3. Audience starting point
4. Concrete entry point
5. Mental model and analogy map
6. Training, prediction, and outcome boundary
7. Prerequisite staircase
8. Causal ledger
9. Causal reveal plan
10. Beat sheet
11. Misconception map
12. Confusion traps
13. Acceptance checks

The first section separates the user's underlying learning goal from examples,
numbers, and proposed scenes. Write a clear Before / After transformation and
say why the chosen teaching strategy serves it. Record non-goals and deferred
mechanics so the later animation cannot add them merely because they exist in
the domain.

## Golden authoring rules

- Start with a familiar human question, situation, or tension. State the
  technical topic only after the viewer has a reason to care.
- Write one transferable mental model before technical vocabulary. Explicitly
  map its everyday elements to the real mechanism and say where the analogy
  stops.
- Separate what happened earlier, what happens now, and what happens after the
  moment being explained. A learned setting must look pre-existing before it is
  applied to the current case.
- Use a prerequisite staircase: ordinary action first, visual causal proof
  second, technical name third, and a teach-back sentence last.
- Never show a number, formula, or conclusion before its human meaning is
  visible. A label alone is not a meaning.
- Every operation needs three things in the brief: why the input exists, why
  that operation is the right action, and what changes in the persistent
  visual model afterward.
- Require a causal reveal plan for every important transition. It must show
  the source and learned rule before the rationale, then reveal the visible
  action and persistent changed state. An operation or output may not be
  initially visible while its reason is still being introduced.
- Make formal notation secondary proof. A viewer should understand a dimmer,
  scorecard, balance, comparison, or conversion ruler before seeing the
  corresponding formula.
- Once the everyday action is understood, introduce its technical name beside
  the actor it names. A learned multiplier is a **weight**; a **bias** is a
  separate added baseline. Do not teach either term by attaching it to the
  wrong operation.
- When the shape of a named function carries the explanation, show the shape.
  For a sigmoid, prefer a semantic `sigmoid_plot` with a traceable input and
  output over an opaque “converter” box.
- Retain one continuous causal world. Reuse the case object, learned rule,
  running state, and stable actor IDs so a value visibly becomes its
  consequence instead of teleporting into a new slide.
- Write a choreography map before JSON: what enters, what moves, what changes,
  what persists, and what the viewer should notice. Do not begin with a page
  template or a list of scenes.
- Treat scene boundaries as edit and render segments inside that world, not as
  slides. Repeated headings, body cards, chapter rails, and footer trackers are
  presentation chrome, not explanatory structure.
- Keep production reasoning out of the finished piece. Narration must not
  announce that it will explain, teach, unpack, slow down, or walk through the
  subject. Those choices belong in the story contract; the output starts with
  the subject itself.
- Reject slogan-like screen copy assembled from symmetrical fragments—such as
  “ONE INPUT · ONE ANSWER”—unless that phrase is authentic language from the
  domain. Do not add meta labels such as “TECHNICAL NAME,” “KEY TAKEAWAY,” or
  “WHAT WE LEARNED.” Display the actual concept, value, or state instead.
- Make the visual prove narration. Arrows and captions cannot be the only
  explanation; show a bar shorten, a scorecard setting persist, a chip move, a
  balance shift, or two scales align.
- Do not add extra domain mechanisms merely because they are real. If they do
  not answer the viewer's question, defer them to a named follow-up.
- End with a transfer moment: a counterfactual, contrast, or new case the
  viewer can predict. A vocabulary recap is not comprehension evidence.

## Teaching voice and pause contract

- Speak like a teacher working beside the learner. Use familiar invitations
  such as “let's assume,” “let's look at,” “notice what happens,” and “what do
  you think should change?” when they genuinely help the viewer participate.
- State simplifying assumptions as shared setup. Prefer “Let's assume we
  already have a trained model, and let's say it only looks for two clues” over
  detached shorthand such as “the learning already happened.”
- Speak directly about the subject. Ban process narration such as “I'm going
  to show you,” “we'll walk through,” “let's slow this down,” “we're about to
  see,” and “first I'll explain.” Interactive questions and necessary shared
  assumptions are welcome; announcements about the teaching plan are not.
- Ask a short question before an important reveal, then give the viewer a beat
  to predict it. Do not answer the question in the same breath.
- Use contractions and ordinary verbs. If a sentence sounds like release
  notes, documentation, a diagram label, or a lecture abstract, rewrite it.
- Slow comprehension with clear speech, sequential motion, and a short hold on
  the changed state. Do not simulate slow teaching by leaving several seconds
  of empty audio at the end of every scene.
- Keep authored durations and `at` values as readable silent-render fallbacks.
  For voice, synthesize first and map each scene to its measured audio duration
  plus the configured lead and hold. Never guess voice timing from word count.
- In a narrated render, aim for roughly one second of combined tail and lead-in
  between adjacent movements. Longer silence must have a named learning job,
  such as prediction, comparison, or teach-back.

## Causal reveal plan requirements

The causal ledger says what a value means. The causal reveal plan says how a
viewer sees that meaning become true over time. For every important operation
or state change, include this table in revealed order:

| Source | Learned rule or fixed context | Rationale before operation/output | Visible action | Persistent changed state | Sound-off causal evidence |
| --- | --- | --- | --- | --- | --- |
| What is already on screen and where it came from | What is saved, fixed, or otherwise constrains the change | Why this operation is warranted before it appears | The observable motion or transformation, not just a pulse | The same actor or state that remains changed afterward | What a muted reviewer can point to as proof of the whole causal link |

Use one row for every relationship a beginner must understand. For example,
an observed clue appears first, then its saved learned rule, then the reason it
is scaled, then the bar visibly shortens through a dimmer, then the same
scorecard and running balance retain the resulting nudge. Do not place a full
formula, an output, or a completed counterfactual on screen before that row's
rationale has been established.

Sound-off evidence must prove the link without narration: a reviewer should be
able to locate the source, the rule, the action in progress, and the persistent
new state in a paused frame. A caption, arrow, or pulse by itself does not
count as an action or changed state.

## Beat sheet requirements

Each beat must change a belief, not merely rename a diagram. Use these columns:

| Incoming belief | Learner question | Everyday action | One visual causal proof | Narration's job | Outgoing belief | 5-second comprehension prompt |
| --- | --- | --- | --- | --- | --- | --- |

Narration explains cause and consequence. Screen copy carries compact labels,
values, and anchors. Do not spend narration time reading the equation that the
viewer can already see.

## Mandatory review gates

Before the first render, a reviewer must pass all four gates:

1. **Sound-off:** On any paused frame, can a novice identify the real-world
   case, the current observation/source, the stored learned rule, the visible
   action or changing intermediate state, and the persistent current answer?
2. **Teach-back:** Can they explain the causal path in ordinary language,
   without repeating technical labels?
3. **Counterfactual:** Before a reveal, can they predict what will move when a
   clue gets weaker, stronger, or points the other direction—and why?
4. **Scale-change:** When one value becomes another kind of value, can they
   explain what stays the same, what the new scale measures, and why the
   transformation exists?

For a Story Editor, the single release question is:

> Can a first-time layperson explain and predict the causal path without using
> the technical labels?

A no is a block. It cannot be averaged away by correct arithmetic, a polished
theme, or a passing renderer test.

## Authoring loop

1. Transform the learning goal into the Markdown contract: learner change,
   viewer question, mental model, and causal proof.
2. Define a persistent visual cast, then give every beat one motion verb,
   visible state change, and narration job. One creative director owns that
   continuous spatial and motion world.
3. Inventory available capabilities against the choreography. Build and review
   an animatic before JSON so unsupported teaching moves are visible early.
4. If the best approved visualization needs a missing reusable primitive, file
   a capability escalation. Include: desired visualization, learning purpose,
   smallest missing primitive, fallback impact, creative authorization, and
   implementation status (`pending`, `implemented`, or `fallback accepted`) in
   the exact `Creative capability requests` table from
   `kaivra://capability-escalation`. Creative approval approves the story; it
   does not make the implementation ready.
5. For every approved primitive, the host/root orchestrator delegates a
   bounded implementation task to a lower-cost agent, supplies acceptance
   tests, and integrates and verifies it personally. Resume JSON only after
   implementation, or after an explicit accepted fallback with its product risk
   recorded in the contract and animatic.
6. Author JSON strictly from the verified choreography. Use scene boundaries
   only when an edit or render segment is useful.
7. Compare the rendered narration and paused frames against the contract.
   Re-run the causal reveal plan row by row, then the sound-off, teach-back,
   counterfactual, and scale-change gates before release. Listen once without
   watching and reject unexplained inter-scene silence longer than two seconds.

Examples demonstrate DSL syntax only. They are never a composition reference,
script, scene outline, set of numbers to repeat, or permission to skip the
learner's mental model.
