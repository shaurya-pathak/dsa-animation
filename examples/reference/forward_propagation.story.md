# How a Model Chooses Between a Dog and a Cat — Story Contract

## Intent translation and learning transformation

The goal is not to preserve a neural-network equation. It is to let a beginner
see why a trained model measures clues, scales them differently, combines their
effects, and only then reports a probability.

- **Before:** multiplication and the jump from `0.57` to `64%` feel arbitrary.
- **After:** the viewer can explain the full prediction in ordinary language
  and predict what happens when one clue weakens.
- **Teaching strategy:** keep one mystery pet on screen while evidence visibly
  moves through saved settings into one changing belief.
- **Deferred:** layers, matrices, bias, loss, backpropagation, and training
  mechanics.

## Viewer question and promise

**How does a trained model look at one new photo and decide whether it sees a
dog or a cat?**

The viewer will follow one answer from photo to probability without needing
machine-learning vocabulary first.

## Audience starting point

The audience understands photos, clues, volume controls, addition, and
percentages. It does not yet know feature, weight, raw score, sigmoid, or
forward propagation. Those names arrive only after their actions are clear.

## Concrete entry point

Begin with one large mystery pet between the words **DOG** and **CAT**. Hold the
image long enough for the question to land. This is a deliberately tiny model:
it notices only floppy ears and a long snout.

## Mental model and analogy map

The mental model is **two clues passing through saved volume settings, making
two small pushes on the same decision**.

| Everyday element | Machine meaning | Where the analogy stops |
| --- | --- | --- |
| What the photo visibly shows | Feature strength | Real models detect many learned patterns, not two hand-labelled parts. |
| A saved volume setting | Learned weight | It is a number learned earlier, not a physical knob. |
| A small push toward dog | Weighted contribution | A contribution is not yet a probability. |
| One dog-or-cat balance | Raw score | The score can extend beyond zero and one. |
| A friendlier ruler | Sigmoid | It changes the scale, not the evidence. |

## Training, prediction, and outcome boundary

- **Earlier:** labelled examples taught the model how much different clues
  should count.
- **Now:** the saved settings stay fixed while one new photo creates clue
  strengths and small pushes.
- **After:** the combined lean is translated into a readable probability. No
  retraining happens during this prediction.

## Prerequisite staircase

| Starting belief | Learner question | Everyday action | Visual proof | Technical name introduced later | Teach-back |
| --- | --- | --- | --- | --- | --- |
| The photo becomes an answer immediately. | What happens first? | Notice two clues. | Ear and snout cues lead to two meters. | Feature | “The first numbers describe the photo, not the answer.” |
| Every clue counts equally. | Why use different multipliers? | Reuse settings learned earlier. | Two fixed rule meters stay unchanged. | Weight | “Each clue has its own saved usefulness.” |
| Multiplication is arbitrary. | Why multiply? | Turn a clue up or down. | The signal passes through its rule and becomes a smaller push. | Weighted contribution | “The saved setting scales the clue.” |
| `0.57` means 57 percent. | Why does 64 percent appear? | Read the same lean on a bounded ruler. | An S-shaped sigmoid curve shows `0 → 50%` and places `0.57 → 64%`. | Sigmoid | “The evidence stayed the same; only the ruler changed.” |

## Causal ledger

| Value | Source | Why the operation happens | Meaning afterward |
| --- | --- | --- | --- |
| Ears `7/10` | The new photo | Measure how strongly floppy ears appear. | One observation, not a dog probability. |
| Ear weight `× 0.50` | Earlier training | Scale the clue by its saved usefulness. | A `+0.35` push toward dog. |
| Snout `4/10` | The new photo | Measure how strongly a long snout appears. | A second observation. |
| Snout weight `× 0.55` | Earlier training | Use this clue's own saved usefulness. | A `+0.22` push toward dog. |
| Lean `+0.57` | The two pushes | Both affect the same decision, so they add. | An internal dogward lean, not 57 percent. |
| Chance `64% dog` | The sigmoid curve | Express the lean on a bounded human-readable scale. | The same evidence as a probability estimate. |

## Causal reveal plan

| Source | Learned rule or fixed context | Rationale before operation/output | Visible action | Persistent changed state | Sound-off causal evidence |
| --- | --- | --- | --- | --- | --- |
| One mystery-pet photo | The model is already trained. | We want a careful dog-or-cat guess. | The pet appears before either answer. | The unresolved pet remains through the film. | The viewer sees one case and two possible outcomes. |
| Floppy ears and long snout | Two settings saved earlier. | Different clues need not count equally. | The two fixed meters appear beside the same pet and are visibly named **WEIGHT**. | Both settings remain unchanged. | “Learned earlier” is visually separate from today's measurements. |
| Ear strength `7/10` | Ear weight `× 0.50`. | Strong evidence is scaled by learned usefulness. | A signal travels pet → clue → weight → `+0.35`. | The ear push remains. | Every number has a visible source and destination. |
| Snout strength `4/10` | Snout weight `× 0.55`. | This clue has its own saved setting. | A second signal travels the parallel path to `+0.22`. | Both pushes remain together. | The different multiplier has visible provenance. |
| `+0.35` and `+0.22` | One shared decision. | Both pushes affect the same question. | Two paths converge on one lean meter. | The meter rests at `+0.57`. | Two sources visibly become one result. |
| Lean `+0.57` | Fixed sigmoid function. | People need a bounded probability. | The lean enters an S-shaped curve; guides visibly map `0.57` to `64% DOG`. | Lean, curve, and chance remain traceable side by side. | The conversion cannot be mistaken for new evidence. |
| Ears weaken from `7/10` to `2/10` | Ear weight remains `× 0.50`. | Test whether the viewer understands direction. | Ear, lean, and chance change in place. | The chance settles near `58% DOG`. | Only the current clue moves; the saved weight does not. |

## Beat sheet

| Movement | Incoming belief | Learner question | Everyday action | Visual proof | Narration's job | Outgoing belief | Five-second prompt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Question | The topic is unknown. | What must the model decide? | Look at one pet. | Pet first; DOG and CAT second. | Establish curiosity and scope. | We are following one prediction. | “What is the choice?” |
| Saved settings | The model invents rules now. | Where do different multipliers come from? | Reuse what earlier examples taught. | Two fixed rule meters appear. | Separate earlier learning from today's prediction. | The settings pre-exist the photo. | “Are the rules changing now?” |
| Two clue paths | Math seems arbitrary. | Why multiply each clue? | Turn each clue through its own saved setting. | Two signals create `+0.35` and `+0.22`. | Explain the action before naming it. | Multiplication scales influence. | “Why are the settings different?” |
| One lean | Two pushes look like two answers. | Why add? | Put both pushes on one balance. | Two paths converge on `+0.57`. | Tie addition to one shared decision. | The result is an internal lean. | “Is this a percentage?” |
| Friendly scale | `0.57` looks like 57 percent. | Why does it become 64 percent? | Follow a point along an S-shaped curve. | The curve anchors `0` at `50%`, then maps `0.57` to `64%`. | Explain that no evidence was added. | A sigmoid makes the score readable. | “What changed: evidence or scale?” |
| Counterfactual | The chain may only be memorized. | What if one clue weakens? | Turn down the ears observation. | Ear, lean, and chance move in place while the rule stays fixed. | Invite a prediction and name the process last. | The viewer can predict forward propagation. | “Which way should the chance move?” |

## Misconception map

| Wrong inference | Repair |
| --- | --- |
| Today's photo trains the model. | Show the saved settings before any current clue moves. |
| `7/10` means 70 percent dog. | Label the meter as how clearly the clue appears. |
| The model chooses `0.50` and `0.55` on the fly. | Keep both settings fixed during the counterfactual. |
| `0.50` or `0.55` is a bias. | Label each multiplier **WEIGHT**. A bias is a separate value that is added as a baseline, and this simplified model deliberately defers it. |
| Multiplication is unexplained arithmetic. | Animate the signal through the saved setting before showing its push. |
| `+0.57` means 57 percent. | Keep the lean and bounded chance on visibly different scales. |
| `64%` means certainty. | Call it a careful guess and retain the dog label. |

## Confusion traps

- Do not open with “forward propagation.”
- Do not use a template, title bar, footer, progress rail, or repeated heading.
- Do not reveal a result before its source and reason are visible.
- Do not show a formula as the main event; motion must establish the operation.
- Do not use glow, pulse, bounce, or idle decoration.
- Do not duplicate narration as screen text.
- Do not introduce more than one new causal relationship at once.
- Do not add slogan copy such as “ONE PHOTO · ONE GUESS,” an eyebrow heading,
  or a meta label such as “TECHNICAL NAME.” Use the real concept name or leave
  the space empty.
- Do not let the production plan leak into narration. Phrases such as “we'll
  walk through,” “let's slow this down,” and “I'm going to show you” describe
  the teaching process instead of teaching the subject.
- Do not misname a multiplied setting as a bias. A multiplier is a weight;
  bias is a separate additive baseline and remains deferred in this story.
- Do not hide a named mathematical transformation inside an opaque box when
  its shape explains the idea. Show the sigmoid curve and the point moving
  from score to probability.

## Teaching voice and pauses

- Talk with the viewer, not at them. Use “let's assume” for the trained-model
  setup, “let's look at” when attention moves, and “what do you think should
  happen?” before the counterfactual.
- Prefer familiar teacher diction over compressed technical narration. Never
  use detached phrases such as “the learning already happened” when the real
  idea is “let's assume we already have a trained model.”
- Speak directly about the idea. Use “we” for a genuine shared assumption or
  question, not to announce the structure, pacing, or intention of the video.
- Keep the silence between movements near one second. A longer pause is allowed
  only before a real prediction or after a visually important state change.
- Determine voiced scene lengths from the measured local-TTS clips plus the
  configured lead and hold. The authored durations remain silent fallbacks;
  never infer the final voice timing from a word-rate estimate.
- Slow down by revealing one relationship at a time, not by leaving an empty
  audio tail after the explanation has finished.

## Acceptance checks

1. **Sound-off:** a paused-frame reviewer can trace photo → clue → saved
   setting → push → lean → chance.
2. **Teach-back:** a beginner can explain why each multiplication happens
   before seeing formal notation.
3. **Counterfactual:** before the reveal, the viewer predicts that weaker ears
   move the chance toward even.
4. **Scale-change:** the viewer can explain why `+0.57` is not `57%` and why the
   same evidence becomes roughly `64%`; the graph makes `0 → 50%` and
   `0.57 → 64%` visible.
5. **Pacing:** every new relationship receives a quiet hold after it moves.
6. **Density:** no frame contains more than one dominant transformation and a
   handful of short labels.
7. **Taste:** every shape belongs to the causal world; no presentation chrome
   or decorative motion appears.
8. **Teacher voice:** the narration uses shared assumptions, direct questions,
   and familiar spoken English rather than documentation-style declarations.
9. **Pause discipline:** no inter-scene silence exceeds two seconds unless the
   story contract names the learning task that occupies it.

## Creative capability requests

No missing capabilities.
