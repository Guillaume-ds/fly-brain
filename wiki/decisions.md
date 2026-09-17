# Decisions

A running log, in the order decisions were actually made. Each entry:
what was being decided, what we chose, and why — including the
alternatives that were seriously considered and rejected, since the
rejection reasons are often as useful as the choice itself.

## 1. Data source: public GCS bucket, not the neuprint API

**Context:** the MaleCNS connectome can be accessed either via
`neuprint-python` (a queryable API requiring a free account + token) or as
flat files in a public, no-auth Google Cloud Storage bucket.

**Decision:** use the flat files (`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`).

**Why:** zero account/auth friction for anyone cloning the repo, and the
file schemas were verified directly (actual columns, actual row counts)
rather than assumed — `body-annotations` (14MB), `body-neurotransmitters`
(43MB), and `connectome-weights-*-significant-only` (500MB) cover
everything the project needs. The larger per-synapse files
(`syn-points`, multi-GB) were deliberately skipped — body-level summed
weights are enough for everything planned here.

## 2. Training approach: real synapses via RL, not a bolted-on model

**Context:** three ways to make the connectome "trainable" were on the
table: (a) keep real topology, train real synaptic weights via RL; (b)
freeze the whole real circuit and train a separate linear readout on top
(reservoir-computing style); (c) implement the real dopamine-gated
plasticity rule the mushroom body actually uses.

**Decision:** (a) — train the real synapses directly via RL, so the
circuit's own output neurons are the decision-maker.

**Why:** option (b) was the cheapest to build, but the actual decision
boundary would live in a separately-trained readout, not in the brain
itself — the brain would just be a fixed feature extractor underneath
someone else's model. That's a legitimate technique, but it doesn't match
the goal of the brain itself acting. Option (c) is the most biologically
precise but narrow — a good future addition, not a foundation. Option (a)
is what the actual Google/DeepMind connectome-constrained research does,
and it's the one where "the brain acts" is literally true, not just a
framing.

## 3. First RL task: escape reflex, not the warehouse

**Context:** the original project pitch was a warehouse-sorting game
(navigate, avoid obstacles, pick up colored boxes, deliver to matching
zones).

**Decision:** don't start there. Start with a much smaller task: react to
a threat by fleeing, using the real Giant Fiber (`DNp01`) circuit that was
already built and demoed.

**Why:** the warehouse bundles several hard RL problems at once
(navigation, obstacle avoidance, object interaction, delivery correctness)
into one long episode with one end-of-episode signal — bad credit
assignment, and no way to tell whether a training failure is the reward
design, the environment, or the algorithm. The escape task has an
immediate, clearly-attributable reward, needs no physics/environment
build to start, and is a real, well-studied fly behavior (looming-evoked
escape) with a genuine speed-accuracy tradeoff — interesting to train, not
trivial.

## 4. Curriculum: escape → noisy escape → forage (with transfer)

**Decision:** three stages —
1. escape from a clean threat signal
2. escape amid distractor noise (harder version of the *same* task —
   curriculum learning / domain randomization, not transfer learning)
3. forage for food while threats and noise are both present (a
   genuinely *different* task — this is where real transfer learning
   happens, by reusing the stage-2 escape circuit)

**Why the 1→2 vs 2→3 distinction matters:** stages 1 and 2 test whether
training generalizes to harder input, not whether it transfers to a new
problem — worth naming precisely rather than calling the whole thing
"transfer learning" loosely.

## 5. Transfer mechanism: freeze + override, not joint fine-tuning

**Context:** moving from stage 2 (escape) to stage 3 (forage), the
trained escape circuit needs to either stay frozen while new foraging
circuitry is added, or be fine-tuned jointly with it.

**Decision:** freeze the escape circuit's weights after stage 2. Add a
separate, newly-trainable pathway for foraging/search. Combine via a
simple priority rule: if the frozen escape circuit fires, it overrides
whatever the foraging pathway was doing for that tick.

**Why:** joint fine-tuning risks catastrophic forgetting — nothing
structurally stops an optimizer chasing foraging reward from quietly
eroding escape performance, especially since threat events may be a small
fraction of total reward if infrequent. Freeze+override guarantees stage-2
competency is provably unchanged (diffable weights), is lower-risk to
implement, and happens to match real fly neuroanatomy — the Giant Fiber is
a dedicated fast-escape pathway that overrides ongoing behavior, not
something blended into general locomotor control. Fine-tuning is still a
good follow-up experiment to compare against later, just not the default.

## 6. RL algorithm: Evolution Strategies first, REINFORCE planned next

**Context:** the LIF circuit's spikes are non-differentiable, so standard
backprop doesn't directly apply. Two realistic options: Evolution
Strategies (black-box optimization, perturb weights + rank by episode
reward) or Policy Gradient/REINFORCE (treat spikes as sampled actions,
backprop through log-probability).

**Decision:** implement ES first. REINFORCE is planned as a deliberate
second implementation on the same task, to compare — not a replacement.

**Why:** ES sidesteps the differentiability question entirely and has far
fewer moving parts (no autodiff framework needed, nothing to make the
circuit differentiable), which matters for a first RL implementation where
minimizing confounding sources of bugs is worth more than sample
efficiency. REINFORCE is the more "canonical" RL algorithm and a better
foundation if this project becomes a stepping stone into deeper RL later
— documenting the ES design with that switch in mind (see `stack.md`)
rather than treating ES as the final answer.

## 7. What's trainable: synaptic gain only, not topology or sign

**Decision:** the real connectome's topology (who connects to whom) and
each synapse's sign (excitatory/inhibitory, from predicted
neurotransmitter) stay fixed. The only trainable parameter is a
per-synapse `synaptic_gain` multiplier, initialized to `1.0`.

**Why:** this is the constraint that makes "trained on real biology"
actually true rather than nominal — training can strengthen or weaken
real connections, but can't rewire the fly's brain into an arbitrary
network or invert what a real inhibitory neuron does. It also means a
freshly-built, untrained circuit behaves identically to the original
`simulate` demo (gains all `1.0`), which is a good sanity check in
itself.

## 8. Escape action: trained *whether*, geometric *where*

**Decision:** `EscapeAgent` trains only the binary decision of whether to
flee (does the real downstream `TTMn` motor neuron spike), and computes
flee *direction* as plain geometry (directly away from the threat),
outside the circuit.

**Why:** the real Giant Fiber circuit triggers a stereotyped takeoff, not
a steered one — asking the trained circuit to also encode direction would
be claiming more biological fidelity than the data supports. `TTMn` (the
real jump-muscle motor neuron) wasn't assumed either — it showed up as an
actual downstream neuron of `DNp01` in the built circuit, confirmed via
the `simulate` demo, before being adopted as the trained agent's decision
point.

## 9. World: survival grid, not the eventual open-world yet

**Context:** the long-term vision is an open-ended, randomly-generated
2D world with traps, threats, and fruit (a "2D Minecraft-like" world).

**Decision:** V1 is a small fixed-size grid: a handful of food/threat
entities placed randomly at reset, a hunger bar that depletes every tick
(death at 0), reward is flat `+1` per tick alive. Food and threat sensing
both use a radius-gated gradient signal that's exactly zero outside range
— finding food requires actually searching, not just climbing a gradient
from anywhere.

**Why:** matches the stated goal of keeping every stage as simple as
possible. The environment is a single parameterized `Environment` class
(food/threats can each be toggled off), so the curriculum stages above are
different configs of one implementation, not three separate ones — the
open-world/random-generation/distinct-trap-types version is an explicit
later addition once this loop is proven out, not part of the foundation.

## 10. Threats: separate sensing radius from a stricter contact/death rule

**Decision:** threats have a larger `sense_radius` (gives a warning
signal, still safe) and a separate, stricter death condition — only
triggers if the fly ends a move exactly on the threat's cell.

**Why:** if sensing and death were the same event, there'd be no window
for the escape circuit to actually react — this split is what gives the
trained reflex something meaningful to do. (This was a judgment call made
while implementing `world/env.py`, flagged for confirmation rather than
assumed silently.)

## 11. Project process: a maintained wiki, not just code

**Decision:** keep this `wiki/` folder of plain markdown files, updated as
the project progresses, rather than only relying on commit messages or
chat history.

**Why:** part of the project's goal is showcasing a deliberate way of
working with AI on a real build — a running architecture/decision record
makes that process visible and reviewable, and doubles as the fastest way
to resume work accurately after time away.

## 12. Found: static threats give ES nothing to train against — OPEN QUESTION

**Context:** `training/trainer.py` and `training/curriculum.py` were built
and mechanically verified (correct OpenAI-ES update, checkpointing, CLI).
Running it on stage 1 (clean escape) produced a completely flat fitness —
`population_reward_std = 0.00` every iteration.

**What's happening:** a swept range of threat densities (from the original
`grid_size=10, num_threats=4` up to a near-maximal `grid_size=6,
num_threats=6, threat_radius=3.0`, covering nearly the whole map) all gave
a death rate of 0-5% under the untrained circuit. Root cause: threats are
static, and "back away one step until the sensed signal hits zero, then
stop" is a robust fixed point against *any* stationary threat — once the
fly retreats out of every threat's sensing range it is safe forever, and
with `food_enabled=False` in stage 1 there's no other reason to ever move
again. Increasing threat count/radius doesn't change this dynamic, it just
makes the fly retreat a bit further before permanently stopping. Since
nearly every rollout in a population scores identically (full survival),
ES has no fitness variance to compute an update from.

**Status: resolved by #14 below** — continuous spawning (not threat
movement) turned out to be the fix, arrived at via the vision pivot in
#13 rather than by directly patching this. Threat movement, a
standing-still cost, and pulling hunger-driven movement forward were the
three candidates on the table; none were needed in the end.

## 13. Vision pivot: adversarial colony-survival, not construction

**Context:** exploring a "Minecraft-like" building game surfaced a real
tension: no real Drosophila circuit does construction, so extending the
fly brain to cover building would mean either inventing a fictional
module disconnected from actual neuroscience, or leaving the game's
flashiest mechanic with no connection to the trained brain at all.

**Decision:** drop "building." New v1 vision: a human (via an LLM
translator) tries to wipe out a fly colony by adjusting world generation
— specifically, spider and food spawn rates. The colony survives via the
trained real connectome and reproduces under that pressure.

**Why:** keeps the fly honest — survival and foraging only, nothing
invented — and reframes the "opponent" as a cleanly separate system (world
generation) rather than something the fly itself has to do. The
predator-pressure-drives-selection framing is literally natural selection,
a better narrative fit than building ever was, not a compromise forced by
dropping scope. Also considered: reproduction-as-gameplay should reuse the
existing ES mechanics directly (offspring = parent's `synaptic_gain` plus
ES's own perturbation step; survival = the fitness/selection signal) rather
than inventing a separate breeding system — not yet implemented, noted
here as the intended design.

## 14. v1 world-generation contract: two independent, bounded spawn-rate knobs

**Decision:** `Environment` gains two continuous per-tick spawn-rate
parameters — `spider_spawn_rate`, `food_spawn_rate` — each bounded to a
fixed range, replacing the current fixed-count-at-reset placement.
"Spider" is the existing `Threat` (instant death on contact) — no
web/immobilize mechanic yet (planned for v2, dropped by spiders on death).
The two rates are controlled independently (not collapsed into one
composite "difficulty" dial — considered, rejected in favor of keeping
spider pressure and food scarcity separately tunable). Nothing else (grid
size, hunger depletion, sensing radii, contact rule, *where* something
spawns) is exposed to the user/LLM in v1.

**Why:** continuous spawning is what actually gives "rate" meaning. Bounded
ranges structurally enforce that the adversary can't trivialize the game.
Pulled Crafter's actual `worldgen.py` source as a comparison point before
assuming it might already offer this: it generates everything once at
world creation with hardcoded probability constants (`uniform() > 0.993`
for zombies), not a runtime-adjustable rate — confirming this needs a real
mechanic change on our side, not something an existing engine already
provides.

**Correction (see #15):** this entry originally also claimed continuous
spawning alone resolves #12's flat-fitness problem. That claim was
asserted without being tested and turned out to be wrong — see #15 for
the measurement and the actual fix.

## 15. Continuous spawning alone did NOT fix #12 — threats needed to move

**Context:** #14 assumed continuous spawning would remove the "retreat
once, safe forever" fixed point on its own. Tested directly instead of
taking that on faith.

**What was measured:** with continuous spawning active (`spider_spawn_rate
0.08, max_spiders=6`), the untrained circuit's death rate over 60 episodes
was still 0%, identical to the pre-#14 static-placement result. Raising
density to the extreme (`max_spiders=25`, ~25% of a 10×10 grid) only
reached 5% — the same order of magnitude as the densest config already
ruled out under #12. Root cause: new spiders deliberately never spawn on
the fly's current cell, and — unchanged from #12 — spiders still didn't
move once placed. So each individually-spawned spider remained a one-time,
permanently-dodgeable event; more of them accumulating over time doesn't
change that, since the fly can still retreat from each in turn and freeze.
Continuous spawning fixed what it was actually designed to fix (giving
"rate" a meaning at all) but not the thing #14 additionally assumed it
would fix.

**Decision:** added threat movement after all — the original leading
candidate from #12, before the vision pivot temporarily sidelined it.
Each tick, every `Threat` has `threat_move_probability` chance of taking
one random step (bounded to the grid, allowed to land on the fly's cell —
that's the actual kill mechanism now). Tuned empirically: `0.3` gave a
12% death rate with real spread (std ≈ 10 ticks) over 60 episodes;
`0.5` gave less (3/60) — plausibly because faster-moving threats trigger
the escape circuit's reaction more reliably. `0.3` is now the default and
the stage-1 curriculum value.

**Verified fix:** re-ran the actual ES training loop (not just the
death-rate probe) — `population_reward_std` is now consistently nonzero
every iteration (range ≈3.5–10.1 across 15 iterations), and fitness
climbs from a 73.3 baseline toward the episode ceiling. `decisions.md`
#12 is genuinely resolved now, confirmed by running the real training
loop, not inferred from the mechanic change alone.

## 16. `director/` architecture: registry + swappable controller interface

**Decision:** built as designed in the earlier discussion —
`director/actions.py` holds `ActionSpec(name, description, fn)` entries
wrapping `Environment`'s four rate methods (provider-agnostic, no LLM
imports); `director/base.py` defines the one-method `WorldController`
interface (`choose_action(request, actions) -> action name or None`);
`director/rule_based_controller.py` is a zero-dependency keyword-match
implementation; `director/claude_controller.py` is the reference LLM
backend, using real tool-use (each action becomes a zero-argument tool)
rather than parsing free-text output.

**Verified:** registry → `RuleBasedController` → `Environment` tested end
to end — correctly maps requests like "make it harder for the flies" to
`increase_spider_rate` (and correctly returns `None` for unrelated
requests like "what is the weather today"), with the rate actually
changing on `Environment`. `ClaudeController` was verified to import,
subclass `WorldController` correctly, and construct an `anthropic.Anthropic`
client — but **not exercised against a live API**: no credentials were
available in this environment (no `ANTHROPIC_API_KEY`, no `ant` CLI). A
call without credentials fails client-side with `TypeError: Could not
resolve authentication method...` (not `AuthenticationError` — that
would require a bad-but-present key; here there's nothing to send at
all). Needs testing with a real API key before relying on it.
