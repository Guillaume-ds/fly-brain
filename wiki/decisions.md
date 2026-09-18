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

## 17. Project renamed: `fly-brain` → `kill-the-flies`

**Decision:** the project's display name/branding (README title, wiki
title) changed to `kill-the-flies` — clearer about the actual game
premise (a player, via an LLM, tries to wipe out the colony) than
`fly-brain`, which described the earlier connectome-exploration phase.
The `fly_brain/` Python package name is unchanged — it's an accurate,
descriptive name for the "fly and its brain" subsystem specifically, not
the project's overall branding, and renaming it would touch every import
for a cosmetic reason.

**Update:** the GitHub repository itself has been renamed to
`kill-the-flies` (manual rename via GitHub Settings, done by the project
owner). The README's clone URL and the local `origin` remote have been
updated to match. The `fly_brain/` Python package name is still
unchanged, per the rationale above.

## 18. Code-quality pass: DTOs, logging, function splits, less privacy

**Decision:** reviewed every module against four rules — no unnecessary
`_private` naming, no long comments (rely on this wiki for rationale
instead of duplicating it in docstrings), no `print()` in library code
(use `logging` instead), carry data around as dataclasses rather than
bare tuples/dicts, and split up functions doing too many distinct things.

**What changed:**
- **DTOs**: `circuit.build_circuit()` used to return a bare
  `tuple[list[int], list[tuple[int,int,int]]]`; now returns a
  `CircuitBlueprint(neuron_ids, edges: list[SynapseEdge])`. Added
  `data.ConnectomeData` bundling the three connectome DataFrames that
  `agent.py` and `simulate.py` both load the same way (previously
  duplicated inline in each). Added `world.SensorReading` and
  `simulate.RasterResult` similarly.
- **Logging**: every `print()` in `analyze.py`, `simulate.py`,
  `trainer.py`, and `run.py` replaced with `logging`. Both CLI entry
  points (`fly_brain/cli.py`, `training/run.py`) call
  `logging.basicConfig(level=INFO, format="%(message)s")` so output still
  reads exactly like the old prints did.
- **Function splits**: `Environment.step()` was doing movement, spawning,
  pickup, and episode-end all inline — split into named methods
  (`move_fly`, `move_threats`, `spawn_entities`, `resolve_food_pickup`,
  `determine_episode_end`) with `step()` left as a short orchestrator.
  Same treatment for `analyze.run()` (→ `report_composition`,
  `report_neurotransmitters`, `report_connectivity`,
  `report_top_neurons`, plus pulling a nested closure out to a top-level
  function), `simulate.run()` (→ `run_simulation`,
  `report_most_active_neurons`, `save_raster_plot`), and
  `trainer.train_es()` (→ `run_es_iteration` returning an
  `ESIterationResult`, isolating the actual ES math from the
  loop/logging).
- **Privacy**: removed underscore-prefixing from methods that were just
  internal decomposition, not genuinely dangerous state (e.g.
  `Environment`'s helper methods, `agent.flee_direction`). Kept it only
  where a real invariant would break if called externally: `Circuit`'s
  derived `_edge_pre`/`_edge_post`/`_edge_base_weight` arrays, and
  `data._download` (bypassing it skips `fetch()`'s cache check).

**Verified:** every module still compiles and imports cleanly; re-ran
`simulate`, `analyze`, `EscapeAgent` construction/`act()`, and a 5-iteration
ES training run after the refactor — all produced identical output to the
pre-refactor runs (same baseline fitness 73.3, same neuron counts, same
spike patterns), confirming this was a pure readability pass with no
behavior change.

## 19. Rendering: Python backend + Next.js/TypeScript/Phaser frontend

**Context:** worried the 2D "Terraria-like" tile aesthetic wouldn't work
well in Python, and separately wanted browser playability.

**Decision:** the rendering worry and the browser-deployment goal turned
out to be the same decision, not two — Python doesn't run natively in a
browser, so browser access effectively requires a JS/TS frontend
regardless of how good Python's own 2D rendering could be. Split:
`fly_brain/`, `world/`, `director/`, `training/` stay exactly as they are
(Python, unchanged) as a backend; a new FastAPI + WebSocket layer streams
`Environment` state out as JSON and routes player requests in; a Next.js
+ TypeScript frontend (chosen for existing familiarity, not re-litigated
against alternatives) hosts a Phaser 3 canvas for the actual tile
rendering. See `stack.md` for the full breakdown including the two
alternatives considered and rejected (Pyodide/WASM Python-in-browser;
precomputed static replay).

**Sequencing:** deliberately deferred until the core Python game loop
(reproduction mechanic, a live director loop) works end to end. Building
the frontend against game mechanics that don't exist yet means building
against a moving target — same "prove the simple version first" discipline
used throughout this project. Not started.

## 20. Reproduction: multi-fly `Environment`, random event, real cost

**Context:** implementing reproduction (design worked out in conversation,
not restated here) required a structural change first: `Environment` only
ever tracked one fly. A colony needs many flies alive at once, sharing the
same spiders/food/director pressure — not N separate single-fly worlds.

**Decision — multi-fly `Environment`:** `self.fly_pos`/`self.hunger`
replaced with `self.flies: list[Fly]` (new DTO in `entities.py`: id,
position, hunger, `vulnerable_ticks_left`). `reset()` now takes no
per-call args (population comes from the `initial_population` constructor
kwarg) and returns `dict[fly_id, Observation]`; `step()` takes
`dict[fly_id, Action]` (one per living fly, raises if any is missing) and
returns a new `ColonyStepResult` DTO (`observations`, `deaths: dict[id,
cause]`, `births: dict[new_id, parent_id]`, `colony_extinct`,
`timed_out`). `sense_nearest`/`observe` now take an explicit fly instead
of reading `self.fly_pos` implicitly.

**Why this doesn't break the existing ES curriculum training:** a
population of 1 makes reproduction structurally impossible — see the
formula below, `mate_availability` is exactly 0 at population 1 — so
`training/trainer.py`'s `rollout()` (updated to the dict-based API, still
always exactly one fly per tick) behaves identically in spirit to before.
No separate single-fly code path needed; same "one parameterized class,
different configs" discipline as everything else here.

**Decision — the reproduction mechanic**, exactly as designed in
conversation: not an action the trained circuit decides (no new `Action`
member; parallel to how food pickup and spider spawning already aren't
agent decisions either) — a world-level stochastic event, checked every
tick per living fly:

```
if hunger_fraction > reproduction_hunger_threshold and population < max_population:
    mate_availability = min(max_mate_availability, (population - 1) / (max_population - 1))
    probability = base_reproduction_rate * mate_availability
    # roll probability; on success:
    #   parent.hunger *= (1 - reproduction_hunger_cost_fraction)
    #   parent.vulnerable_ticks_left = reproduction_vulnerability_ticks  (forces STAY, overriding any requested action)
    #   offspring spawns near parent, hunger = max
```

Selection is purely differential, not reward-shaped: a better escape
circuit survives longer, which means more per-tick reproduction rolls
over its lifetime, which means more offspring on average — no fitness
term added to the reproduction probability itself (would be redundant
with, and would distort, that already-existing signal). `population`
used in the formula is colony-wide, not spatial proximity — a deliberate,
cheaper stand-in for "a mate needs to be nearby" that a reviewer flagged
as the harder version to defer, not build now. `max_mate_availability`
(< 1) guarantees the probability can never reach certainty even at full
capacity — added specifically so a kill near the population cap doesn't
create a near-guaranteed immediate replacement, which would make kills
near capacity feel like they don't matter.

**Verified:** every mechanic tested in isolation directly against
`Environment` (not just "should work") — population-1 reproduction is
provably impossible even at `base_reproduction_rate=1.0`; reproduction
fires and respects the `max_population` cap under direct test; parent
hunger cost and `vulnerable_ticks_left` are applied and actually force
`STAY` the following tick even when a different action is requested;
per-fly threat/starvation death still work (one test initially used a
non-moving threat against a stationary fly and — as expected from
`decisions.md` #12/#15 — never killed it, which is correct behavior, not
a bug, and was fixed by moving the threat, not the code). Full ES
training re-run end to end: real nonzero `population_reward_std` every
iteration preserved, fitness still climbs to the episode ceiling — the
baseline shifted slightly (78.5 vs. the earlier 73.3) from the RNG draw
sequence changing under the refactor, not from any behavior change.

## 21. `training/colony.py`: real per-fly circuits/genomes wired to reproduction

**Context:** #20 made the reproduction mechanic real at the `Environment`
level, but flies had no actual brains — nothing decided their actions.

**Decision:** first, a prerequisite refactor to `fly_brain/agent.py` —
building a circuit (loading the connectome, expanding it, finding
seed/motor neurons) is expensive and identical for every fly of the same
`seed_type`; doing it once and stamping out many cheap instances needed
splitting that from `EscapeAgent` itself. `EscapeCircuitTemplate` (+
`build_escape_template()`) now holds the expensive, shared part;
`EscapeAgent(template)` is cheap — just a fresh `Circuit` off the shared
blueprint. `EscapeAgent`'s existing `set_params()` was already enough to
give an instance its own genome; no constructor argument needed for that.

`training/colony.py`'s `Colony` class owns `dict[fly_id, EscapeAgent]`
alongside the `Environment`. Each tick: every living fly acts via its own
agent, `Environment.step()` resolves the world, then for each birth the
parent's `get_params()` plus Gaussian noise (`mutation_sigma`, same idea
as ES's `sigma`) becomes the offspring's genome via a fresh agent's
`set_params()`; each death just deletes that fly's agent. Starting genome
(`load_starting_gains`) comes from a trained ES checkpoint when one's
given and exists, falling back to untrained (gain=1.0) otherwise — the
"ES pretrains a starting gene pool, reproduction continues evolving it
live" split from `decisions.md` #13, now actually wired up rather than
just described. A small CLI (`training/colony_run.py`) runs a colony
headlessly and logs population/births/deaths — the cheap "watch it work"
checkpoint we agreed on before touching the frontend, not the real game
loop (no `director/` involved yet).

**Verified:** unit-tested directly against `Colony` — birth creates a new
agent with genuinely mutated (not identical) params and keeps
`colony.agents` in sync with living flies; death removes exactly that
fly's agent, confirmed with real threat-caused deaths. Ran the actual CLI
twice: once at `Environment`'s plain defaults (extinct by tick 115, pure
starvation — expected, not a bug: `EscapeAgent`'s circuit only ever
reacts to `threat_signal`, it has no foraging behavior at all yet, so a
passive fly only survives on food it happens to wander into by luck), and
once with more generous food settings, where a real, naturally-triggered
birth (population 3→4 at tick 127) was observed before the colony still
eventually went extinct through a mix of starvation and a threat kill —
a full lifecycle, driven by a real trained circuit, no artificial forcing.
