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

## 22. Sensing/learning redesign: anonymous percepts, real mushroom-body circuit, encoder-based item authoring

**Context:** the current `Observation` (`food_signal/dx/dy, threat_signal/dx/dy,
hunger`) hardcodes exactly two object types by name. The goal stated for this
phase is AGI-like (not full AGI): a fly should discover and react to new item
types by how they physically present, never by a hardcoded type/name, with
valence (good/bad) learned entirely from lived experience within a single
fly's life — not something evolution hands it pre-solved. A generalized
`dict[str, SensorReading]` version was considered and rejected: a dict key is
still a name, so it's the same hardcoding with different syntax.

**Decision, in four parts:**

1. **Perception boundary — anonymous `Percept` list.** `Observation.nearby`
   becomes a variable-length list of `Percept(attributes: vector, dx, dy,
   distance)` — no name, type, or id ever crosses from `world/` into what the
   fly perceives. `world/`'s own internals (`Food`, `Threat`, a future `Web`)
   stay typed, same as today — typing is fine on the world's side of the
   boundary, never on the fly's.

2. **A second, real, connectome-grounded circuit for lifetime learning.**
   Kept fully separate from the frozen, ES-only escape circuit (`fly_brain`'s
   existing `EscapeAgent`/`DNp01`/`TTMn`), combined via the same
   freeze+override pattern as #5 (escape always wins when `TTMn` spikes).
   Built the same way the escape circuit is (`build_circuit()`, real topology,
   real synapse sign, trainable `synaptic_gain`), seeded from real Kenyon
   Cell (KC) / Mushroom Body Output Neuron (MBON) / Dopaminergic Neuron (DAN)
   types.

   **Verified empirically before committing to this** (the standing
   discipline from #14/#15 — check the real data, don't assume): the MaleCNS
   annotations contain 4,064 real KCs (15 subtypes), 97 real MBONs (37
   types), and 358 real DANs, which split by type into `PAM`/`PPL`/`PPM`
   clusters — a real reward-coding/punishment-coding split already present in
   the data, not something assigned. Real connectivity is dense: 61,210
   `KC→MBON` synapses, 129,137 `DAN→KC` synapses. The substrate this design
   needs actually exists and is well-connected.

   Mechanism: a percept's attribute vector is injected as stimulus current
   into the KC population via one fixed (untrained) random projection matrix
   `W` — standing in for the real ~6-random-PN-inputs-per-KC wiring, the same
   simplification already used for direct stimulus injection at `DNp01`.
   Multiple simultaneous nearby percepts sum into the same population, so
   sparse combinatorial KC coding gives any attribute combination — including
   one never seen before — its own largely distinct activity pattern, with no
   training or lookup table needed for that separation. This is the actual
   discovery mechanism: novel objects get a novel-but-similarity-overlapping
   code automatically, as a byproduct of how real KCs work.

   Reinforcement ties to real outcomes already in `world/env.py` (food
   pickup, damage/death) by injecting current into `PAM`/`PPL` DANs. A local
   three-factor Hebbian rule (`Δsynaptic_gain(KC_i→MBON_j) = -η ·
   dopamine_signal · kc_i_activity_trace`) updates `KC→MBON` gains per tick —
   this is the real mushroom-body depression mechanism, and it's what makes
   learning happen within one fly's life rather than only across generations.
   MBON output (approach-coding spikes minus avoid-coding spikes) becomes a
   movement bias outside the circuit, same externalized-motor-decision
   pattern as flee-direction.

   **What `dopamine_signal` concretely is:** not an abstract number injected
   directly into the update rule — the real spiking output of the real DAN
   population, same discipline as everything else here. Reinforcement events
   inject stimulus current into `PAM`/`PPL` DANs exactly the way
   `threat_signal` is injected into `DNp01` in the escape circuit; those DAN
   neurons then spike or don't through the same LIF dynamics as every other
   neuron in the circuit, and `dopamine_signal` is that real spiking
   activity — never a hardcoded ±1. The reward-vs-punishment sign falls out
   of which real DAN type actually fired (`PAM` = reward-coding, `PPL1` =
   punishment-coding in the literature, and this split is the real one found
   in the data, not assigned by us). Because a fly senses an object slightly
   before it learns the outcome (approach, then eat), the dopamine spike
   generally arrives a few ticks after the KCs that represented that
   object's attributes fired — `kc_i_activity_trace` is a short decaying
   eligibility trace of recent KC activity that exists specifically to let a
   slightly-delayed dopamine signal still tag the right KCs, rather than
   requiring exact same-tick coincidence.

3. **Evolution's role changes, doesn't disappear.** ES no longer needs to
   evolve the answer (that's learned live); it evolves the *prior* —
   starting `synaptic_gain` values and/or the plasticity rule's own
   hyperparameters (learning rate, trace decay). `training/colony.py`'s
   offspring-genome step must copy only that inherited prior, never whatever
   gains a parent's synapses drifted to during its life — learned
   associations aren't inherited, same as real biology; only the capacity to
   learn them is.

4. **Item authoring: local small text encoder, not hand-written vectors.**
   Considered three options: (a) hand-write each item's attribute vector
   directly; (b) hand-write a small vocabulary of base-category + modifier
   vectors and compose new items additively (word2vec-style arithmetic,
   guaranteed-linear by construction, zero training); (c) a real pretrained
   text-embedding model, frozen (no training), encoding a short authored
   description (`"a smelly piece of raw meat"`) into a vector.

   **Chosen: (c).** Rationale given: (a) doesn't scale — full manual
   authoring per item; (b) scales better but the linear structure is
   artificial/engineered rather than emergent. (c) was picked specifically
   for scalability, on the explicit understanding that "same encoder for
   everything" gives *approximately* linear analogical structure (`king -
   man + woman ≈ queen`), not a guaranteed algebraic property — that
   behavior is an emergent, empirically-observed feature of embeddings
   trained on large corpora, not something a shared encoder logically
   guarantees. Expectation is set accordingly: item vectors will cluster and
   compose sensibly, not exactly.

   The encoder is a small **local** model (not a hosted API) — no network
   dependency, no per-call cost, deterministic across runs, and it never
   trains, only encodes, keeping it in the same "off-the-shelf, fully
   inspectable tool" category as everything else in the project.

   **Dimensionality reduction: Matryoshka truncation, not PCA.** Both were
   considered. PCA (fit once on a compact model's output, e.g.
   `all-MiniLM-L6-v2`) was rejected once a concrete problem was worked
   through: items get added mid-game (e.g. a new "smelly apple" variant),
   and PCA's axes are only as good as the vocabulary they were fit on — an
   item varying along a direction the original fit saw little/no variance in
   gets compressed away, silently under-differentiated. Refitting PCA later
   to fix that is worse, not better: it shifts the axes, which shifts every
   *existing* item's vector too, including ones a live fly already has
   learned `KC→MBON` associations against — a mid-game refit would silently
   corrupt what a fly has already learned.

   Chosen instead: a Matryoshka Representation Learning model
   (`nomic-embed-text-v1.5`, confirmed to natively support truncation to any
   size from 64–768 dims), truncated to a small fixed prefix (e.g. 64 dims).
   Truncation is a fixed, vocabulary-independent operation — "take the first
   N dimensions" is defined identically for item 1 and item 10,000, forever,
   with no fitting step and therefore nothing that can go stale or need
   retroactive correction as new items get added.

   Reduction itself is not mathematically required (the fixed KC projection
   `W` works at any input size) — it's still done because most of a
   general-purpose encoder's dimensions are irrelevant noise for this narrow
   domain, because it keeps the injected dimensionality in the same rough
   order of magnitude as the real PN population KCs actually sample from,
   and because a small vector is something a human can actually read and
   reason about. Exact final truncation size is an implementation detail to
   tune, not fixed here. Per-instance jitter is still applied numerically,
   after encoding/truncation, exactly as already designed — never by varying
   the text.

   This authoring pipeline (description → encoder → truncation → jitter)
   lives entirely in `world/`'s content-authoring layer. It never crosses
   into `Observation` — the fly only ever receives the resulting numeric
   `Percept`, never text, never a type name.

5. **Reinforcement mechanism: a Result registry, real state channels, reward
   derived from real deltas only.** Extends part 2's "reinforcement ties to
   real outcomes" with a general mechanism instead of a two-outcome special
   case (food pickup / death).

   New real state on `Fly`: `health` and `stuck_ticks`, alongside the
   existing `hunger`. Death gains a new cause (`health <= 0`), alongside the
   existing `'starved'`/`'threat'`. Motivation: instant death gives the
   lifetime plasticity circuit nothing to learn from — a fly that dies on
   first contact with something bad never survives to express whatever its
   synapses just updated to. Graded damage lets a fly survive a bad
   encounter, receive the punishment signal, and visibly behave differently
   afterward — the only way learned avoidance is observable within one
   lifetime.

   A small, fixed **Result registry** (`food`, `damage`, `immobilize`,
   extensible later — e.g. a `stick`/web result down the roadmap): each
   encoded the same way items are (short description → the same frozen
   encoder + truncation from part 4), each mapped to one real physiological
   channel (`food`→`hunger`, `damage`→`health`, `immobilize`→`stuck_ticks`).
   On interaction, `weight = max(0, cosine_similarity(item_vector,
   concept_vector))` per Result, applied simultaneously (not argmax/hard
   pick) — this is what makes an item close to both `food` and `damage`
   (poisonous meat, Minecraft-rotten-flesh-style) both heal and hurt at
   once, and what makes a stronger food-resemblance produce a bigger
   `Δhunger` than a weak one, with no per-item authored number needed.

   **The dopamine/reward signal is derived from the real resulting
   `Δhealth`/`Δhunger`/`Δstuck_ticks`, never directly from the item-Result
   similarity score.** This is the one rule that keeps this consistent with
   part 2: computing reward straight from item-to-concept similarity would
   quietly reintroduce object-level valence through a back door (reward
   depending on what the item semantically resembles, rather than on what
   actually happened to the fly). Routing it through the real state change
   keeps the fly reinforced only by lived consequences.

   Authoring `damage is bad, food is good` at the **state/channel** level is
   not a violation of "no assigned object valence" — it's the same category
   as real biology hardwiring pain/satiation as innate drives, not learned.
   Items themselves stay completely unlabeled and valence-free until
   experienced; only this small, fixed Result universe carries authored
   sign and weight. One new small authored quantity follows from this: a
   per-channel weight to combine `Δhunger`/`Δhealth`/`Δstuck_ticks`
   (different units/scales) into one dopamine magnitude — a handful of
   tunable numbers at the channel level, not per item.

   **Distinct from ES's fitness.** Checked `training/trainer.py`: ES's
   current reward is purely `ticks_survived` (mean over
   `episodes_per_eval` rollouts, standardized across the population each
   iteration) — no item/food/damage term exists there at all, and stage 1
   even runs with `food_enabled=False`. This Result-registry reward feeds
   only the lifetime dopamine/plasticity circuit. Whether ES's fitness
   should later also incorporate lifetime-accumulated reward is a separate,
   open question — not decided here.

**Status:** design only, empirically grounded (KC/MBON/DAN existence and
connectivity verified against the real data above) but **nothing in this
entry is implemented yet** — no `Percept`, no second circuit, no plasticity
rule, no encoder pipeline. Substantial new work when built: new data
structures in `world/`, a new circuit-builder path + plasticity loop in
`fly_brain/`, a new local-encoder dependency, and integration changes to
`training/colony.py`.

## 23. Live game loop: `training/live_run.py`

**Context:** `director/` (the swappable `WorldController` + action
registry) and `training/colony.py` (a real per-fly-circuit, reproducing
colony) both existed and were independently tested, but nothing connected
them into a continuously-running session a player could actually address
requests to. This was the last item blocking the frontend
(`decisions.md` #19's sequencing condition).

**Decision:** a background thread reads player requests from stdin into a
`queue.Queue`; the main thread ticks the `Colony` at a fixed real-time rate
(`--ticks-per-second`, default 5) and, each tick, drains whatever requests
arrived since the last tick and applies them via the existing
`WorldController`/registry, before advancing the colony. This means typing
a request never pauses the world — the two happen on separate threads,
with the queue as the only handoff between them; the background thread
never touches `Environment`/`Colony` state directly, keeping all world
mutation on the main thread.

`Environment.max_ticks` is set to an effectively unbounded value
(`10**9`) for a live session — unlike training rollouts or the headless
`colony_run.py` CLI, a live session isn't meant to time out; it ends only
on colony extinction or the player typing `quit`/`exit` (or closing
stdin).

The request-handling logic is split from I/O on purpose, the same
discipline used everywhere else in this project (e.g. `trainer.py`'s
`rollout`/`evaluate` vs `run.py`'s CLI wrapper): `apply_requests()` and
`drain()` are pure functions with no I/O, so they're directly testable
without a real terminal or thread; `read_requests()` (the blocking
`input()` loop) and `main()` are the only parts that touch actual stdin/
threading.

**Verified:** `apply_requests()`/`drain()` tested directly against a real
`Environment` + `RuleBasedController` — confirmed a matched request
mutates `spider_spawn_rate` and an unmatched one doesn't, confirmed queue
draining stops at the `None` sentinel. Ran the full CLI end-to-end twice
with piped stdin: once with all input available immediately (confirms
requests apply correctly and `quit` stops the loop), and once with input
staggered by real `sleep` calls (confirms the colony keeps ticking
continuously in real time between requests — 39 ticks over ~6s wall time
at `--ticks-per-second 10` — rather than only advancing when input
arrives). `--controller claude` reuses the existing, still
not-live-tested `ClaudeController` (`decisions.md` #16) — no credentials
available in this environment.

## 24. #22 implementation, phase 1: `world/items.py`, swappable item encoder

**Context:** starting implementation of #22. The first, self-contained
piece is the item-encoding pipeline (description → encoder → truncation →
jitter). Before writing it, checked whether `nomic-embed-text-v1.5`
(chosen in #22) is actually reachable from this dev environment, the same
diligence already applied to the connectome data and to `ClaudeController`.

**Found:** `huggingface.co` is policy-blocked in this sandbox — the proxy
status endpoint shows an explicit `403`/`"policy denial"` on
`huggingface.co:443`. Per this session's proxy rules, blocked hosts are
reported, not routed around. So the real encoder's weights cannot be
fetched or exercised here, in this environment specifically — not a
problem with the design.

**Decision:** `ItemEncoder` (`world/items.py`) is an abstract swap point,
same pattern as `director/`'s `WorldController`:
- `NomicItemEncoder` — the real backend, `nomic-embed-text-v1.5` via
  `sentence-transformers`, truncated to a small `output_dim` using its
  native Matryoshka support, exactly as decided in #22.
  **Not exercised live here** — `huggingface.co` unreachable in this
  sandbox — the same situation `ClaudeController` is already in with no
  API credentials (#16). Needs testing on a machine that can reach HF
  before relying on it.
- `HashingItemEncoder` — a zero-dependency stub (classic feature-hashing
  over character trigrams), used to build and test everything downstream
  without a network dependency. Explicitly **not** a semantic model —
  similarity comes from shared substrings, not meaning — documented as
  such in its docstring so it's never mistaken for the real thing.

`encode_with_jitter()` adds per-instance Gaussian noise to a prototype
vector and re-normalizes to unit norm, so every downstream consumer
(Result-registry blending, KC injection) can use plain cosine similarity
(dot product) throughout, per #22.

**Verified:** ran `HashingItemEncoder` directly — deterministic (same
description → identical vector across calls); descriptions sharing
substrings cluster (`"a crisp red apple"` vs `"a smelly red apple"`:
cosine 0.48) more than unrelated ones (`"a crisp red apple"` vs `"a
smelly piece of raw meat"`: cosine −0.13); two independently jittered
instances of the same description stay close to each other (0.85) and to
the un-jittered prototype (0.94) rather than drifting arbitrarily; all
vectors confirmed unit-norm after jitter.

`sentence-transformers` added to `requirements.txt`, noted as needing
`huggingface.co` access.

**Status:** phase 1 of #22 done and tested (stub path only). Remaining
phases: anonymous `Percept`/`Observation` + `Fly` health/`stuck_ticks`
state + Result registry (world/); the KC/MBON/DAN circuit + dopamine-gated
plasticity rule (fly_brain/); resolving how the escape circuit gets its
stimulus without a named `threat_signal` field (open design point, not
yet resolved — see discussion in the same session); `training/colony.py`
integration.

## 25. #22 implementation, phase 2: anonymous `Percept`s, Result registry, `Fly` health/`stuck_ticks`, escape circuit resolved

**Context:** continuing #22's implementation. This phase replaces
`Observation`'s fixed named fields with the anonymous `Percept` design
(#22 part 1), builds the Result registry (#22 part 5), adds the
`health`/`stuck_ticks` state it depends on, and resolves the one open
design gap from #24: the escape circuit read `obs.threat_signal`
directly, a field that no longer exists once `Observation` is anonymous.

**`Observation` redesign** (`world/env.py`): `food_signal/dx/dy,
threat_signal/dx/dy` replaced by `nearby: list[Percept]`, where
`Percept(attributes, dx, dy, distance)` carries only a unit-norm item
vector and relative position — no name, type, or id. `SensorReading`/
`sense_nearest`/`Observation.as_array()` removed (grepped the whole repo
first; nothing else referenced them). `Environment.perceive()` still
uses each entity's own radius (`food_radius`/`threat_radius`) as a
world-internal visibility cutoff — that stays type-specific on the
world's side of the boundary, per #22's own rule; only the returned
`Percept` is anonymous.

**Result registry** (`world/results.py`, `ResultConcept` +
`blend_deltas()` + `build_default_results()`): `food`/`damage`/
`immobilize`, each encoded via the same `ItemEncoder` as items,
mapped to `hunger`/`health`/`stuck_ticks`. `Environment.apply_result()`
replaces the old fixed `fly.hunger = self.max_hunger` on pickup with the
clipped-cosine-similarity blend across all three, applied simultaneously
and clamped to each channel's bounds.

**New `Fly` state**: `health: int` (starts at `max_health`) and
`stuck_ticks: int` (starts at 0). `move_flies()` forces `STAY` and
decrements `stuck_ticks` while it's positive, same pattern as the
existing `vulnerable_ticks_left` reproduction-cooldown mechanic.
`determine_fly_death()` gains a `"damage"` cause (`health <= 0`),
alongside the existing `"threat"`/`"starved"`.

**Escape circuit resolution** (`fly_brain/agent.py`, resolving #24's open
gap, confirmed before implementing): `EscapeCircuitTemplate` gains a
`danger_vector` (`encoder.encode("a dangerous, fast predator")`, computed
once, same encoder as everything else). `sense_danger()` replaces the
direct `obs.threat_signal` read: cosine similarity between each nearby
percept and `danger_vector`, clipped at 0, weighted by `1/(1+distance)`,
**summed across all matching percepts** (not just the nearest) so
multiple simultaneous threats never look less urgent than one — then
flee direction points away from the single nearest above-threshold
percept, since fleeing needs one concrete direction even though the
spike-triggering current is a sum. `build_escape_template()` now takes
an `ItemEncoder` argument; every call site (`training/run.py`,
`colony_run.py`, `live_run.py`) constructs one `HashingItemEncoder` and
passes it to both `Environment` and `build_escape_template()` so both
sides read attribute vectors in the same space.

**Verified:**
- `Environment` unit-level: a food-matching item on a full-hunger fly is
  a no-op (correctly clamped); a "nourishing toxic dangerous poison
  meat"-flavored item heals *and* damages the same fly in one call
  (50→63 hunger, 100→41 health) — the Minecraft-style dual effect from
  the original request; an unrelated item's cross-similarity is small
  but non-zero under the crude hashing stub (expected — it's explicitly
  orthographic, not semantic, see #24).
- A web-flavored item sets `stuck_ticks`, and a fly with `stuck_ticks >
  0` is confirmed forced to `STAY` even when a different action is
  requested, decrementing correctly.
- `health <= 0` confirmed to produce `"damage"` as the death cause.
- **Retrained stage 1 from scratch** (`python -m training.run --stage 1`,
  8 iterations, smoke-sized) against the new `sense_danger()` stimulus:
  real `population_reward_std` every iteration (same discipline as #15),
  fitness baseline 65.3 → 80.0 (max survival). The old checkpoint's gains
  were tuned against the old signal's statistics and weren't expected to
  transfer cleanly to the new one's different shape — retraining
  confirms the new stimulus is real and trainable, not just
  non-crashing. (Checkpoints are gitignored; the retrained one isn't
  tracked.)
- `training/colony_run.py` end to end: reproduction still fires (birth
  observed), deaths still occur. Traced `Environment.apply_result()`
  directly in a live `Colony` run: with threats present (so the fly
  actually moves via real flee reactions) and food enabled, a real
  pickup fired mid-run and visibly changed `hunger`/`stuck_ticks`. With
  threats disabled, zero pickups occurred in 300 ticks — not a bug, the
  known "no foraging behavior, only reacts to danger" limitation
  (`decisions.md` #21) means a fly with nothing to flee from never moves
  at all yet.
- `training/live_run.py` re-run end to end after these changes — still
  applies requests and quits cleanly.

**Status:** phases 1–2 of #22 done and tested. Remaining: the KC/MBON/
DAN circuit + dopamine-gated plasticity rule (phase 3); `training/
colony.py` integration, including the prior-vs-live gain split on
reproduction (phase 4).

## 26. #22 implementation, phases 3–4: the KC/MBON/DAN plasticity circuit, `Colony` integration, auditability

**Context:** the last, biggest piece of #22 — a second real circuit
implementing lifetime, dopamine-gated learning, wired into `Colony`
alongside the frozen escape circuit. The user asked explicitly to be
able to audit and see learning/improvement over a fly's life, so
observability was a first-class requirement, not an afterthought.

**Circuit construction** (`fly_brain/plasticity.py`,
`build_plasticity_blueprint`/`build_plasticity_template`): the real
populations (4,064 KCs / 97 MBONs / 354 DANs, verified in #22) are far
too large to simulate per-tick, per-fly across a live colony, so the
circuit is built from a bounded, real subgraph rather than the full
population — and rather than `fly_brain/circuit.py`'s existing
`build_circuit()`, which expands outward hop-by-hop under a *global*
edge budget (built for a different problem: discovering a path from one
seed type to a motor neuron). That approach was tried first and
produced 4,618 neurons (DANs broadcast far too broadly in the real data
to cap globally) — a new, purpose-built extraction was needed instead:
top-100 real KCs by total real KC→MBON weight (not an arbitrary
subsample — the KCs whose real output matters most for this circuit),
their real top-20 targets each (auto-discovering the MBON population,
50 neurons, rather than guessing it), then every DAN with a real edge
into that KC/MBON universe, each capped to its own top-30 edges
*restricted to that universe* (not DAN's globally-broadest targets,
which mostly point elsewhere). Result: 481 real neurons, 6,442 real
edges, 1,000 plastic KC→MBON synapses, 300 PAM + 31 PPL/PPM DANs — a
real, well-connected, computationally tractable subgraph.

`Circuit` gained one small public method, `pre_neuron_of_edge()` — the
plasticity rule needs real per-edge structural identity (which KC feeds
which edge) to apply a local update, unlike ES's uniform whole-vector
perturbation, which never needed to know. Everything else about edge
structure stays private, per `Circuit`'s existing design.

**MBON approach/avoid split is an explicit, documented simplification**
(`split_mbon_valence`): there's no real downstream-of-MBON connectivity
in this bounded subgraph to derive functional roles from data, and
mapping the dataset's numeric MBON type codes to published literature
roles is out of scope for verification here. The split is structural
(sorted body id parity), not a biological claim.

**Two real bugs found and fixed empirically, not assumed away** — both
matter for the design's actual correctness, not just code hygiene:

1. **The textbook depression-only Hebbian rule doesn't work with a
   direct-spike-readout MBON.** The real mushroom body rule depresses
   whichever KC→MBON synapses were just active, full stop — that shifts
   real fly behavior because of real anatomy *downstream* of MBON that
   isn't part of this bounded subgraph. Applied uniformly here (where
   MBON output is read directly as approach-minus-avoid), it depresses
   both pathways equally for the same active KCs and never moves the
   difference — confirmed directly: net valence stayed frozen even as
   `gain_drift` climbed steadily. Fixed by making the rule direction-
   aware: reward potentiates the approach pathway and depresses avoid
   for the active KCs; punishment does the reverse. This is a deliberate
   adaptation to the simplified readout, not the textbook rule, and it's
   documented as such in `PlasticityAgent.reinforce()`.

2. **Uniform current injected into a DAN population saturates instead of
   grading with magnitude.** Injecting the same current into every
   neuron in a group makes that group's response an all-or-nothing
   switch — any reward/punishment above a tiny floor makes the *entire*
   group spike, so `pam_signal - ppl_signal` collapses to ~0 regardless
   of true magnitude. Confirmed directly in a live `Colony` run: a real
   +47 hunger pickup produced **zero** reinforcement events. Fixed with
   a fixed, per-neuron random gain (`pam_gain`/`ppl_gain`, `Uniform(0.01,
   1.0)`, drawn once at template-build time) — turns the population into
   a genuine graded rate code (verified: PAM spike fraction 0.0 → 0.81 →
   0.97 → 1.0 as reward goes 1 → 5 → 20 → 100, a smooth curve instead of
   a step function).

**Auditability** (the explicit ask): `PlasticityAgent.probe(attributes)`
is a read-only diagnostic — net valence (approach − avoid **membrane
potential**, not spikes; see below) this agent's *current* gains would
produce for a given vector, with zero side effects (a disposable circuit
copy is stepped, the live agent's own state is never touched). Call it
with the same vector at intervals through a fly's life to see a learning
curve. `gain_drift()` is a cheap scalar summary (L2 distance between
current and inherited-prior KC→MBON gains). `Colony.plasticity_summary()`
gives a population-level view (mean gain drift, total reinforcement
events); `run_colony()`/`run_live()` now log it periodically (every 50
ticks) rather than only at the end, since a colony that goes extinct
would otherwise lose the whole record the moment its learning flies die.

**A third empirical correction, found while building the audit
tooling:** `probe()`'s readout was originally spike-count-based
(approach spikes − avoid spikes), matching the plain English of #22
part 2 ("MBON output ... spikes"). Directly comparing it against a
membrane-potential-based readout on the same reinforcement sequence
showed the spike-count version frozen at exactly 0.0 for ten consecutive
exposures while `gain_drift` climbed the whole time — a binary spike
count is too coarse to reflect gradual synaptic change tick to tick.
The membrane-potential version tracked the same learning smoothly (0.29
→ 3.82). Switched both the audit probe and the live behavioral readout
(`_mbon_valence`) to membrane potential; the circuit's own internal
spiking dynamics (KC, DAN, and MBON's own spike-or-not state carried
into the next tick) are unchanged, only what gets *read out* differs.
Added a small deadzone (`VALENCE_DEADZONE = 0.1`) so continuous-valued
noise near zero doesn't cause constant twitchy movement, since
`valence == 0.0` (safe for the old binary readout) almost never holds
for a real-valued one.

**`Colony` integration** (`training/colony.py`): each fly now has both
an `EscapeAgent` and a `PlasticityAgent`. Each tick: the plasticity
circuit senses and decides *every* tick regardless of outcome (its
eligibility trace has to keep running even on ticks the fly actually
fled instead); the escape circuit's `decide()` (added to `EscapeAgent`
alongside the existing `act()`) returns `None` when `TTMn` doesn't spike
rather than defaulting to `STAY`, so `Colony` can distinguish "no escape
override" from "chose to stay" and fall through to the plasticity
circuit's own decision — the override rule from #5/#22, now concretely
wired. Real state is snapshotted before `env.step()` and diffed after
for every survivor, and that real `Δhunger/Δhealth/Δstuck_ticks` (never
a percept similarity score) is what `reinforce()` is called with —
`Environment` itself has no idea `fly_brain` exists; `Colony` computes
the diff itself rather than `Environment` exposing a reward-shaped API.
On birth, escape genome inherits as before (#21); plasticity genome
inherits from `parent.prior_gains` (what the parent was *born* with)
plus mutation — never `parent.get_params()` (what the parent's synapses
drifted to during its own life), per #22 part 3. Verified directly:
artificially drifted a parent's live gains away from its prior, then
confirmed a spawned offspring's gains matched the prior exactly and
differed from the drifted live gains.

**Verified, end to end:**
- Isolated reward learning curve (repeated `+47 hunger` exposures to the
  same item): valence 0.29 → 3.82 → plateau, `gain_drift` climbing
  steadily, `reinforcement_events` incrementing correctly.
- Isolated punishment learning curve (repeated `-20 health` exposures):
  valence 1.02 → -3.15, correctly trending negative (avoidance).
  Slower to move than reward — expected, the PPL/PPM population (31
  neurons) is much smaller than PAM (300), so its mean needs more
  consistent activation to shift by the same amount.
- Generalization: reinforcing only "apple" produced a smaller positive
  shift in a never-reinforced, wording-similar "pear" than in apple
  itself, and a much smaller shift than nothing at all would — a real
  transfer gradient, though noisier than a real semantic encoder would
  give (the stub is explicitly orthographic, not semantic, #24).
- Full live `Colony` run (`training/colony_run.py`): traced a real food
  pickup mid-run, confirmed `reinforce()` fired with the correct deltas
  and `gain_drift`/`reinforcement_events` updated on the correct fly's
  agent immediately.
- Offspring inheritance test (above).
- Regression: ES training (`training/run.py`) and `training/live_run.py`
  both re-verified end to end after the `EscapeAgent.decide()`/`act()`
  refactor — unaffected.

**Not built in this pass, left open:** an ES-style (or other)
evolutionary process specifically for the plasticity circuit's *prior*
gains/hyperparameters (#22 part 3 says "and/or" — it's currently just
untrained, gain=1.0, same fallback as the escape circuit); a dedicated
audit CLI/visualization beyond the periodic log lines (the hooks
`probe()`/`gain_drift()`/`plasticity_summary()` are there for one to be
built on top of).

## 27. Player-driven item creation: a generic `Item`, a parameterized director action

**Context:** the original vision stated after #22 was completed —
"an open World where the user can add items... by describing them to an
LLM" — needed two things that didn't exist yet: a director action that
takes a real argument (the v1 contract, #14, was zero-argument-only —
anticipated but not built, per `claude_controller.py`'s original
docstring), and a way for `Environment` to spawn item types beyond the
two fixed slots (`food`, `threat`). Explicit instruction: keep it clean
and simple.

**Decision: one generic `Item`, not a configurable entity system.** A
player-created item is stationary, single-use (consumed on contact,
same as `Food`), and its entire behavior — what a fly perceives, what
happens on contact — comes from its attribute vector via the exact same
Percept/Result-registry mechanism `Food` already uses. Nothing about it
is separately configurable (no movement, no custom radius, no instant-
kill special case): a "dangerous" description just means high similarity
to the `damage` Result concept, which already produces real harm through
the existing health system (#25/#26) — no new mechanic needed. `Threat`
was deliberately left untouched (still moves, still instant-kills on
contact) — unifying it with the new generic `Item` would have been a
bigger, riskier change to already-tested code for no benefit `Item`'s
use case actually needs.

**`world/items.py`** gained `ItemType` (a registered description + its
unjittered prototype vector — the type-level record) and a standalone
`jitter()` function extracted from `encode_with_jitter()` (which now
just calls it). This let `Environment` cache the `food`/`threat`
prototypes *once* at construction instead of re-encoding the same fixed
description from scratch on every single spawn — a real inefficiency
that existed since #25 and would have mattered a lot once a real
(non-stub) encoder is in use; player-created item types get the same
treatment for free. **`world/entities.py`** gained `Item` (position,
interaction radius, attributes) — a live spawned instance, parallel to
`Food`/`Threat`.

**`Environment.add_item_type(description)`** is the actual entry point:
encodes and registers a new `ItemType`, capped at `MAX_ITEM_TYPES = 20`
(silently ignored past the cap — simple, no error-reporting channel back
to the player exists to make anything fancier worthwhile). Spawning,
sensing, and pickup all reuse existing generic machinery unchanged
(`perceive()`, `apply_result()`) — extended in three small, symmetric
places: `spawn_entities()` (one shared `item_spawn_rate`/`max_items`
across *all* player-created types, not per-type — deliberately simple:
more variety naturally means more stuff spawning overall, no per-type
tuning surface), `observe()`, and a new `resolve_item_pickup()`
paralleling `resolve_food_pickup()`.

**`director/`'s v1 contract is extended, not replaced.** `ActionSpec`
gained `takes_argument: bool = False`; the three existing zero-argument
actions are unaffected. `create_item` is the fifth action, wrapping
`env.add_item_type` directly. `WorldController.choose_action()`'s return
type changed from `str | None` to `tuple[str, str | None] | None` (name,
argument) — a clean break across both controllers, no backward-compat
shim, consistent with this project's stated anti-pattern list.
`RuleBasedController` handles it the same crude way as everything else
it does: a fixed trigger phrase (`"create item:"` / `"add item:"` /
`"new item:"`), everything after it taken verbatim as the description —
no attempt at real language understanding, matching its own documented
role. `ClaudeController` gets a real `input_schema` for `create_item`
(`{"description": "string"}`, `required`) instead of an empty one, and
extracts `block.input["description"]` from the tool-use response — the
first action to actually exercise the parameterized-tool-use path the
controller was built to support from the start.

**Verified:**
- `Environment`-level: `add_item_type()` registers correctly, spawning
  respects `max_items`, a forced adjacent item is sensed as a `Percept`
  and, on pickup, produces real `Δhealth` via the Result registry (a
  "smelly, poisonous meat" item dropped a fly's health 66→31) — same
  mechanism already verified for `Food`, now confirmed for a
  player-created type too. `MAX_ITEM_TYPES` cap confirmed at 20 after
  registering 25.
- `RuleBasedController` correctly extracts the description from
  `"create item: a smelly poisonous piece of meat"` and routes it to
  `create_item`; `ClaudeController`'s tool schema confirmed to carry a
  real `description` string parameter or an empty one depending on
  `takes_argument`.
- Full `training/live_run.py` CLI, piped stdin: `create_item` fires
  correctly alongside the pre-existing rate actions in the same
  session, live loop keeps running normally afterward.
- Full `Colony` pipeline, unambiguously: a player-created item sensed
  (but out of pickup range) for several ticks builds a real KC
  eligibility trace with zero reinforcement events; once moved into
  pickup range, a single real pickup fires (hunger 47→98) and
  `reinforce()` correctly updates gain (`gain_drift` 0.0 → 0.0028) —
  the entire chain (item creation → spawn → sense → pickup → Result
  blend → real state delta → `Colony` diff → dopamine-gated update)
  confirmed working end to end on a freshly created item type, not just
  the pre-existing `food`/`threat` types.

**Left open, as before:** who decides an item's finer behavior (spawn
rate weighting, whether it should ever move) beyond the one shared
default is still an open question if it comes up later — deliberately
not addressed now, per "keep it simple."

## 28. Quality review before the frontend: one item concept, an enforced Channel, `game/` split out

**Context:** a full review of the codebase against the wiki's own
contracts, before starting the frontend. The question asked was whether
the code structure matches the stated goal, whether files are clean, and
specifically whether the intent is clear in code for *what an item is
and how it affects a fly*. Findings below are the ones that were acted
on; the review also confirmed several things were already fine
(`world/` never imports `fly_brain/`, private helpers are sparse and
justified, naming is intention-revealing, no file is oversized).

**Finding: "item" was one concept in the wiki and three near-duplicates
in code.** `Food`, `Item`, and `Threat` each had their own class, list,
spawn branch, `observe()` loop, rate, cap, and — worst — their own name
for the same distance (`pickup_radius` / `interaction_radius` /
`sense_radius`). `Food` and `Item` were behaviourally *identical*:
`resolve_food_pickup()` and `resolve_item_pickup()` were the same seven
lines with the nouns swapped. Nothing in the code said "food is just an
item," so the wiki asserted a unified concept the code didn't have.

**Decision:** one `Item` entity, one `radius` (perception *and* pickup —
they were always the same number), one `self.items` list, one
`resolve_item_pickup()`, one `observe()` loop, one `spawn_of()`. Food is
now an ordinary registered `ItemType`; it keeps a `name` only so
`director/` has a handle on the one rate it's allowed to tune.
`ItemType` grew `spawn_rate` and `radius` so every type carries its own
spawn behaviour in one place, and became mutable because `spawn_rate` is
exactly what the director's rate actions adjust. `max_food` folded into
one shared `max_items` — total clutter bounded globally is simpler and
no worse.

**Finding: the wiki's flagship item contract failed, and was provably
failing.** `world.md` specified a test over "`Food`, `Threat`, a freshly
created `Item`" asserting effects come from the Result registry. Run
against the real code, `Threat` failed that third property — it kills
through `determine_fly_death()` and never touches `apply_result()`.

**Decision:** keep the carve-out, but make it honest and *pinned*.
Routing threat lethality through Result-registry similarity would make a
spider's deadliness depend on the encoder's judgment — and with the
current hashing stub, "a fast, venomous spider" scores only weakly
against "toxic, dangerous, harmful poison," so spiders would quietly
stop being lethal and the ES-trained escape checkpoint would no longer
mean what it was trained to mean. Too large a behavioural change to make
on a stub encoder. So: `Threat` satisfies (a) unit-norm attributes and
(b) anonymous perception, explicitly *not* (c) registry-derived effect;
that asymmetry is now documented in `entities.Threat`'s own docstring
rather than only in the wiki, and `tests/test_item_contract.py` asserts
it precisely so the carve-out can't silently widen in either direction.

**Finding: the `Channel` contract was four duplicated string literals
across four files with no enforcement.** `"hunger"`/`"health"`/
`"stuck_ticks"` appeared as bare strings in `results.py`, `env.py`,
`plasticity.py`, and `colony.py`. `results.py`'s docstring claimed the
registry was "extensible without touching anything else" — false: a new
channel also needed edits in `apply_result()`'s three hardcoded lines,
`Fly`, and `CHANNEL_WEIGHTS`. A Result on an unknown channel was
silently dropped.

**Decision:** `Channel` is a `StrEnum` whose every member's value is
exactly the `Fly` attribute it writes. `apply_result()` now iterates the
blend generically against `Environment.channel_limits` instead of
branching per channel, so an unregistered channel raises `KeyError`
rather than vanishing (verified). `plasticity.CHANNEL_WEIGHTS` and
`Colony.state_of()` key off the same enum, so the world and the brain
can't drift apart. `Colony.state_of()` now derives from `Channel`
directly rather than restating the field names.

**Finding: the game lived in `training/`.** `Colony` — the central noun
of the whole design — and `live_run.py`, the actual game loop, both sat
in the package `architecture.md` described as "the ES loop." "Where is
the game?" answered with `training/live_run.py`.

**Decision:** new `game/` package holding `colony.py`, `colony_run.py`,
and `live_run.py`. `training/` keeps only the offline ES process that
writes a checkpoint. CLI entry points are now `python -m game.live_run`
and `python -m game.colony_run`. Done *before* the frontend
deliberately, since the frontend will import from wherever the game
lives.

**Finding: starvation produces no learning signal at all.** Reward is
`max(0, Δhunger)`, so hunger *loss* is never punished — only gaining
hunger rewards. Verified directly: a normal tick, and a fly starving to
death, both produce `reward=0, punishment=0`. Since starvation is the
most common death in every run recorded in this project, a fly routinely
dies having learned nothing. This means the learning rule can only
reinforce *after* a lucky success and cannot bootstrap search — so "no
foraging behaviour" is not purely a training-curriculum gap as
`decisions.md` #21 framed it; it is partly a reward-design gap.

**Decision:** documented honestly in `colony.md` rather than papered
over, and left as an open design question. Fixing it (punishing hunger
loss, or rewarding approach-to-food) would change what the plasticity
circuit optimises and deserves its own decision, not a quiet tweak
during a cleanup pass.

**Also fixed:** two dead symbols deleted (`encode_with_jitter`, whose own
docstring told callers to prefer `jitter()`, and `STATE_CHANNELS`, never
referenced); `Observation`'s pointless field defaults removed;
`load_starting_gains`'s untyped `checkpoint_path` annotated;
`Colony._state_of` unprivatised to match #18's convention;
`build_default_results`'s inconsistent default dropped; `results.py`'s
stale "fly_brain, not built yet" docstring corrected.

**Doc drift corrected rather than code churned:** `architecture.md`
claimed `fly_brain/` imports only `world.env.Action`. In reality it
imports `Action`, `Observation`, `Percept`, `Channel`, and
`ItemEncoder`. That dependency is *real and correct* — a fly's danger
reference vector must live in the same embedding space as the items it's
compared against, so both sides must use the same encoder — so the doc
was wrong, not the code. `architecture.md` now describes the boundary as
"the world's vocabulary" and says why. It also said "three top-level
packages" when there were four (now five).

**Verified:** `tests/test_item_contract.py` — 14 tests, parametrised
over every item-producing pathway. Crucially, the test was checked for
teeth by mutation: leaking a `kind` field into `Percept` fails the
anonymity tests, hardcoding `fly.hunger = max_hunger` on pickup fails
the registry tests, and returning un-normalised vectors from `jitter()`
fails the unit-norm tests. A contract test that can't fail is worthless,
so this was confirmed rather than assumed. Full regression after the
refactor: `game.colony_run`, `game.live_run` (including `create_item`),
and `training.run` stage 1 all re-run end to end.

**Explicitly not done:** `Environment.__init__` still takes ~23
parameters. Grouping them into config objects was considered and
rejected for now — the flat-kwargs shape is what lets `curriculum.py`
express a training stage as a plain dict, which is a documented choice
(#9). Worth revisiting only if a stage ever needs to vary something the
flat shape makes awkward.

## 29. Item/threat/tile brainstorm: structure vs effect; a future two-player "god vs devil" mode

**Context:** two forward-looking conversations, recorded together since
the second reframes an open question the first left behind. Neither is
implemented; both are vision, not a plan.

**Item/threat/tile brainstorm.** The stated ambition is more "god user"
control — different player-created threats, food that lets a fly evolve
permanently (bigger reserves, maybe even a faster learning rate) rather
than just heal — aimed at a feel somewhere between Minecraft (open,
describable placement) and Spore (consequences that accumulate over a
lifetime and across generations). The brainstorm's central move: what
looked like one axis ("item vs threat") is actually two, and conflating
them is what made the boundary feel awkward.

- **Structure** — does it move, is it consumed, does it occupy a point
  or an area. This is a simulation-loop question (which list it lives
  in, what touches it each tick) and should stay a small, discrete,
  *explicitly chosen* set of archetypes — picked by which director
  action/tool gets called (`create_item` vs a future `create_threat` /
  `create_tile`), not inferred from embedding similarity. Inferring
  structure from a fuzzy score is riskier than inferring flavor from
  one: getting it wrong doesn't just misjudge tone, it corrupts the
  simulation.
- **Effect** — what happens to a fly that touches it. This should stay
  continuous and description-driven, the way it already is (#25), but
  split into two tiers: **resource effects** (today's `Channel` deltas —
  transient, bounded, never inherited) and a new **trait effects** tier
  (permanent — max_hunger, max_health, maybe even the plasticity
  circuit's own learning rate — inherited through the same prior-vs-live
  split already built for plasticity gains, #22 part 3/#26). The two
  tiers share one mechanism; they differ in whether the change resets or
  becomes part of what the fly *is*.

Consequence: instant-kill-on-contact stops needing to be its own
mechanism and becomes just the extreme end of the damage scale — a
threat *could* deal graded, survivable damage through the same registry
a food item uses, without literally merging the `Threat` and `Item`
classes (structure can stay separate while effect becomes shared). This
reframes, but does not remove, the blocker already on record: under the
current stub encoder a spider reads as more food-like than damage-like
(verified in the #28 review), so nothing lethal can safely draw on the
effect vocabulary until the real semantic encoder is in place. It was
never really "should `Threat` become `Item`" — it's "is the encoder
trustworthy enough to let any mobile thing's danger be description-
driven," one gate, not two.

Threat *behavioral* variety (stalks vs wanders vs sits-and-hits-hard vs
swarms) is a structure question too, and would reuse the same registry
shape as Results — a small fixed set of archetypes — rather than a new
mechanism. Danger *perception* already varies by description today, for
free, via the existing `danger_vector` similarity (#24) — a
scarier-described threat already provokes a stronger flee response,
before any of this is built.

Real risk flagged, not solved: permanent trait grants compound in a way
transient heals don't. A single overpowered description could make the
colony unkillable, which breaks the premise. Scale constants for trait
effects will need to be deliberately smaller/rarer than resource-effect
scales — a tuning question for whenever this is built, not now.

One boundary reaffirmed, not changed: the player still never edits a
fly directly. Evolutionary food is a placed, describable thing; whether
a fly ever benefits from it is still a consequence of the fly's own
movement and luck, same as everything else in the world today.

**A future two-player "god vs devil" mode.** For better mechanics: two
players — human vs computer, or human vs a friend — one ("god") trying
to grow/protect the colony, one ("devil") trying to wipe it out. Noted
here; **explicitly not started, multiplayer is out of scope for now.**

The reason this doesn't need a new architecture, only a future one:
the loop stays exactly what it already is — *instructions → affect the
world in real time → flies learn from it and benefit/suffer* — just
multiplexed across two instruction sources with opposite objectives
instead of one. Nothing about `Environment`, `director/`, or the live
loop is single-player-shaped; the registry-driven, LLM-translated
control surface was already built generic enough to have more than one
source of instructions pointed at it.

This also gives the still-open "what does winning mean" question
(`wiki/loop.md`) a natural answer for free, rather than needing an
invented single-player score: god wins if the colony survives/grows past
some bar, devil wins if it goes extinct. Worth remembering when that
question is actually decided, since it may make the two questions one
decision instead of two.

Open questions, deliberately left open: human-vs-computer needs an AI
playing one role convincingly (itself an LLM-controller instance,
presumably reusing `WorldController`), and human-vs-human needs a
session/turn model that doesn't exist yet; whether both sides share one
action registry or get asymmetric ones (a "devil" registry biased toward
harm, a "god" one toward relief) is undecided. None of this blocks
anything currently planned before the frontend.

## 30. Per-kind typed creation functions: `create_item` vs `create_mob`, and why only one of them can lie

**Context:** the game was restated as a pipeline — two players (human or
AI) send instructions, instructions shape the world (items, mobs,
environment), the world acts on the colony, the colony learns from what
it survives (`wiki/state.md`, rewritten for this). Assessing the code
against that statement turned up a concrete gap: of the three element
kinds an instruction is supposed to be able to create, only **item** has
a creation function. `env.add_item_type(description)` takes free text and
nothing else; there is no `create_mob` (the one `Threat` type is fixed at
`Environment.__init__`, rate-tunable only), and no tile concept at all.

The proposal that prompted this: give each kind its own function with a
real parameter schema, so `"create a venomous spider that spits fires"`
becomes `create_mob(name='venomous spider', strength=3, type='poison')`,
and the unimplemented part ("spits fire") is simply dropped — never
worked around, and never a reason for anything to edit this project's
source.

**Decision: per-kind typed creation functions, with the schema itself as
the mechanism that drops unsupported requests.** One function per
element kind, each declaring real typed parameters rather than today's
single free-text string. An instruction asking for something the schema
doesn't declare has nowhere to put it — no validation branch, no
rejection path, no fallback. The engine's vocabulary *is* the schema.

This is the structure/effect split from #29 made concrete: *structure* is
which function gets called (discrete, engine-defined, chosen by the
translating model from a fixed list), *effect* is what the arguments and
the embedding produce.

**Decision: the two functions are deliberately asymmetric.**

```
create_item  ->  the embedding drives perception AND effect
create_mob   ->  the embedding drives perception ONLY; effect comes from the parameters
```

**Why:** the encoder-trust gate from #28/#29, applied honestly to each
case rather than uniformly. An item that *reads* as food and *feeds* you
is self-consistent — if the encoder misjudges it, the result is a boring
berry, not a trap. A mob is different: its lethality is what the
ES-trained escape circuit was trained against, and letting the encoder
decide that can silently invalidate the training without anything
failing visibly. So a mob's damage comes from `strength` and `type`,
read directly; `blend_deltas()` is never called on a mob's vector.

This asymmetry also settles the naming question below, and it is why the
`Threat`-vs-`Item` carve-out documented in `wiki/world.md` stays: a
created mob is still not an item on the effect side, on purpose.

**The naming problem, and the decision on it.** A created mob's *name*
has to reach the encoder. Without it every mob sharing a
`(type, strength)` would share one vector, and flies would lose the
ability to tell a spider from a wasp — which is exactly the
open-endedness the encoder exists to provide. But a name that reaches
the encoder can also contradict the parameters:
`create_mob(name='nourishing berry', strength=3, type='poison')`
perceives as food while dealing full poison damage.

Half of that is a feature, not a bug, and worth stating plainly: the
mob still deals its full damage — `strength` is read directly, so a
deceptive name can never make a mob *harmless*, only *unrecognizable*.
Perceptual deception is aggressive mimicry, it is a legitimate move for
a "devil" player, and it is **learnable**: the fly takes damage,
`reinforce()` fires with a negative delta, and the plasticity circuit
drops valence for that vector. The colony learns that berries lie.

What is *not* acceptable is the effect on the **escape reflex**. That
circuit is frozen for life — `danger_vector` is fixed at template build
and ES-trained before the session starts — so unlike the plasticity
circuit it can never learn around a convincing mimic within a
generation. A perfect mimic doesn't just fool a fly, it switches off the
one defense the colony cannot repair.

**Decision: a mandatory mechanical clause, phrased in the Result
registry's own vocabulary.** `create_mob` composes the embedded string
from the player's name *plus* a clause the engine always adds, worded by
deliberately reusing `DANGER_DESCRIPTION` and the `damage` Result's own
reference string. Similarity to exactly those vectors is what
`sense_danger()` and `blend_deltas()` measure, so shared vocabulary is
the mechanism the floor relies on. The mimic becomes imperfect: the
reflex keeps a nonzero signal, and the deception costs the devil some of
its effect instead of being free.

**The two signatures, pinned:**

```
create_item(name: str, description: str)
    embeds: "{name}, {description}"          -- all free text, nothing composed
    effect: blend_deltas() on that vector    -- appearance IS effect

create_mob(name: str, strength: int 1..5, type: enum[poison|physical|sticky])
    embeds: "{name}, a dangerous, fast predator dealing
             {strength_word} {type_phrase} damage"
    perception: that vector, via Percept + sense_danger()
    effect:     strength + type, read directly -- never the vector
```

Items can't lie (appearance and effect are the same string). Mobs can
(they're separate fields). That is why only `create_mob` needs the
clause.

**Worked examples**, real values from the current code (on
`HashingItemEncoder`, see the caveat below):

```
create_item(name='rotten meat', description='rotten poisonous meat')
  embedded : "rotten meat, rotten poisonous meat"
  sim food -0.159  sim damage +0.343
  -> hunger +0.00, health -34.52        both channels, from one string

create_mob(name='venomous spider', strength=3, type='poison')
  embedded : "venomous spider, a dangerous, fast predator dealing
              moderate toxic, harmful poison damage"
  sim danger +0.467  sim damage +0.648  sim food +0.120
  -> damage from strength=3; blend_deltas() NOT called
```

That last line is the point of the asymmetry: if this mob's vector *did*
go through the Result registry it would give `hunger +12.68` alongside
`health -67.72` — the spider would partly **feed** the fly it attacks.
That is the #28 finding reproduced exactly.

**Gated on a measurement, not approved for implementation yet.** Whether
the clause survives being swamped by the name is empirical — a sentence
embedding is dominated by its content words. `world/measure_encoder.py`
(`python -m world.measure_encoder --encoder nomic`) measures three
things, because the proposal can fail in two opposite directions:

1. **Floor** — deceptive mobs' similarity to `danger_vector`, expressed
   as a fraction of an honestly-named spider's own score, since that
   honest score is what ES actually trained against.
2. **Confusion** — does the clause move deceptive mobs off `food` and
   onto `damage`?
3. **Discrimination** — do same-`(type, strength)` mobs stay
   distinguishable? A clause heavy enough to guarantee a floor can
   collapse every mob onto one vector, which is a worse trade than the
   problem it solves. On the stub the registry-vocab clause already
   pushes pairwise similarity to 0.838 mean / 0.903 max, against 0.593 /
   0.723 for a terse phrasing — length alone costs discrimination, so
   clause length is the real tuning dial here.

The script also compares the registry-vocabulary phrasing against a
terse one rather than assuming the former wins; if terse matches it, the
shared-vocabulary reasoning above was wrong and the simpler phrasing
should be used.

**Caveat on every number in this entry:** they come from
`HashingItemEncoder`, which is orthographic — it compares character
trigrams, not meaning. The starkest illustration is that "a sweet ripe
berry" scores `-0.117` against the `food` reference and produces
`hunger +0.00`: on the stub, the flagship item does not feed a fly at
all, while a spider scores `+0.120` on food. The *structure* of both
traces above is real and testable today; the *values* are placeholders
until `NomicItemEncoder` runs somewhere that can reach huggingface.co
(#24).

**Rejected alternatives:**

- **An `appearance` enum** (`create_mob(name=..., appearance='arachnid',
  ...)`) with `name` purely cosmetic. Closes the deception hole
  completely, but puts perception back on a fixed discrete vocabulary —
  precisely what the encoder-based design exists to avoid. Rejected:
  the cure removes the feature.
- **Keeping `name` out of the embedding entirely.** Same problem in a
  cheaper form: every `(type, strength)` collapses to one vector.
- **Letting a mob's lethality come from its embedding**, the way an
  item's does. This is the uniform design, and it's what #29 assumed
  would eventually happen. Rejected here because structured parameters
  make it unnecessary: `strength` gives graded, authored damage without
  asking the encoder to adjudicate anything. Worth noting this
  *sidesteps* the #29 gate rather than waiting on it — creatable mobs no
  longer block on the real encoder for correctness, only for quality.
- **Validating and reporting unsupported requests** ("fire isn't
  implemented"). Rejected as strictly worse than the schema doing it
  silently: a validation path is a place for the engine to grow.

**Still open, deliberately:** `create_tile` has no design at all — the
structure question from #29 (is a tile an `Item` with a lingering radius,
or a new subsystem?) is untouched. A cap on mob types, in the spirit of
`MAX_ITEM_TYPES = 20`, is probably needed: the real exploit isn't one
misleading name but spamming twenty semantically-distant food words so
the colony never converges on any of them. And whether `strength` should
be perceptible at all (section 4 of the measurement script) is upside,
not a requirement.

**Implementation cost, for when this is approved:** the director layer
needs a signature change, not an addition. `ActionSpec` carries
`takes_argument: bool` and `claude_controller.build_tool_definitions()`
hardcodes every parameterized action's schema as
`{"description": {"type": "string"}}`. Supporting per-action typed
parameters means `ActionSpec` carrying a real schema and
`choose_action()` returning arguments rather than
`tuple[str, str | None]` — which touches `base.py`, both controllers, and
`game/live_run.py`'s `apply_requests()`.

## 31. Finishing item/mob/tile: `create_tile`, bidirectional mob valence via signed strength, `Threat` → `Mob`

**Context:** finishing the design pass #30 started. Three things
prompted a revisit rather than a straight extension: (1) `create_tile`
was still undesigned, (2) mobs were reconsidered — "they should be
either good or bad, that's where the custom functions come in," the
same axis items and tiles already have, rather than hostile-only, and
(3) once valence needed to flip sign, keeping it a separate declared
field turned out to be unnecessary. This entry **supersedes #30's mob
signature** (the `disposition`-free, hostile-only version reasoned
through immediately before this one) rather than rewriting it — #30
still stands as the record of why mob effects are authored, never
encoder-derived.

**`create_item`:** unchanged from #30, `name`/`description` split
confirmed.

**`create_tile`: designed.** Reuses the item mechanism entirely —
`perceive()` is already generic over anything with
`position`/`attributes`/`radius`, and `apply_result()` is already
generic over `Channel` — so a `Tile` needs no new perception or effect
code, only a new entity and a per-tick resolution loop instead of a
pickup-and-remove one.

```
create_tile(name: str, description: str)
    embeds: "{name}, {description}"
    effect: blend_deltas() on that vector, scaled down and re-applied
            EVERY TICK a fly remains within radius — never removed
```

Needs a `MAX_TILE_TYPES` cap (same reasoning as `MAX_ITEM_TYPES`: an
unbounded number of lingering area effects is a much bigger exploit
than an unbounded number of consumed items) and a per-tick scale-down
constant so a lava tile isn't instant death and a healing tile isn't
infinite hunger in one step.

**`create_mob`, reworked: bidirectional valence via signed `strength`,
not a declared disposition.**

The insight that resolved this: healing is just damage with the sign
flipped, so `strength` itself can carry valence — `create_mob(name,
strength, channel)`, with `strength` ranging `[-5,-1] ∪ [1,5]` and
`channel` **reusing `Channel` directly** (`HUNGER`/`HEALTH`/
`STUCK_TICKS`) instead of #30's hostile-flavored `poison`/`physical`/
`sticky`, which baked valence into the label itself and couldn't
represent a beneficial effect cleanly.

This does **not** mean `delta = strength * scale` directly — "more is
good" isn't true for every channel:

```
HUNGER:       delta = +strength * scale   (positive strength feeds; negative starves)
HEALTH:       delta = +strength * scale   (positive strength heals; negative damages)
STUCK_TICKS:  delta = -strength * scale   (positive strength FREES; negative traps)
```

The per-channel sign flip is what makes "positive strength always helps
the fly" hold everywhere, including the case #30 didn't anticipate: a
negative-strength `STUCK_TICKS` mob is a coherent trap (as before), and
a **positive-strength one is a rescuer that frees an already-stuck
fly** — mechanically sound under this scheme with no extra code.

Still authored, never `blend_deltas()` — the whole reason #30 kept mob
effect off the encoder (protecting the frozen escape circuit's
training) is unaffected by adding a sign.

**The mandatory clause now triggers off the computed net effect, not a
declared field.** #30's forced embedding clause existed to defend the
frozen escape reflex against a mimic that could actually kill. That
risk is real only when `strength * channel-sign` comes out net-harmful
on `HEALTH` or `HUNGER` — a harmful `STUCK_TICKS` mob is unpleasant, not
lethal, and needs no forced defense either. So disposition is never a
field the player or the translator sets; it falls out of the sign
they already had to provide to get the effect they wanted. That's the
resolution to a bias question raised in this same conversation:
requiring the player to state the *act* ("heals +5"), not just an
adjective ("bad"), is what lets the translator extract a correct signed
`strength` even from a self-contradictory description — but that only
disambiguates the **mechanical** effect. The **name** still goes
untouched into the embedding, so a mob can perceive as dangerous while
mechanically healing (a scary-looking healer a fly flees from) or
perceive as harmless while mechanically harmful (caught by the clause
when net-harmful, same as #30). Two separate risk surfaces, resolved
by two separate mechanisms.

```
create_mob(name: str, strength: int in [-5,-1] ∪ [1,5], channel: Channel)

    embeds:
        net effect harmful on HEALTH/HUNGER:
            "{name}, a dangerous, fast predator dealing {strength_word}
             ...damage"                          -- #30's clause, forced,
                                                     worded from the
                                                     relevant Result's own
                                                     reference vocabulary
        otherwise (beneficial, or harmful-but-not-lethal on STUCK_TICKS):
            "{name}"                             -- flavor only, no forced clause

    perception: that vector, via Percept -- unchanged
    effect: per the sign table above, applied per tick of contact -- authored
```

Worked example, real values, `HashingItemEncoder` (values are
placeholders, see #24/#30's standing caveat — only the *structure* is
load-bearing here):

```
create_mob(name='bad sorcerer', strength=+5, channel=HEALTH)
    net effect: +5 on HEALTH -> beneficial -> no forced clause
    embedded: "bad sorcerer"
    -- still reads as dangerous to sense_danger() if "bad sorcerer"
       scores high on the danger vector: a fly may flee something
       that would have healed it. Not a bug -- the inverse-mimicry
       case this entry's reasoning predicts.
```

**Decision: `Threat` is renamed `Mob`.** The class stopped meaning
"the thing that's always dangerous" the moment it could be beneficial.
`entities.Threat` becomes `entities.Mob`; `world/env.py`'s
`self.threats`/`THREAT_TYPE_NAME`/`move_threats()`/`self.threat_type`
and `world/world.md`'s "the single documented exception" language all
need updating to match at implementation time — noted here as scope,
not done in this entry, which is design only. The built-in spider is
unaffected mechanically (still `strength=None`, still unconditional
insta-kill on contact, from #30's previous entry) — only its class
changes name.

**Rejected alternative:** a separate `disposition: enum[hostile,
beneficial]` field (what the immediately-preceding, now-superseded
reasoning in #30 proposed). Rejected once signed `strength` made it
redundant — carrying the same information twice, with the two
capable of disagreeing (`disposition=beneficial, strength=-5`), is
strictly worse than one field that can't contradict itself.

**Still open:** `Threat`/`Mob`'s effect-composition function needs
per-channel clause wording (a `HUNGER`-harmful mob and a
`HEALTH`-harmful mob shouldn't share one sentence) — left as an
implementation detail, not a design fork; each channel's own existing
Result reference vocabulary (`world/results.py`) is the natural source,
same principle as #30. `MAX_TILE_TYPES`'s value and the tile per-tick
scale-down constant are tuning, not design, and can be set when
implemented.

**Sequencing, reaffirmed:** item/mob/tile creation is now fully
designed. Multi-colony ownership and the resource-cost system (both
raised in this same conversation, not yet logged) are deliberately
**not** folded in here — resource cost would change how `strength`'s
range/cost is bounded, and ownership would add a `target` to every
create_* call, but neither changes the effect mechanics this entry
pins. Finish item/mob/tile first, as agreed; those come after as their
own decisions.

## 32. `channel` renamed `effect` (mob), a uniform `strength` magnitude added to item/tile

**Context:** two questions raised while reviewing #31's signatures. (1)
is `channel` just `effect` under a more implementation-facing name, and
(2) shouldn't item/tile share the same `effect`+`strength` shape mobs
just got. Answered differently — the first was a real naming bug worth
fixing outright; the second would have deleted the actual reason items
work the way they do, so it gets a narrower answer.

**Decision 1: `create_mob`'s `channel` parameter is renamed `effect`,
and absorbs the sign.** `channel` named *which `Fly` field gets
written* — internal plumbing vocabulary, not what a player or a
translating LLM is thinking in. Worse, keeping valence on signed
`strength` (`[-5,-1] ∪ [1,5]`, #31) asked the translator to get a sign
right with nothing to check it against — "create a healer" silently
becomes wrong the moment `strength` comes out positive instead of
negative, with no name attached to catch it.

```
effect ∈ {heal, damage, feed, starve, trap, free}
strength: int in [1, 5]   -- plain positive magnitude, no sign anywhere
```

`heal`/`damage` → `HEALTH`; `feed`/`starve` → `HUNGER`; `trap`/`free` →
`STUCK_TICKS` — recovers #31's "a positive-strength mob can free an
already-stuck fly" case for free, just reachable by naming it directly
(`effect='free'`) instead of inferring it from a sign.

This also simplifies #31's mandatory-clause rule from a sign
computation to a lookup: **force the clause when `effect` is `damage`
or `starve`** (harmful and capable of killing); skip it for `trap`
(harmful but never lethal) and for `heal`/`feed`/`free` (beneficial) —
identical outcomes to #31's rule, cleaner to state and to implement.

**Decision 2: item/tile do NOT get an authored `effect` field — the
emergent, description-derived blend is unchanged.** Full parity was
considered and rejected, for reasons already on record rather than new
ones:

- it would delete the Minecraft case (#22's poisonous meat healing
  *and* damaging from one description, discovered by the encoder,
  never declared) — an `effect` enum forces one channel, reintroducing
  exactly the per-item authoring burden the encoder exists to remove
- it would contradict the actual reason the real encoder was chosen
  in the first place (this project's very first item-authoring
  decision): same encoder for everything only gives the
  `king - man + woman = queen`-style generalization property if effect
  is *derived*, never declared per item
- it would break the item contract as tested
  (`tests/test_item_contract.py`): effect must be "fully explained by
  `blend_deltas()` on its own attribute vector" — an authored `effect`
  field is precisely the hardcoded-by-name constant that contract
  exists to forbid

None of that reasoning applies to mobs, and the difference is why the
asymmetry is deliberate rather than an oversight carried over
unexamined: mob effect is authored because the encoder can't be
trusted with anything that invalidates the frozen escape circuit's
training (#30); items/tiles carry no such risk — a misjudged blend
gives an oddly-shaped item, never a training-invalidating one.

**What item/tile do gain: a uniform `strength` that scales magnitude
only, never shape or sign.**

```
create_item(name: str, description: str, strength: int in [1, 5])
    embeds: "{name}, {description}"
    effect: blend_deltas(vector) * magnitude(strength)
            -- shape (which channels move) and sign (heal vs harm) stay
               100% from the description; strength only sets overall
               potency

create_tile(name: str, description: str, strength: int in [1, 5])
    same, applied per tick, persistent
```

"Rotten poisonous meat, strength 5" still heals *and* damages from the
description alone — both deltas scale up together, proportionally.
`strength` never selects a channel or a sign here, so the pillar
contract's actual wording ("nothing about *what it is*... as a name or
type") is untouched; the item contract test gains "scaled by
`strength`" to its check, not a new failure mode.

**Consequence, noted for when the resource-cost system (raised
alongside multi-colony ownership, #31's sequencing note, neither
designed yet) gets designed:** all three creation functions now share
one `strength` field, which is exactly the single hook a uniform
creation-cost formula needs — one place to price creation across item,
tile, and mob, rather than three different shapes to reconcile later.

**Signatures, final:**

```
create_item(name, description, strength: 1..5)
create_tile(name, description, strength: 1..5)
create_mob(name, effect: {heal,damage,feed,starve,trap,free}, strength: 1..5)
```

## 33. Resource-cost system, designed: per-player energy, gating creation only

**Context:** raised alongside the item/mob/tile implementation
(`decisions.md` #30–#32's sequencing note): a resource pool bounding
creation, so spamming it is a trade-off rather than either a hard wall
(today's `MAX_ITEM_TYPES`-style caps) or unlimited. Designed here, not
implemented — no code changes in this entry.

**Decision: the pool is keyed by player, not by colony, and this is
load-bearing, not a naming choice.** Energy represents *capacity to
issue instructions* — a property of whoever's sending them (human or
AI, `wiki/state.md`), never of the fly population those instructions
land on. Concretely, per-colony breaks the other multiplayer mode
already on record: **god vs devil** (`decisions.md` #29) is one shared
colony with two players who need two separate pools — there's nowhere
to hang a second pool on a single colony. **Multi-colony** (raised in
this same conversation, not yet designed) is one colony per player, so
player and colony happen to be 1:1 there — which is exactly why the two
scopes look interchangeable in that mode and nowhere else. Keying by
player is the one structure that works for both: a dict from player-id
to energy, independent of how many colonies exist or what happens to
any particular one (a colony dying and a fresh one starting shouldn't
reset or duplicate the player's banked energy).

Today there's only one instruction source, so this is a **single global
pool for now** (`Environment.energy: float`) — but structured as the
scalar a future `dict[player_id, float]` trivially generalizes from,
not as something attached to `Colony`.

**Decision: gates creation only, not the rate nudges.**
`increase_/decrease_spider_rate` and `increase_/decrease_food_rate` stay
free — they're already self-limiting (bounded `[0, 0.2]`, a fixed
`0.02` step, fully reversible). Creation adds a standing capability to
the world that persists until the type cap forces something out; that's
what's worth pricing. Rate-nudging remains the always-available baseline
tool, unconstrained by energy.

**Decision: cost is a function of `(kind, strength)` only — never the
description's derived blend potency**, even though `strength` alone
gives a less "honest" number than pricing off what an item/tile
actually turns out to do. Considered and rejected: pricing items/tiles
off `blend_deltas()`'s real magnitude reopens exactly what #32 closed —
`strength` was made the one dial that's simple, predictable, and
decoupled from the description, which drives shape/sign only. Pricing
off both couples two systems #32 deliberately kept apart, and makes
cost unknowable to the translating LLM ahead of time. If flat pricing
proves exploitable in practice, potency-weighted pricing is the fallback
to revisit — not the v1 default.

```python
KIND_BASE_COST = {"item": 2.0, "tile": 4.0, "mob": 3.0}  # tiles cost more: persistent, re-applies every tick
def creation_cost(kind: str, strength: int) -> float:
    return KIND_BASE_COST[kind] * clamp_strength(strength)
```

Linear in `strength`, not convex — simplest, and there's no evidence yet
that flat-rate max-strength spam is a real problem worth pre-solving;
easy to swap for something superlinear later if playtesting says
otherwise.

**Decision: energy regenerates on a fixed per-tick schedule, not tied to
colony state.** Considered and rejected: regen scaling with colony
health/population (a thriving colony arms whoever's controlling it with
more power) creates a feedback loop whose direction depends on which
side a player is on — and there's no fixed god/devil alignment on
record yet to reason about which way that should cut. Flat regen is
fair regardless of which side a player is playing, and doesn't presume
an alignment the game doesn't have yet.

**Decision: the existing type-count caps (`MAX_ITEM_TYPES` /
`MAX_TILE_TYPES` / `MAX_MOB_TYPES`, #27/#31) stay, as a much-higher
backstop, not removed.** Energy becomes the limiter a player actually
feels during play; the cap becomes a sanity/memory bound that's
essentially never hit. Guard order in `add_item_type`/`add_tile_type`/
`add_mob_type`: cap check first, then affordability, deduct only on
success — a rejected request never costs anything.

**Decision: `add_item_type`/`add_tile_type`/`add_mob_type` gain a `bool`
return (created or not), instead of today's unconditional `None`.** A
cap-rejected call and an energy-rejected call are different failures
worth distinguishing from a schema-rejected one (nothing to say there —
the request never named a real capability) — an energy-rejected request
was understood perfectly and still couldn't be afforded, which
`game/live_run.py` should be able to log distinctly (`"insufficient
energy"` vs `"created"`). Small signature change; also the natural hook
the preview/confirm UX (raised alongside multi-colony and this system,
`decisions.md` #32's sequencing note) would build on later.

**Open, not decided in this entry:** the resource's name (`energy` used
throughout, following the player's own mockup wording — not
re-litigated here); starting/max/regen constants (no strong opinion
yet — enough for a couple of `strength=5` mobs early, not a constant
stream — tune empirically rather than pick now); whether the
translating LLM should ever see the current balance (kept invisible to
the controller for v1 — the environment enforces the budget the same
way it already enforces `effect`/`strength` validity, silently, not as
something the model reasons about).

**Not implemented.** Next real decision before building it: the exact
starting/max/regen constants, informed by actually playing with #30–#32
first.
