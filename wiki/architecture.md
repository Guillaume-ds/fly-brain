# Architecture

## The pattern: Agent / Environment / Game / Trainer

The project is split into five top-level packages. The first three mirror
a standard RL software pattern (the same shape as Gym/RLlib-style
codebases); the other two sit on top:

```
fly_brain/     the fly and its brain (the "Agent")
world/         the simulation (the "Environment")
game/          the colony living in the world, and the live loop you play
training/      the offline ES process that produces a starting genome
director/      the swappable LLM layer the player edits the world through
tests/         contract tests (see wiki/world.md for what a contract is here)
```

`game/` and `training/` are deliberately separate (`decisions.md` #28):
`training/` is an offline process that writes a checkpoint, `game/` is
what you actually play. The live loop and the `Colony` belong in the
latter — they used to live in `training/`, which made "where is the
game?" an unnecessarily confusing question.

The core discipline: **`world/` never imports from `fly_brain/`** — nothing
in `world/` knows a brain exists, so the environment can be driven with
random actions and no brain at all. The reverse direction is allowed but
narrow: `fly_brain/` imports only the world's *boundary vocabulary* —
`Action`, `Observation`/`Percept`, `Channel`, and `ItemEncoder`. That
last one is a real, deliberate dependency, not an accident: a fly's
danger reference vector has to live in the same embedding space as the
items it's compared against, so both sides must use the same encoder.
`game/` depends on both by design, and reads `training/`'s checkpoint
path.

## `world/` — the simulation

- **`entities.py`** — plain dataclasses: `Position`, `Item`, `Threat`,
  `Fly`. There is exactly **one** item entity (`decisions.md` #28):
  food is not a class, it's a registered `ItemType` like any
  player-created one. `Item` is stationary, single-use, and has no
  behavior beyond what its attribute vector produces through the Result
  registry; `radius` is both how far it can be perceived and how close a
  fly must be to pick it up — one distance, one name. `Threat` is the
  documented exception: same shape, same anonymous perception, but it
  moves, is never consumed, and kills through `determine_fly_death()`
  rather than the Result registry (see its docstring for why). `Fly`
  carries id, position, hunger, `health`, `stuck_ticks`, and
  `vulnerable_ticks_left`. No behavior, just data.
- **`items.py`** (`decisions.md` #22/#24/#27) — the content-authoring
  pipeline that turns a short text description into an item's small,
  fixed-dimension attribute vector: `ItemEncoder` is a swap point (same
  pattern as `director/`'s `WorldController`) — `NomicItemEncoder` is
  the real backend (local, frozen `nomic-embed-text-v1.5`,
  Matryoshka-truncated), **not exercised live in this dev environment**
  (`huggingface.co` is policy-blocked here, same situation as
  `ClaudeController` with no API credentials); `HashingItemEncoder` is a
  zero-dependency, deterministic stub (feature hashing over character
  trigrams — orthographic, not semantic) used to test everything
  downstream without network access, and what every CLI entry point
  actually runs today. `ItemType` is everything the world needs to keep
  spawning one kind of item — name, description, prototype vector
  (encoded once), `spawn_rate`, `radius`. Food, threats, and
  player-created types are all `ItemType`s; `name` exists only so
  `director/` has a handle on the food rate it's allowed to tune.
  `jitter()` applies per-instance Gaussian noise to an already-encoded
  prototype at spawn time, re-normalized to unit norm. Entirely a
  content-authoring concern — never exposed to a fly.
- **`results.py`** (`decisions.md` #22 part 5, #25, #28) — the Result
  registry: a small, fixed set of real mechanical outcomes, each encoded
  the same way items are. `Channel` is a `StrEnum` whose every member's
  value names the `Fly` field it writes (`HUNGER`→`hunger`, etc.) — that
  link is what lets `apply_result()` stay generic instead of branching
  per channel, and it means a Result on an unregistered channel raises
  rather than being silently dropped. `blend_deltas()` is a
  clipped-cosine-similarity blend across every registered Result,
  applied simultaneously — the mechanism that lets one item be both
  nourishing and damaging at once. The one place in this system that's
  deliberately discrete rather than open-ended, on purpose.
- **`env.py`** — the `Environment` class, multi-fly: `self.flies:
  list[Fly]` share one grid, one set of items and threats, one hunger
  clock each. Every registered `ItemType` — food included — spawns
  through the same `spawn_of()` path into the same `self.items` list at
  its own `spawn_rate`, bounded by one shared `max_items`; threats spawn
  the same way into their own list and wander
  (`threat_move_probability` chance of a random step per tick), which
  the escape circuit needs so there's real, ongoing pressure to react to
  rather than a one-time, permanently-dodgeable placement (see
  `decisions.md` #14, #15). `add_item_type(description)` (`decisions.md`
  #27) is the player-driven item-creation entry point — registers a new
  `ItemType`, capped at `MAX_ITEM_TYPES`. Picking up an item doesn't
  restore hunger by a fixed amount — `apply_result()` runs its attribute
  vector through the Result registry and writes whatever real deltas
  come out, generically over `Channel` (`decisions.md` #28). A fly with
  `stuck_ticks > 0` is forced to `STAY`, same pattern as the existing
  `vulnerable_ticks_left` reproduction-cooldown mechanic. Flies
  reproduce — a world-level stochastic event, not an agent decision,
  gated by hunger + a population-based `mate_availability` factor, with a
  real cost to the parent (hunger + a forced-`STAY` vulnerability
  window); see `decisions.md` #20 for the formula.
  `increase_/decrease_spider_rate` and `increase_/decrease_food_rate` are
  the only way those rates change — the control surface `director/` calls
  into. Exposes a Gym-style interface: `reset() -> dict[fly_id,
  Observation]`, `step(actions: dict[fly_id, Action]) ->
  ColonyStepResult(observations, deaths, births, colony_extinct,
  timed_out)`. A population of 1 makes reproduction structurally
  impossible (the formula), which is exactly the single-fly
  curriculum-training case — one class serves both, no separate
  single-fly environment. `food_enabled` / `threats_enabled` flags mean
  training stages configure this one class differently rather than
  needing separate environment implementations.

The `Observation` the environment exposes is an **anonymous** `nearby:
list[Percept]` plus `hunger` (`decisions.md` #22/#25) — each `Percept`
carries only a unit-norm attribute vector and relative position (`dx,
dy, distance`), no name, type, or id. Items and threats go through the
same `perceive()` call and are indistinguishable once they're percepts.
Each entity's own `radius` decides visibility, a world-internal cutoff
that stays on the world's side of the boundary. `hunger` is always
visible (interoceptive, not sensed). This is the property
`tests/test_item_contract.py` pins.

Action space is 5 discrete moves: `STAY, UP, DOWN, LEFT, RIGHT`.

## `fly_brain/` — the fly and its brain

- **`data.py`** — downloads and caches the 3 MaleCNS connectome files used
  (`body-annotations`, `body-neurotransmitters`,
  `connectome-weights-*-significant-only`) from Google's public bucket, no
  auth needed. Returns pandas DataFrames.
- **`circuit.py`** — `build_circuit()` expands outward from a seed neuron
  type (e.g. `DNp01`) following the strongest real outgoing connections, up
  to N hops / max neurons. `Circuit` is a small LIF (leaky integrate-and-fire)
  spiking network over that real subgraph: real topology, real synapse
  sign (from predicted neurotransmitter — GABA is inhibitory, everything
  else excitatory), real relative synapse strength. Each synapse also has
  a `synaptic_gain` multiplier (init `1.0`), which is the one thing
  training is allowed to touch — see `decisions.md` on why topology/sign
  stay fixed.
- **`agent.py`** — `build_escape_template(encoder, ...)` does the
  expensive, shared part once (load the connectome, expand the circuit,
  find the seed/motor neuron indices, encode a `danger_vector` from the
  same `ItemEncoder` everything else uses) and returns an
  `EscapeCircuitTemplate`; `EscapeAgent(template)` is cheap — a fresh
  `Circuit` off that shared blueprint, so a colony of many flies builds
  the connectome/topology once and stamps out one lightweight agent per
  fly. `EscapeAgent` wraps that `Circuit` as a policy for the survival
  task: `sense_danger()` computes cosine similarity between each nearby
  anonymous `Percept` and `danger_vector` (clipped at 0, weighted by
  proximity, **summed** across every matching percept so multiple
  simultaneous threats are never less urgent than one), injects that as
  stimulus current into `DNp01` (the Giant Fiber), and checks whether the
  real downstream `TTMn` neuron (the jump-muscle motor neuron) spikes to
  decide *whether* to flee. *Which direction* to flee is plain geometry
  (away from the single nearest above-threshold percept), computed
  outside the circuit — not something the circuit decides. See
  `decisions.md` #5 for why that split exists, and #24/#25 for why
  `sense_danger()` replaced a named `threat_signal` field once
  `Observation` became anonymous. No foraging behavior yet — a fly only
  eats what it happens to wander into.
- **`plasticity.py`** (`decisions.md` #22/#26) — the second real circuit:
  Kenyon Cells (KC) → Mushroom Body Output Neurons (MBON), gated by
  Dopaminergic Neurons (DAN), implementing lifetime, dopamine-gated
  Hebbian plasticity, fully separate from the frozen escape circuit
  above. Built from a bounded, real subgraph (481 neurons, 6,442 edges,
  1,000 plastic KC→MBON synapses) rather than the full ~4,500-neuron
  real population, which is computationally impractical per-tick,
  per-fly: the top-100 real KCs by total real KC→MBON weight, their
  real top-20 targets each (auto-discovering 50 real MBONs), and every
  DAN with a real edge into that universe (300 PAM=reward-coding, 31
  PPL/PPM=punishment-coding). `PlasticityAgent.sense_and_decide()` runs
  every tick regardless of outcome (its KC eligibility trace has to keep
  running even on ticks the escape circuit overrides the actual move);
  `reinforce(deltas)` takes the real `Δhunger/Δhealth/Δstuck_ticks` a
  tick produced (never a percept similarity score directly) and applies
  a direction-aware Hebbian update — reward potentiates the approach
  pathway and depresses avoid for the KCs that were active, punishment
  the reverse (a deliberate adaptation of the textbook depression-only
  rule to fit this circuit's direct-spike-readout MBON, not the
  biological rule verbatim). DAN stimulus uses a fixed per-neuron random
  gain rather than uniform current — needed for the DAN population's
  spike fraction to actually grade with magnitude instead of saturating
  all-or-nothing (found empirically, `decisions.md` #26). Net valence
  (approach − avoid) reads MBON **membrane potential**, not spike count
  — also found empirically: spike count is too coarse to track gradual
  synaptic change tick to tick. `probe()` is a read-only diagnostic (net
  valence for a given attribute vector, zero side effects) and
  `gain_drift()` a cheap "how much has this fly learned" scalar — the
  auditability hooks built in per an explicit ask; see `Colony` below
  for how they surface at the population level.
- **`analyze.py` / `simulate.py`** — the original standalone
  exploration/demo commands (`python -m fly_brain analyze|simulate`),
  independent of the survival-game project; still useful for poking at the
  raw connectome.

## `training/` — the offline ES loop

Produces a starting genome and writes it to a checkpoint. Nothing here
runs while you're playing.

- **`curriculum.py`** — stage configs as plain `Environment` kwargs (only
  stage 1 defined so far).
- **`trainer.py`** — Evolution Strategies (the OpenAI-ES update), operating
  only on `EscapeAgent.get_params()`/`set_params()` (the `synaptic_gain`
  vector). Rewards standardized per iteration.
- **`run.py`** — CLI (`python -m training.run --stage 1`); checkpoints to
  `training/checkpoints/` (gitignored).

## `game/` — the colony and the live loop

What you actually play (`decisions.md` #28). Depends on `world/`,
`fly_brain/`, and `director/`; reads `training/`'s checkpoint path.

- **`colony.py`** — `Colony` glues `Environment`'s multi-fly reproduction
  mechanic to real `fly_brain` circuits: every living fly has both an
  `EscapeAgent` and a `PlasticityAgent` (`decisions.md` #26). Each tick,
  the plasticity circuit always senses and decides first (its eligibility
  trace must keep running regardless of outcome); the escape circuit's
  `decide()` returns `None` when it has no override (`TTMn` didn't
  spike), and only then does the plasticity circuit's own decision get
  used — the freeze+override rule from `decisions.md` #5, concretely
  wired. `Colony` snapshots each fly's real state before `env.step()`
  and diffs it after for every survivor, feeding that real
  `Δhunger/Δhealth/Δstuck_ticks` to `reinforce()` — `Environment` itself
  never knows `fly_brain` exists or that "reward" is a concept.
  On birth: the escape genome inherits the parent's `get_params()` plus
  Gaussian noise, as before; the plasticity genome inherits the parent's
  `prior_gains` (what *it* was born with) plus mutation — never the
  parent's own lifetime-drifted live gains, per `decisions.md` #22 part
  3. `plasticity_summary()` gives a population-level audit snapshot
  (mean gain drift, total reinforcement events). `load_starting_gains()`
  seeds the escape circuit from a trained ES checkpoint when available,
  or untrained (gain=1.0) otherwise; the plasticity circuit currently
  always starts untrained (gain=1.0) — no ES-style training loop exists
  yet for its prior, see roadmap.md.
- **`colony_run.py`** — CLI (`python -m game.colony_run`) that runs a
  colony headlessly and logs population/births/deaths — a cheap way to
  watch it work before the real frontend exists, not the live game loop
  itself (no `director/` involved yet).
- **`live_run.py`** — the live game loop (`python -m game.live_run`,
  `decisions.md` #23): a background thread reads player requests from
  stdin into a queue; the main loop ticks the `Colony` at a fixed
  real-time rate (`--ticks-per-second`) and drains/applies pending
  requests through `director/`'s registry each tick, so a request never
  pauses the world. `Environment.max_ticks` is effectively unbounded here
  — the session ends on colony extinction or the player typing `quit`,
  not a tick cap. `apply_requests()`/`drain()` are pure, I/O-free
  functions (testable directly); only `read_requests()`/`main()` touch
  real stdin/threading.

## `director/` — the swappable LLM world-controller

Sits on top of `Environment`, not on top of the fly. A player's request
gets translated into at most one call against a fixed action registry —
see `decisions.md` #16 for why each piece is split the way it is.

- **`actions.py`** — the registry: `ActionSpec(name, description, fn,
  takes_argument=False)` wrapping `Environment.increase_/
  decrease_spider_rate`, `increase_/decrease_food_rate`, and (since
  `decisions.md` #27) `create_item` — the first action with
  `takes_argument=True`, wrapping `Environment.add_item_type` directly.
  No LLM-specific code — doesn't know Claude or any other provider
  exists.
- **`base.py`** — `WorldController`, a one-method interface
  (`choose_action(request, actions) -> (action name, argument) or None`).
  This is the entire swap point. `argument` only matters when that
  action's `takes_argument` is True.
- **`rule_based_controller.py`** — zero-dependency keyword-match backend,
  used to test the registry/`Environment` wiring for free before any real
  model is involved. Handles `create_item` the same crude way as
  everything else it does: a fixed trigger phrase (`"create item:"` etc.),
  everything after it taken verbatim as the description.
- **`claude_controller.py`** — the reference LLM backend: real tool-use,
  not free-text parsing. `create_item` is the first tool with a real
  `input_schema` (a `description` string) instead of an empty one — the
  parameterized-tool-use path the controller was built to support from
  the start, now actually exercised.

## Data flow through one tick (per fly)

```
Environment.step({fly_id: action, ...}) -> ColonyStepResult.observations[fly_id]
        |
EscapeAgent.act(Observation)      (that fly's own circuit/gains)
        |
   sense_danger(obs.nearby, danger_vector) -> (flee_dx, flee_dy, strength)
        |
   inject strength as current into DNp01
        |
   Circuit.step(current) -> spikes  (real topology, trainable gains)
        |
   TTMn spiked? -> flee (direction = -flee_dx/-flee_dy) : STAY
        |
        v
      Action  ->  Environment.step() on the next tick
```

(Curriculum training always runs a population of 1, so this collapses to
exactly one fly per tick, same shape as before multi-fly support existed.
A live colony repeats this per living fly each tick, and also runs the
plasticity circuit alongside the escape circuit shown here — see
`game/colony.py` above and `decisions.md` #26 for how the two
combine.)

## How a player's request changes the world

```
player request (free text)
        |
WorldController.choose_action(request, registry)   <- Claude, rule-based, or any future backend
        |
   (action name, argument) or None
        |
   registry[name].fn()            -> Environment.increase_/decrease_*_rate()
   registry[name].fn(argument)    -> Environment.add_item_type(description)   [decisions.md #27]
        |
        v
  spider_spawn_rate / food_spawn_rate  ->  affects every future spawn_entities()
  a new ItemType                       ->  starts spawning as a generic Item
```
