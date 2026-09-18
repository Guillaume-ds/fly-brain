# Architecture

## The pattern: Agent / Environment / Trainer

The project is split into three top-level packages that mirror a standard
RL software pattern (the same shape as Gym/RLlib-style codebases):

```
fly_brain/     the fly and its brain (the "Agent")
world/         the game/simulation (the "Environment")
training/      the process that connects them and trains the brain
```

The core discipline: **`world/` never imports from `fly_brain/`, and
`fly_brain/` never imports the world's internals** (agent.py does import
`world.env.Action` for its return type, since the agent has to speak the
environment's action vocabulary — but nothing in `world/` knows the brain
exists). Only `training/` is allowed to depend on both. This means the
environment can be tested and debugged with random actions and no brain at
all, and the brain's circuit can be tested and debugged with synthetic
input and no environment at all — which is exactly how both were built and
verified before being wired together.

## `world/` — the game

- **`entities.py`** — plain dataclasses: `Position`, `Food`, `Threat`,
  `Fly` (id, position, hunger, `vulnerable_ticks_left`). No behavior,
  just data.
- **`items.py`** (new, `decisions.md` #22/#24, in progress) — the
  content-authoring pipeline that turns a short text description into an
  item instance's small, fixed-dimension attribute vector: `ItemEncoder`
  is a swap point (same pattern as `director/`'s `WorldController`) —
  `NomicItemEncoder` is the real backend (local, frozen
  `nomic-embed-text-v1.5`, Matryoshka-truncated), **not exercised live in
  this dev environment** (`huggingface.co` is policy-blocked here, same
  situation as `ClaudeController` with no API credentials);
  `HashingItemEncoder` is a zero-dependency, deterministic stub (feature
  hashing over character trigrams — orthographic, not semantic) used to
  test everything downstream without network access.
  `encode_with_jitter()` adds per-instance Gaussian noise and
  re-normalizes to unit norm, so every downstream consumer can use plain
  cosine similarity. Entirely a content-authoring concern — never
  exposed to a fly.
- **`env.py`** — the `Environment` class, multi-fly: `self.flies:
  list[Fly]` share one grid, one set of spiders/food, one hunger clock
  each. Food and threats spawn continuously (not placed once at reset) at
  `spider_spawn_rate`/`food_spawn_rate`, and threats wander
  (`threat_move_probability` chance of a random step per tick) — both
  needed for the escape circuit to have real, ongoing pressure to react
  to rather than a one-time, permanently-dodgeable placement (see
  `decisions.md` #14, #15). Flies reproduce — a world-level stochastic
  event, not an agent decision, gated by hunger + a population-based
  `mate_availability` factor, with a real cost to the parent (hunger +
  a forced-`STAY` vulnerability window); see `decisions.md` #20 for the
  formula. `increase_/decrease_spider_rate` and
  `increase_/decrease_food_rate` are the only way those rates change —
  the control surface `director/` calls into. Exposes a Gym-style
  interface: `reset() -> dict[fly_id, Observation]`,
  `step(actions: dict[fly_id, Action]) -> ColonyStepResult(observations,
  deaths, births, colony_extinct, timed_out)`. A population of 1 makes
  reproduction structurally impossible (the formula), which is exactly
  the single-fly curriculum-training case — one class serves both, no
  separate single-fly environment. `food_enabled` / `threats_enabled`
  flags mean training stages configure this one class differently rather
  than needing separate environment implementations.

The `Observation` the environment exposes is a fixed 7-value vector:
`food_signal, food_dx, food_dy, threat_signal, threat_dx, threat_dy,
hunger`. Food/threat signal and direction are **zero outside that entity's
radius** — the fly gets no information about something it hasn't come
close enough to sense (this is deliberate: it's what makes finding food an
actual search problem instead of pure gradient-climbing from anywhere on
the map). `hunger` is always visible (interoceptive, not sensed).

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
- **`agent.py`** — `build_escape_template()` does the expensive, shared
  part once (load the connectome, expand the circuit, find the seed/motor
  neuron indices) and returns an `EscapeCircuitTemplate`; `EscapeAgent
  (template)` is cheap — a fresh `Circuit` off that shared blueprint, so a
  colony of many flies builds the connectome/topology once and stamps out
  one lightweight agent per fly. `EscapeAgent` wraps that `Circuit` as a
  policy for the survival task: injects `threat_signal` as stimulus
  current into `DNp01` (the Giant Fiber), and checks whether the real
  downstream `TTMn` neuron (the jump-muscle motor neuron) spikes to decide
  *whether* to flee. *Which direction* to flee is plain geometry (away
  from the threat), computed outside the circuit — not something the
  circuit decides. See `decisions.md` for why that split exists. No
  foraging behavior yet — food/hunger are part of `Observation` but
  unused here; a fly only eats what it happens to wander into.
- **`analyze.py` / `simulate.py`** — the original standalone
  exploration/demo commands (`python -m fly_brain analyze|simulate`),
  independent of the survival-game project; still useful for poking at the
  raw connectome.

## `training/` — the ES loop

- **`curriculum.py`** — stage configs as plain `Environment` kwargs (only
  stage 1 defined so far).
- **`trainer.py`** — Evolution Strategies (the OpenAI-ES update), operating
  only on `EscapeAgent.get_params()`/`set_params()` (the `synaptic_gain`
  vector). Rewards standardized per iteration.
- **`run.py`** — CLI (`python -m training.run --stage 1`); checkpoints to
  `training/checkpoints/` (gitignored).
- **`colony.py`** — `Colony` glues `Environment`'s multi-fly reproduction
  mechanic to real `fly_brain` circuits: `dict[fly_id, EscapeAgent]`, one
  per living fly. Each tick every fly acts via its own agent; each birth
  gets a new agent whose genome is the parent's `get_params()` plus
  Gaussian noise (`mutation_sigma`); each death deletes that fly's agent.
  `load_starting_gains()` seeds the colony from a trained ES checkpoint
  when available, or untrained (gain=1.0) otherwise.
- **`colony_run.py`** — CLI (`python -m training.colony_run`) that runs a
  colony headlessly and logs population/births/deaths — a cheap way to
  watch it work before the real frontend exists, not the live game loop
  itself (no `director/` involved yet).
- **`live_run.py`** — the live game loop (`python -m training.live_run`,
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

- **`actions.py`** — the registry: `ActionSpec(name, description, fn)`
  wrapping `Environment.increase_/decrease_spider_rate` and
  `increase_/decrease_food_rate`. No LLM-specific code — doesn't know
  Claude or any other provider exists.
- **`base.py`** — `WorldController`, a one-method interface
  (`choose_action(request, actions) -> action name or None`). This is the
  entire swap point.
- **`rule_based_controller.py`** — zero-dependency keyword-match backend,
  used to test the registry/`Environment` wiring for free before any real
  model is involved.
- **`claude_controller.py`** — the reference LLM backend: real tool-use
  (each action is a zero-argument tool), not free-text parsing.

## Data flow through one tick (per fly)

```
Environment.step({fly_id: action, ...}) -> ColonyStepResult.observations[fly_id]
        |
EscapeAgent.act(Observation)      (that fly's own circuit/gains)
        |
   inject threat_signal as current into DNp01
        |
   Circuit.step(current) -> spikes  (real topology, trainable gains)
        |
   TTMn spiked? -> flee (direction = -threat direction) : STAY
        |
        v
      Action  ->  Environment.step() on the next tick
```

(Curriculum training always runs a population of 1, so this collapses to
exactly one fly per tick, same shape as before multi-fly support existed.
A live colony repeats this per living fly each tick — driving each fly
with its own circuit/genome is `training/colony.py`'s job, not yet built.)

## How a player's request changes the world

```
player request (free text)
        |
WorldController.choose_action(request, registry)   <- Claude, rule-based, or any future backend
        |
   action name (or None)
        |
   registry[name].fn()   -> Environment.increase_/decrease_*_rate()
        |
        v
  spider_spawn_rate / food_spawn_rate  ->  affects every future _spawn_tick()
```
