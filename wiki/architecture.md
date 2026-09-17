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

- **`entities.py`** — plain dataclasses: `Position`, `Food`, `Threat`. No
  behavior, just data.
- **`env.py`** — the `Environment` class. Fixed-size grid, a hunger bar
  that depletes every tick. Food and threats spawn continuously (not
  placed once at reset) at `spider_spawn_rate`/`food_spawn_rate`, and
  threats wander (`threat_move_probability` chance of a random step per
  tick) — both needed for the escape circuit to have real, ongoing
  pressure to react to rather than a one-time, permanently-dodgeable
  placement (see `decisions.md` #14, #15). `increase_/decrease_spider_rate`
  and `increase_/decrease_food_rate` are the only way those rates change —
  the control surface `director/` calls into. Exposes a Gym-style
  interface: `reset() -> Observation`,
  `step(action) -> StepResult(observation, reward, done, cause)`.
  Curriculum-friendly by construction — `food_enabled` / `threats_enabled`
  flags mean later training stages configure this one class differently
  rather than needing separate environment implementations.

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
- **`agent.py`** — `EscapeAgent` wraps a `Circuit` as a policy for the
  survival task. Seeds on `DNp01` (the Giant Fiber), injects
  `threat_signal` as stimulus current into it, and checks whether the real
  downstream `TTMn` neuron (the jump-muscle motor neuron) spikes to decide
  *whether* to flee. *Which direction* to flee is plain geometry (away
  from the threat), computed outside the circuit — not something the
  circuit decides. See `decisions.md` for why that split exists.
- **`analyze.py` / `simulate.py`** — the original standalone
  exploration/demo commands (`python -m fly_brain analyze|simulate`),
  independent of the survival-game project; still useful for poking at the
  raw connectome.

## `training/` — not yet built

Will hold `curriculum.py` (stage configs — which entities are enabled, at
what difficulty) and `trainer.py` (the ES training loop: perturb
`synaptic_gain`, roll out episodes via `Environment` + `EscapeAgent`,
update toward higher survival time). See `roadmap.md`.

## Data flow through one tick, once training exists

```
Environment.step(action) -> Observation
        |
EscapeAgent.act(Observation)
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
