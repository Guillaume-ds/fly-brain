"""WanderAgent: a fixed fallback movement reflex, evolved across
generations, never touched by lifetime learning -- the "movement" half
of the exploration-bootstrapping gap (wiki/decisions.md #39 fixed the
"sensing" half: a fly can now perceive food from a real distance via
WORLD_SENSING_RADIUS, but until this, a fly with nothing perceived did
nothing but Action.STAY, forever -- there was no fallback movement
anywhere in the codebase).

Structurally parallel to EscapeAgent, not PlasticityAgent: like the
escape circuit's synaptic_gain, a WanderAgent's genome
(`wander_persistence`) is fixed for a fly's entire lifetime, inherited
from its parent plus mutation at birth, and never modified by
`reinforce()` -- there is no synapse for a dopamine-gated rule to act on
here, and no credit-assignment problem to solve, because selection
itself is the mechanism: a fly whose inherited persistence happens to
search well survives and reproduces more, per the pillar contract in
wiki/colony.md (fixed reflex, learned response, or evolved structure --
never a hardcoded rule).

Deliberately NOT a connectome-derived circuit -- no spiking, no Circuit
instance, just a single evolved scalar and one bit of per-fly live state
(the direction currently being held).
"""

from __future__ import annotations

import numpy as np

from world.env import Action, Observation

MOVE_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

DEFAULT_WANDER_PERSISTENCE = 5.0  # mean ticks a fly holds one direction before re-picking
WANDER_PERSISTENCE_BOUNDS = (1.0, 20.0)  # never instant-jittery, never a straight line forever


class WanderAgent:
    def __init__(self, persistence: float = DEFAULT_WANDER_PERSISTENCE) -> None:
        self.persistence = float(np.clip(persistence, *WANDER_PERSISTENCE_BOUNDS))
        self.current_direction: Action | None = None

    def decide(self, obs: Observation, rng: np.random.Generator) -> Action | None:
        """Returns None whenever there's something to react to (`obs.nearby`
        non-empty) -- the caller falls through to the plasticity circuit's
        own decision, unchanged, exactly as it would today. Only when a fly
        perceives nothing at all does this return an actual movement:
        re-picked with probability `1 / persistence` each such tick, held
        otherwise, so a blind fly covers ground in bouts instead of
        jittering step to step.
        """
        if obs.nearby:
            return None
        if self.current_direction is None or rng.random() < 1.0 / self.persistence:
            self.current_direction = MOVE_ACTIONS[int(rng.integers(0, len(MOVE_ACTIONS)))]
        return self.current_direction
