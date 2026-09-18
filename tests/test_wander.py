"""Evolved wander behavior (wiki/decisions.md #40).

Before this entry, a fly with nothing perceived did nothing but
Action.STAY, forever (PlasticityAgent._valence_to_action() returns STAY
whenever obs.nearby is empty, and there was no fallback movement
anywhere in the codebase) -- the "movement" half of the exploration-
bootstrapping gap decisions.md #39 left open after fixing the "sensing"
half. WanderAgent (fly_brain/wander.py) closes it: a fixed, evolved
movement reflex that only ever fires when there's nothing to react to,
inherited and mutated at birth exactly like escape gains, never touched
by reinforce().
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_brain.wander import (
    DEFAULT_WANDER_PERSISTENCE,
    MOVE_ACTIONS,
    WANDER_PERSISTENCE_BOUNDS,
    WanderAgent,
)
from world.env import Action, DEFAULT_OWNER, Environment, Observation, Percept

OWNER = DEFAULT_OWNER


class _FixedRNG:
    """A stand-in for np.random.Generator that always re-picks (random()
    below any 1/persistence threshold) and cycles through a fixed
    sequence of direction indices, so direction changes are deterministic
    and observable rather than a statistical guess.
    """

    def __init__(self, indices: list[int]) -> None:
        self._indices = indices
        self._i = 0

    def random(self) -> float:
        return 0.0  # always below 1/persistence -- forces a re-pick every call

    def integers(self, low: int, high: int) -> int:
        value = self._indices[self._i % len(self._indices)]
        self._i += 1
        return value


class _NeverRepickRNG:
    def random(self) -> float:
        return 1.0  # always above 1/persistence -- never triggers a re-pick

    def integers(self, low: int, high: int) -> int:
        raise AssertionError("integers() should not be called once a direction is already held")


def make_obs(nearby: list[Percept]) -> Observation:
    return Observation(nearby=nearby, hunger=0.5)


SOME_PERCEPT = Percept(attributes=np.ones(4), dx=1.0, dy=0.0, distance=1.0)


# --- WanderAgent unit behavior --------------------------------------------

def test_returns_none_when_something_is_perceived():
    agent = WanderAgent()
    action = agent.decide(make_obs([SOME_PERCEPT]), _FixedRNG([0]))
    assert action is None


def test_returns_a_real_movement_when_nothing_is_perceived():
    agent = WanderAgent()
    action = agent.decide(make_obs([]), _FixedRNG([0]))
    assert action in MOVE_ACTIONS
    assert action != Action.STAY


def test_direction_is_held_once_picked_when_rng_never_triggers_a_repick():
    agent = WanderAgent(persistence=5.0)
    rng = _FixedRNG([1])  # only used for the very first pick
    first = agent.decide(make_obs([]), rng)

    held_rng = _NeverRepickRNG()
    second = agent.decide(make_obs([]), held_rng)
    third = agent.decide(make_obs([]), held_rng)

    assert first == second == third


def test_direction_can_change_when_rng_always_triggers_a_repick():
    agent = WanderAgent(persistence=5.0)
    rng = _FixedRNG([0, 1, 2, 3])  # UP, DOWN, LEFT, RIGHT in turn

    actions = [agent.decide(make_obs([]), rng) for _ in range(4)]

    assert actions == list(MOVE_ACTIONS)


@pytest.mark.parametrize("raw_persistence", [-5.0, 0.0, 999.0])
def test_persistence_is_clamped_to_bounds(raw_persistence):
    agent = WanderAgent(persistence=raw_persistence)
    lo, hi = WANDER_PERSISTENCE_BOUNDS
    assert lo <= agent.persistence <= hi


def test_default_persistence_is_used_when_unspecified():
    agent = WanderAgent()
    assert agent.persistence == DEFAULT_WANDER_PERSISTENCE


# --- Colony integration ---------------------------------------------------

@pytest.fixture(scope="module")
def templates():
    from fly_brain.agent import build_escape_template
    from fly_brain.plasticity import build_plasticity_template

    env = Environment(initial_population=1, seed=1)
    encoder = env.encoder
    return build_escape_template(encoder), build_plasticity_template(encoder)


def make_colony(templates, **env_kwargs):
    from game.colony import Colony

    escape_template, plasticity_template = templates
    env = Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_spawn_rate=0.0,
        tile_spawn_rate=0.0,
        starting_energy=1000.0,
        seed=3,
        **env_kwargs,
    )
    gains = np.ones(len(escape_template.blueprint.edges))
    return Colony(env, escape_template, plasticity_template, gains, seed=3)


def test_a_blind_fly_now_moves_instead_of_freezing(templates):
    """The regression this entry exists to fix: an empty world (nothing
    to perceive, ever) used to leave a fly permanently at Action.STAY.
    """
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    start_position = colony.env.flies[0].position

    moved = False
    for _ in range(15):
        colony.step()
        fly = next((f for f in colony.env.flies if f.id == fly_id), None)
        if fly is None:
            pytest.skip("fly died before moving -- unrelated to wander itself")
        if fly.position != start_position:
            moved = True
            break

    assert moved


def test_escape_still_overrides_wander(templates, monkeypatch):
    """Freeze+override precedence (decisions.md #5/#22) must still hold:
    the escape reflex wins even on a tick where wander would otherwise
    fire (nothing perceived).
    """
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    monkeypatch.setattr(colony.escape_agents[fly_id], "decide", lambda obs: Action.LEFT)
    monkeypatch.setattr(colony.wander_agents[fly_id], "decide", lambda obs, rng: Action.RIGHT)

    start_x = colony.env.flies[0].position.x
    colony.step()
    fly = next(f for f in colony.env.flies if f.id == fly_id)

    assert fly.position.x == start_x - 1  # LEFT, not wander's RIGHT


def test_every_fly_gets_a_wander_agent_including_newborns(templates):
    colony = make_colony(templates, max_population=10, base_reproduction_rate=1.0)
    colony.add_colony(OWNER, 1, np.ones(len(colony.escape_template.blueprint.edges)))
    for fly in colony.env.flies:
        fly.hunger = colony.env.max_hunger

    class _AlwaysReproduces:
        def __init__(self, real_rng):
            self._real = real_rng

        def random(self):
            return 0.0

        def __getattr__(self, name):
            return getattr(self._real, name)

    colony.env.rng = _AlwaysReproduces(colony.env.rng)

    result = colony.step()

    assert result.births
    for new_id in result.births:
        assert new_id in colony.wander_agents


def test_wander_persistence_inherits_from_parent_with_mutation(templates):
    colony = make_colony(templates, max_population=10, base_reproduction_rate=1.0)
    colony.wander_mutation_sigma = 0.0  # deterministic: rng.normal(0, 0) == 0.0
    colony.add_colony(OWNER, 1, np.ones(len(colony.escape_template.blueprint.edges)))
    parent_id = next(iter(colony.wander_agents))
    colony.wander_agents[parent_id].persistence = 9.0
    for fly in colony.env.flies:
        fly.hunger = colony.env.max_hunger

    class _AlwaysReproduces:
        def __init__(self, real_rng):
            self._real = real_rng

        def random(self):
            return 0.0

        def __getattr__(self, name):
            return getattr(self._real, name)

    colony.env.rng = _AlwaysReproduces(colony.env.rng)

    result = colony.step()
    assert result.births
    for new_id, parent_id in result.births.items():
        assert colony.wander_agents[new_id].persistence == colony.wander_agents[parent_id].persistence


def test_wander_agent_removed_on_death(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.env.flies[0].health = 0

    result = colony.step()

    assert fly_id in result.deaths
    assert fly_id not in colony.wander_agents
