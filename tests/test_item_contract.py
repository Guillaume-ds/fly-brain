"""The item contract (wiki/world.md).

The one test to re-run whenever item-related code changes. It does not
care *how* an item came to exist or what changed internally about it --
only that "item-ness" survives:

  (a) it carries a unit-norm attribute vector
  (b) a fly perceives it ONLY as an anonymous Percept -- no name, type,
      or id ever reaches Observation
  (c) its effect on a fly comes purely from the Result registry, never a
      hardcoded per-type constant

Parametrised over every item-producing pathway there is, so a newly
added one is covered by construction. `Threat` is the one documented
carve-out -- it satisfies (a) and (b) but deliberately not (c) -- and
that carve-out is pinned here too, so it can't silently widen.
"""

from __future__ import annotations

import numpy as np
import pytest

from world.entities import Item, Position, Threat
from world.env import Environment
from world.results import Channel, ResultConcept

PLAYER_ITEM_DESCRIPTION = "a smelly, poisonous piece of meat"
PERCEPT_FIELDS = {"attributes", "dx", "dy", "distance"}


@pytest.fixture
def env() -> Environment:
    """A quiet world: nothing spawns on its own, so each test places
    exactly the one entity it's about.
    """
    environment = Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        seed=1,
    )
    environment.add_item_type(PLAYER_ITEM_DESCRIPTION)
    return environment


def item_types(env: Environment):
    """Every pathway that can put an item in the world: the built-in
    food type and every player-created one. They go through the same
    spawn path precisely because they're the same kind of thing.
    """
    return env.spawnable_item_types


def place(env: Environment, entity) -> None:
    env.items, env.threats = [], []
    (env.threats if isinstance(entity, Threat) else env.items).append(entity)
    env.flies[0].position = Position(5, 5)


def spawned(env: Environment, item_type, entity_class=Item):
    entity = env.spawn_of(entity_class, item_type)
    entity.position = Position(5, 5)
    return entity


# --- (a) unit-norm attribute vector ------------------------------------

@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_carries_unit_norm_attributes(env, type_index):
    item = spawned(env, item_types(env)[type_index])
    assert item.attributes.shape == (env.encoder.output_dim,)
    assert np.linalg.norm(item.attributes) == pytest.approx(1.0)


def test_threat_also_carries_unit_norm_attributes(env):
    threat = spawned(env, env.threat_type, Threat)
    assert np.linalg.norm(threat.attributes) == pytest.approx(1.0)


# --- (b) perceived only as an anonymous Percept ------------------------

@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_is_perceived_anonymously(env, type_index):
    item = spawned(env, item_types(env)[type_index])
    place(env, item)

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == PERCEPT_FIELDS
    np.testing.assert_array_equal(percepts[0].attributes, item.attributes)


def test_threat_is_perceived_identically_to_an_item(env):
    """A fly must not be able to tell a threat from an item by the shape
    of what it receives -- that's the whole point of anonymous percepts.
    """
    place(env, spawned(env, env.threat_type, Threat))

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == PERCEPT_FIELDS


def test_nothing_outside_its_radius_is_perceived(env):
    item = spawned(env, item_types(env)[0])
    item.position = Position(5 + int(item.radius) + 2, 5)
    place(env, item)

    assert env.observe(env.flies[0]).nearby == []


# --- (c) effect comes from the Result registry -------------------------

@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_effect_matches_the_result_registry(env, type_index):
    """Not just "something changed" -- the change must be exactly what
    the Result registry says, which is what rules out a hardcoded
    per-type constant hiding somewhere.
    """
    from world.results import blend_deltas

    item = spawned(env, item_types(env)[type_index])
    place(env, item)
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 0

    expected = {
        channel: int(min(env.channel_limits[channel], max(0, round(getattr(fly, channel.value) + delta))))
        for channel, delta in blend_deltas(item.attributes, env.results).items()
    }
    env.resolve_item_pickup()

    for channel, value in expected.items():
        assert getattr(fly, channel.value) == value, f"{channel} did not match the registry"


@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_is_consumed_on_contact(env, type_index):
    place(env, spawned(env, item_types(env)[type_index]))

    env.resolve_item_pickup()

    assert env.items == []


def test_threat_is_the_documented_carve_out(env):
    """Threats deliberately do NOT satisfy (c): contact kills through
    determine_fly_death() instead of the Result registry, and a threat is
    never consumed. Pinned so the carve-out can't quietly widen -- see
    entities.Threat and wiki/world.md.
    """
    place(env, spawned(env, env.threat_type, Threat))
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 0

    env.resolve_item_pickup()

    assert (fly.hunger, fly.health, fly.stuck_ticks) == (50, 60, 0)
    assert len(env.threats) == 1
    assert env.determine_fly_death(fly) == "threat"


# --- the registry's own guard rail -------------------------------------

def test_result_on_an_unregistered_channel_raises(env):
    """A Result whose channel has no bound must fail loudly rather than
    be silently dropped -- the reason Channel is an enum (decisions.md
    #28).
    """
    env.results.append(
        ResultConcept("speed", env.encoder.encode("fast"), channel="speed", sign=1.0, scale=10.0)
    )

    with pytest.raises(KeyError):
        env.apply_result(env.flies[0], env.encoder.encode("a fast moving thing"))


def test_every_channel_has_a_limit_and_a_matching_fly_field(env):
    """Channel.value must name a real Fly attribute, and every channel
    must be bounded -- the link apply_result() relies on.
    """
    for channel in Channel:
        assert channel in env.channel_limits
        assert hasattr(env.flies[0], channel.value)
