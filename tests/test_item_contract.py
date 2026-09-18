"""The item contract (wiki/world.md).

The one test to re-run whenever item-related code changes. It does not
care *how* an item came to exist or what changed internally about it --
only that "item-ness" survives:

  (a) it carries a unit-norm attribute vector
  (b) a fly perceives it ONLY as an anonymous Percept -- no name, type,
      or id ever reaches Observation
  (c) its effect on a fly comes purely from the Result registry (scaled
      by `strength`'s magnitude only, decisions.md #32), never a
      hardcoded per-type constant

Parametrised over every item-producing pathway there is, so a newly
added one is covered by construction: the built-in food, a player-
created item, and a player-created tile all satisfy (a)/(b)/(c) --
tiles are exactly an item mechanism that never gets consumed
(decisions.md #31). `Mob` is the one documented carve-out -- it
satisfies (a) and (b) but deliberately not (c), its effect is authored
from `effect`/`strength` instead (decisions.md #30) -- and that
carve-out is pinned here too, so it can't silently widen.
"""

from __future__ import annotations

import numpy as np
import pytest

from world.entities import Item, Mob, Position, Tile
from world.env import Environment
from world.results import Channel, Effect, ResultConcept, blend_deltas, strength_magnitude

PLAYER_ITEM_DESCRIPTION = "a smelly, poisonous piece of meat"
PLAYER_TILE_DESCRIPTION = "a warm, healing spring"
ITEM_STRENGTH = 4
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
        starting_energy=1000.0,  # not what this file is about -- see test_creation_cost.py for the energy gate itself
        seed=1,
    )
    environment.add_item_type("player item", PLAYER_ITEM_DESCRIPTION, ITEM_STRENGTH)
    environment.add_tile_type("player tile", PLAYER_TILE_DESCRIPTION, ITEM_STRENGTH)
    return environment


def item_types(env: Environment):
    """Every pathway that can put an item in the world: the built-in
    food type and every player-created one. They go through the same
    spawn path precisely because they're the same kind of thing.
    """
    return env.spawnable_item_types


def tile_types(env: Environment):
    return env.spawnable_tile_types


def place(env: Environment, entity) -> None:
    env.items, env.tiles, env.mobs = [], [], []
    if isinstance(entity, Mob):
        env.mobs.append(entity)
    elif isinstance(entity, Tile):
        env.tiles.append(entity)
    else:
        env.items.append(entity)
    env.flies[0].position = Position(5, 5)


def spawned(env: Environment, item_type, entity_class=Item):
    entity = env.spawn_of(entity_class, item_type)
    entity.position = Position(5, 5)
    return entity


def spawned_mob(env: Environment, mob_type) -> Mob:
    mob = env.spawn_mob_of(mob_type)
    mob.position = Position(5, 5)
    return mob


# --- (a) unit-norm attribute vector ------------------------------------

@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_carries_unit_norm_attributes(env, type_index):
    item = spawned(env, item_types(env)[type_index])
    assert item.attributes.shape == (env.encoder.output_dim,)
    assert np.linalg.norm(item.attributes) == pytest.approx(1.0)


def test_tile_carries_unit_norm_attributes(env):
    tile = spawned(env, tile_types(env)[0], Tile)
    assert np.linalg.norm(tile.attributes) == pytest.approx(1.0)


def test_mob_also_carries_unit_norm_attributes(env):
    mob = spawned_mob(env, env.spider_type)
    assert np.linalg.norm(mob.attributes) == pytest.approx(1.0)


# --- (b) perceived only as an anonymous Percept ------------------------

@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_is_perceived_anonymously(env, type_index):
    item = spawned(env, item_types(env)[type_index])
    place(env, item)

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == PERCEPT_FIELDS
    np.testing.assert_array_equal(percepts[0].attributes, item.attributes)


def test_tile_is_perceived_identically_to_an_item(env):
    place(env, spawned(env, tile_types(env)[0], Tile))

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == PERCEPT_FIELDS


def test_mob_is_perceived_identically_to_an_item(env):
    """A fly must not be able to tell a mob from an item by the shape of
    what it receives -- that's the whole point of anonymous percepts.
    """
    place(env, spawned_mob(env, env.spider_type))

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == PERCEPT_FIELDS


def test_nothing_outside_the_sensing_radius_is_perceived(env):
    """Sensing and interaction are two different distances (decisions.md
    #39) -- this pins the *sensing* boundary, WORLD_SENSING_RADIUS, not
    an entity's own (much smaller) interaction radius.
    """
    from world.env import WORLD_SENSING_RADIUS

    item = spawned(env, item_types(env)[0])
    item.position = Position(5 + int(WORLD_SENSING_RADIUS) + 2, 5)
    place(env, item)

    assert env.observe(env.flies[0]).nearby == []


def test_something_well_outside_its_interaction_radius_is_still_sensed(env):
    """The positive case the split exists for: a fly can smell food long
    before it's close enough to eat it -- interaction radius no longer
    doubles as the sensing distance.
    """
    item = spawned(env, item_types(env)[0])
    item.position = Position(5 + int(item.radius) + 2, 5)  # well outside interaction radius
    place(env, item)

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1


# --- (c) effect comes from the Result registry -------------------------

@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_effect_matches_the_result_registry(env, type_index):
    """Not just "something changed" -- the change must be exactly what
    the Result registry says (scaled by strength's magnitude only,
    decisions.md #32), which is what rules out a hardcoded per-type
    constant hiding somewhere.
    """
    item = spawned(env, item_types(env)[type_index])
    place(env, item)
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 0

    magnitude = strength_magnitude(item.strength)
    expected = {
        channel: int(min(env.channel_limits[channel], max(0, round(getattr(fly, channel.value) + delta * magnitude))))
        for channel, delta in blend_deltas(item.attributes, env.results).items()
    }
    env.resolve_item_pickup()

    for channel, value in expected.items():
        assert getattr(fly, channel.value) == value, f"{channel} did not match the registry"


def test_tile_effect_matches_the_result_registry_scaled_per_tick(env):
    """Same registry, same strength scaling as an item, but only
    TILE_EFFECT_FRACTION of it per tick (decisions.md #31, #32).
    """
    from world.results import TILE_EFFECT_FRACTION

    tile = spawned(env, tile_types(env)[0], Tile)
    place(env, tile)
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 0

    magnitude = strength_magnitude(tile.strength) * TILE_EFFECT_FRACTION
    expected = {
        channel: int(min(env.channel_limits[channel], max(0, round(getattr(fly, channel.value) + delta * magnitude))))
        for channel, delta in blend_deltas(tile.attributes, env.results).items()
    }
    env.resolve_tile_effects()

    for channel, value in expected.items():
        assert getattr(fly, channel.value) == value, f"{channel} did not match the registry"


@pytest.mark.parametrize("type_index", [0, 1], ids=["built-in food", "player-created"])
def test_item_is_consumed_on_contact(env, type_index):
    place(env, spawned(env, item_types(env)[type_index]))

    env.resolve_item_pickup()

    assert env.items == []


def test_tile_is_never_consumed(env):
    place(env, spawned(env, tile_types(env)[0], Tile))

    env.resolve_tile_effects()
    env.resolve_tile_effects()

    assert len(env.tiles) == 1


def test_mob_is_the_documented_carve_out(env):
    """Mobs deliberately do NOT satisfy (c): the built-in spider kills
    through determine_fly_death() instead of the Result registry, and
    is never consumed. Pinned so the carve-out can't quietly widen --
    see entities.Mob and wiki/world.md.
    """
    place(env, spawned_mob(env, env.spider_type))
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 0

    env.resolve_item_pickup()
    env.resolve_tile_effects()

    assert (fly.hunger, fly.health, fly.stuck_ticks) == (50, 60, 0)
    assert len(env.mobs) == 1
    assert env.determine_fly_death(fly) == "threat"


def test_created_mob_effect_is_authored_not_derived_from_the_result_registry(env):
    """A player-created mob's effect must come from `effect`/`strength`
    directly (decisions.md #30) -- never blend_deltas() on its
    attribute vector, which drives perception only. Verified by giving
    it a description that would score strongly against `food` (so a
    Result-registry-derived effect would raise hunger) while its
    authored `effect` is `damage`: only the authored channel may move.
    """
    env.add_mob_type("suspicious berry", "damage", 5)
    mob_type = env.mob_types[0]
    place(env, spawned_mob(env, mob_type))
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 0

    env.resolve_mob_contact()

    assert fly.hunger == 50, "hunger moved -- effect leaked from the embedding, not the authored field"
    assert fly.health < 60, "authored damage effect did not apply"
    assert fly.stuck_ticks == 0


@pytest.mark.parametrize(
    "effect,channel,direction",
    [
        (Effect.HEAL, Channel.HEALTH, "up"),
        (Effect.DAMAGE, Channel.HEALTH, "down"),
        (Effect.FEED, Channel.HUNGER, "up"),
        (Effect.STARVE, Channel.HUNGER, "down"),
        (Effect.TRAP, Channel.STUCK_TICKS, "up"),
        (Effect.FREE, Channel.STUCK_TICKS, "down"),
    ],
)
def test_every_mob_effect_moves_only_its_own_channel_in_the_right_direction(env, effect, channel, direction):
    """A positive `strength` always helps the fly, whichever channel it
    targets (decisions.md #31) -- this is the per-channel sign table
    that makes that true, pinned against every Effect member.
    """
    env.add_mob_type("test mob", effect.value, 5)
    mob_type = env.mob_types[0]
    place(env, spawned_mob(env, mob_type))
    fly = env.flies[0]
    fly.hunger, fly.health, fly.stuck_ticks = 50, 60, 5

    before = getattr(fly, channel.value)
    env.resolve_mob_contact()
    after = getattr(fly, channel.value)

    for other in Channel:
        if other != channel:
            assert getattr(fly, other.value) == (50 if other == Channel.HUNGER else 60 if other == Channel.HEALTH else 5)
    if direction == "up":
        assert after > before
    else:
        assert after < before


def test_unsupported_mob_effect_is_silently_dropped(env):
    """An effect outside the real vocabulary has nowhere to land --
    the schema is what makes a request the engine can't fulfil produce
    nothing, rather than the engine growing to accommodate it
    (decisions.md #30)."""
    before = len(env.mob_types)

    env.add_mob_type("dragon", "breathe_fire", 5)

    assert len(env.mob_types) == before


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
        env.apply_result(env.flies[0], env.encoder.encode("a fast moving thing"), ITEM_STRENGTH)


def test_every_channel_has_a_limit_and_a_matching_fly_field(env):
    """Channel.value must name a real Fly attribute, and every channel
    must be bounded -- the link apply_result() relies on.
    """
    for channel in Channel:
        assert channel in env.channel_limits
        assert hasattr(env.flies[0], channel.value)
