"""Entity id/type_name (wiki/decisions.md #43).

A future API layer serializing world state for a human player to render
needs a stable id per world object (to track "this is the same tile
across ticks") and a real name (to pick a sprite/label) -- neither
existed on Item/Tile/Mob/Corpse before this entry, only on Fly.

What's pinned here: every spawn path assigns a unique id from one shared
counter (decisions.md #43 deliberately keeps it separate from Fly's own
`next_fly_id` space); `type_name` reflects the real registered type's
name. And, the boundary that actually matters: none of this ever reaches
a fly through `Observation`/`Percept` -- id/type_name are for a consumer
outside the fly's own anonymity contract (#22), never inside it.
"""

from __future__ import annotations

import pytest

from world.entities import Item, Tile
from world.env import DEFAULT_OWNER, Environment

OWNER = DEFAULT_OWNER
PERCEPT_FIELDS = {"attributes", "dx", "dy", "distance"}


@pytest.fixture
def env() -> Environment:
    return Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_spawn_rate=0.0,
        tile_spawn_rate=0.0,
        starting_energy=1000.0,
        seed=1,
    )


def test_item_carries_its_real_type_name(env):
    env.add_item_type("poison meat", "a smelly poisonous piece of meat", 4)
    item = env.spawn_of(Item, env.item_types[-1])
    assert item.type_name == "poison meat"


def test_tile_carries_its_real_type_name(env):
    env.add_tile_type("healing spring", "a warm healing spring", 3)
    tile = env.spawn_of(Tile, env.tile_types[-1])
    assert tile.type_name == "healing spring"


def test_mob_carries_its_real_type_name(env):
    env.add_mob_type("venomous spider mob", "damage", 3)
    mob = env.spawn_mob_of(env.mob_types[-1])
    assert mob.type_name == "venomous spider mob"


def test_built_in_food_carries_its_name_too(env):
    item = env.spawn_of(Item, env.food_type)
    assert item.type_name == "food"


def test_corpse_type_name_is_a_fixed_constant(env):
    fly = env.flies[0]
    fly.health = 0
    env.resolve_deaths()
    assert env.corpses[0].type_name == "corpse"


def test_ids_are_unique_and_shared_across_all_four_kinds(env):
    env.add_item_type("i", "a plain grey pebble", 3)
    env.add_tile_type("t", "a warm healing spring", 3)
    env.add_mob_type("m", "damage", 3)
    item = env.spawn_of(Item, env.item_types[-1])
    tile = env.spawn_of(Tile, env.tile_types[-1])
    mob = env.spawn_mob_of(env.mob_types[-1])
    env.flies[0].health = 0
    env.resolve_deaths()
    corpse = env.corpses[0]

    ids = [item.id, tile.id, mob.id, corpse.id]
    assert len(set(ids)) == len(ids)


def test_ids_reset_with_the_environment(env):
    env.add_item_type("i", "a plain grey pebble", 3)
    first = env.spawn_of(Item, env.item_types[-1])
    assert first.id == 0

    env.reset()
    env.add_item_type("i", "a plain grey pebble", 3)
    second = env.spawn_of(Item, env.item_types[-1])

    assert second.id == 0  # the counter starts over with the environment, same as next_fly_id does


# --- the boundary that actually matters: id/type_name never reach a fly ---

def test_id_and_type_name_never_leak_into_a_percept(env):
    env.add_item_type("poison meat", "a smelly poisonous piece of meat", 4)
    item = env.spawn_of(Item, env.item_types[-1])
    item.position = env.flies[0].position
    env.items = [item]

    percepts = env.observe(env.flies[0]).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == PERCEPT_FIELDS
    assert not hasattr(percepts[0], "id")
    assert not hasattr(percepts[0], "type_name")
