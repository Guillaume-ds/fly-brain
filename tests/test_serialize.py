"""server/serialize.py -- pure wire-format functions (wiki/decisions.md #44).

No FastAPI, no WebSocket, no async: these are plain functions over
Environment/ColonyStepResult, tested the same way as everything else in
this codebase.
"""

from __future__ import annotations

import pytest

from server.serialize import serialize_tick, serialize_world_init
from world.entities import Item, Tile
from world.env import Action, DEFAULT_OWNER, Environment
from world.results import Effect

OWNER = DEFAULT_OWNER


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


def test_world_init_includes_built_in_and_player_created_types(env):
    env.add_item_type("poison meat", "a smelly poisonous piece of meat", 4)
    env.add_mob_type("venomous spider mob", "damage", 3)

    data = serialize_world_init(env)

    assert data["grid_size"] == env.grid_size
    item_names = {t["name"] for t in data["item_types"]}
    assert item_names == {"food", "poison meat"}
    mob_names = {t["name"] for t in data["mob_types"]}
    assert mob_names == {"spider", "venomous spider mob"}
    created = next(t for t in data["mob_types"] if t["name"] == "venomous spider mob")
    assert created["effect"] == Effect.DAMAGE.value
    spider = next(t for t in data["mob_types"] if t["name"] == "spider")
    assert spider["effect"] is None  # the built-in spider has no authored effect


def test_world_init_never_exposes_a_percept_shaped_thing(env):
    """The wire format is built from the real object graph, never a
    fly's own Percept -- there should be no attribute vectors leaking
    into JSON-bound data at all.
    """
    data = serialize_world_init(env)
    for item_type in data["item_types"]:
        assert "attributes" not in item_type


def test_tick_includes_real_positions_and_stats(env):
    fly = env.flies[0]
    result = env.step({fly.id: Action.STAY})

    data = serialize_tick(env, result)

    assert data["tick"] == 1
    assert len(data["flies"]) == 1
    payload = data["flies"][0]
    assert payload["id"] == fly.id
    assert payload["x"] == fly.position.x
    assert payload["y"] == fly.position.y
    assert payload["hunger"] == fly.hunger
    assert payload["owner"] == OWNER


def test_tick_serializes_items_tiles_corpses_with_real_names(env):
    env.add_item_type("poison meat", "a smelly poisonous piece of meat", 4)
    env.add_tile_type("healing spring", "a warm healing spring", 3)
    item = env.spawn_of(Item, env.item_types[-1])
    tile = env.spawn_of(Tile, env.tile_types[-1])
    env.items, env.tiles = [item], [tile]
    env.flies[0].health = 0
    env.resolve_deaths()  # spawns a corpse

    data = serialize_tick(env, env.step({}))

    item_payload = data["items"][0]
    assert item_payload["type_name"] == "poison meat"
    assert item_payload["x"] == item.position.x
    tile_payload = data["tiles"][0]
    assert tile_payload["type_name"] == "healing spring"
    corpse_payload = data["corpses"][0]
    assert corpse_payload["type_name"] == "corpse"


def test_tick_events_reflect_deaths_births_and_extinction(env):
    fly = env.flies[0]
    fly.health = 0

    result = env.step({fly.id: Action.STAY})
    data = serialize_tick(env, result)

    assert data["events"]["deaths"] == [{"fly_id": fly.id, "cause": "damage"}]
    assert data["events"]["births"] == []
    assert data["events"]["colony_extinct"] == {OWNER: True}
