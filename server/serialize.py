"""Wire-format serialization (wiki/decisions.md #44) -- pure functions,
no I/O, turning `Environment`/`ColonyStepResult` into the JSON shapes
agreed for the WebSocket contract. Built from `Environment`'s real
object graph, never from a fly's own anonymous `Percept`/`Observation`
-- a human player watching the game isn't bound by decisions.md #22's
anonymity contract, so these functions use the real `type_name`/`id`
fields added in #43 specifically for this purpose.

Kept separate from `server/app.py` on purpose: no FastAPI/WebSocket
import here at all, so this is testable with plain pytest, the same way
everything else in this codebase is.
"""

from __future__ import annotations

from typing import Any

from world.entities import Corpse, Fly, Item, Mob, Tile
from world.env import ColonyStepResult, Environment
from world.items import ItemType, MobType

WorldEntity = Item | Tile | Mob | Corpse


def serialize_item_type(item_type: ItemType) -> dict[str, Any]:
    return {"name": item_type.name, "description": item_type.description, "strength": item_type.strength}


def serialize_mob_type(mob_type: MobType) -> dict[str, Any]:
    return {
        "name": mob_type.name,
        "description": mob_type.description,
        "strength": mob_type.strength,
        "effect": mob_type.effect.value if mob_type.effect is not None else None,
    }


def serialize_world_init(env: Environment) -> dict[str, Any]:
    """Sent once per connection -- everything that doesn't change tick to
    tick. `item_types`/`tile_types`/`mob_types` include the built-in food/
    spider alongside any player-created ones, same list `director/`'s
    registry draws on -- not a fly-facing list, so there's no anonymity
    concern in naming them here.
    """
    return {
        "grid_size": env.grid_size,
        "max_hunger": env.max_hunger,
        "max_health": env.max_health,
        "max_stuck_ticks": env.max_stuck_ticks,
        "owners": list(env.owners),
        "item_types": [serialize_item_type(t) for t in (env.food_type, *env.item_types)],
        "tile_types": [serialize_item_type(t) for t in env.tile_types],
        "mob_types": [serialize_mob_type(t) for t in (env.spider_type, *env.mob_types)],
    }


def serialize_fly(fly: Fly) -> dict[str, Any]:
    return {
        "id": fly.id,
        "x": fly.position.x,
        "y": fly.position.y,
        "hunger": fly.hunger,
        "health": fly.health,
        "owner": fly.owner,
        "stuck": fly.stuck_ticks > 0,
        "vulnerable": fly.vulnerable_ticks_left > 0,
    }


def serialize_entity(entity: WorldEntity) -> dict[str, Any]:
    """Item/Tile/Mob/Corpse are serialized identically -- id, real type
    name (decisions.md #43), position. The client tells them apart by
    which array they arrived in (`items`/`tiles`/`mobs`/`corpses`), not
    by any field on the entity itself.
    """
    return {"id": entity.id, "type_name": entity.type_name, "x": entity.position.x, "y": entity.position.y}


def serialize_tick(env: Environment, result: ColonyStepResult) -> dict[str, Any]:
    """The main per-tick broadcast -- full state, not a diff (deliberate
    for a first iteration: at this grid size and entity count, full
    state is simpler to reason about on both ends than tracking deltas,
    and it can't ever drift out of sync with a client that missed a
    message).
    """
    return {
        "tick": env.tick,
        "flies": [serialize_fly(fly) for fly in env.flies],
        "items": [serialize_entity(item) for item in env.items],
        "tiles": [serialize_entity(tile) for tile in env.tiles],
        "mobs": [serialize_entity(mob) for mob in env.mobs],
        "corpses": [serialize_entity(corpse) for corpse in env.corpses],
        "events": {
            "deaths": [{"fly_id": fly_id, "cause": cause} for fly_id, cause in result.deaths.items()],
            "births": [{"new_id": new_id, "parent_id": parent_id} for new_id, parent_id in result.births.items()],
            "colony_extinct": dict(result.colony_extinct),
        },
    }
