"""Plain data holders for what lives in the grid. No behavior here on
purpose -- placement, movement, and rules all live in env.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import Effect


@dataclass
class Position:
    x: int
    y: int

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class Item:
    """A live instance of a registered ItemType (world/items.py) --
    generic, single-use, stationary. There is exactly one item entity in
    this world: food is an ItemType like any other, not a separate class
    (decisions.md #28). Nothing about an item's behavior is configured
    per instance; what happens on contact comes entirely from
    `attributes` via the Result registry (world/results.py), scaled by
    `strength` (decisions.md #32) -- `strength` sets overall magnitude
    only, never which channel moves or in which direction.

    `radius` is the interaction distance only -- how close a fly must be
    to pick it up. Since decisions.md #39, sensing distance is a separate,
    much larger, shared constant (`Environment.WORLD_SENSING_RADIUS`);
    `radius` no longer governs perception at all.
    """

    position: Position
    radius: float
    attributes: np.ndarray
    strength: int


@dataclass
class Tile:
    """An `Item` in every way except one: never consumed. A live instance
    of a registered ItemType (world/items.py) used as an area effect
    (decisions.md #31, #32) -- same encoder-driven attributes, same
    Result-registry effect, same `strength` magnitude scaling -- but its
    effect re-applies every tick a fly remains within `radius`, at a
    fraction of what a one-shot item pickup would give
    (`Environment.resolve_tile_effects()`), rather than being removed on
    contact.
    """

    position: Position
    radius: float
    attributes: np.ndarray
    strength: int


@dataclass
class Mob:
    """A live instance of a registered MobType (world/items.py) -- moves,
    is never consumed. Renamed from `Threat` (decisions.md #31): the
    class stopped meaning "always dangerous" the moment it could be
    beneficial.

    `attributes` drives perception only, via the same anonymous Percept
    every other entity uses -- never a mob's actual effect on a fly.
    That comes entirely from `effect`/`strength`, authored and read
    directly, never derived from `attributes` (decisions.md #30): the
    encoder can't be trusted with anything that could invalidate the
    frozen escape circuit's training, the way it safely can be for an
    item or a tile.

    `effect`/`strength` are `None` only for the one built-in case: the
    spider, which still kills unconditionally on contact
    (`Environment.determine_fly_death()`), exactly as before this
    entry -- unrelated to anything a player can create. A player-created
    mob always has both set, and its contact effect is graded per tick
    of contact, not instant (decisions.md #31) -- see
    `Environment.resolve_mob_contact()`.
    """

    position: Position
    radius: float
    attributes: np.ndarray
    effect: Effect | None = None
    strength: int | None = None


@dataclass
class Corpse:
    """What's left where a fly died -- from any cause (combat, starvation,
    a mob), not just a kill (decisions.md #41). Structurally an item in
    every way that matters: perceived the same anonymous way, eaten once
    by whichever fly (any owner) reaches it first, then gone. The one
    difference from a real `Item`: `hunger_value` is authored per
    instance from the dead fly's own remaining hunger at death, never
    derived from `attributes` via the Result registry -- there's no
    description to derive it from, and the whole point is that it's
    exactly what that fly had left, not an encoder's guess at it.
    `attributes` exists purely so a corpse is perceivable like anything
    else; it carries no relationship to `hunger_value`.
    """

    position: Position
    radius: float
    attributes: np.ndarray
    hunger_value: float


@dataclass
class Fly:
    id: int
    position: Position
    hunger: int
    health: int
    owner: str = "player"  # decisions.md #34 -- which colony this fly belongs to, inherited by offspring; must match env.DEFAULT_OWNER (can't import it here without a cycle)
    vulnerable_ticks_left: int = 0  # >0 while recovering from reproduction; forces STAY
    stuck_ticks: int = 0  # >0 while immobilized (e.g. a web-like item); forces STAY, see decisions.md #22 part 5
