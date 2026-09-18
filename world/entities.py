"""Plain data holders for what lives in the grid. No behavior here on
purpose -- placement, movement, and rules all live in env.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    `attributes` via the Result registry (world/results.py).

    `radius` is both how far away a fly can perceive it and how close a
    fly must be to pick it up -- one distance, one name.
    """

    position: Position
    radius: float
    attributes: np.ndarray


@dataclass
class Threat:
    """Deliberately NOT an Item (decisions.md #28). A threat is perceived
    exactly like one -- same anonymous Percept, same attribute vector --
    but it moves, it is never consumed, and contact kills through
    `Environment.determine_fly_death()` rather than through the Result
    registry. That asymmetry is a known, deliberate carve-out, not an
    oversight: routing lethality through Result-registry similarity
    would make a spider's deadliness depend on the encoder's judgment,
    which would silently change what the ES-trained escape circuit was
    trained against. See wiki/world.md for the open question of whether
    threats should eventually become items.
    """

    position: Position
    radius: float
    attributes: np.ndarray


@dataclass
class Fly:
    id: int
    position: Position
    hunger: int
    health: int
    vulnerable_ticks_left: int = 0  # >0 while recovering from reproduction; forces STAY
    stuck_ticks: int = 0  # >0 while immobilized (e.g. a web-like item); forces STAY, see decisions.md #22 part 5
