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
class Food:
    position: Position
    pickup_radius: float
    attributes: np.ndarray  # this instance's item vector (prototype + jitter); see world/items.py


@dataclass
class Threat:
    position: Position
    sense_radius: float
    attributes: np.ndarray  # this instance's item vector (prototype + jitter); see world/items.py


@dataclass
class Fly:
    id: int
    position: Position
    hunger: int
    health: int
    vulnerable_ticks_left: int = 0  # >0 while recovering from reproduction; forces STAY
    stuck_ticks: int = 0  # >0 while immobilized (e.g. a web-like item); forces STAY, see decisions.md #22 part 5
