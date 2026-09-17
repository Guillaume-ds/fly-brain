"""Plain data holders for what lives in the grid. No behavior here on
purpose -- placement, movement, and rules all live in env.py.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class Threat:
    position: Position
    sense_radius: float


@dataclass
class Fly:
    id: int
    position: Position
    hunger: int
    vulnerable_ticks_left: int = 0  # >0 while recovering from reproduction; forces STAY
