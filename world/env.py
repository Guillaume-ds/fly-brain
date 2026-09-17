"""Minimal grid-world survival environment for the fly.

Deliberately simple (V1): fixed-size bounded grid, a handful of food and
threat entities placed randomly at reset, no rendering, no open-world
generation yet -- those are explicit later steps, not part of this.

Design choices made here (flag if you'd rather have these different):
  - Food and threats both give a linear-falloff "gradient" signal that is
    exactly zero outside their radius -- the fly gets no information at all
    about something it hasn't come close enough to sense.
  - Picking up food = entering its radius (as agreed). It refills hunger
    to max and the food item is removed (no respawn in V1 -- food supply
    shrinking over an episode is itself part of the survival pressure).
  - Threat "contact" (death) is stricter than sensing: it only happens if
    the fly ends a move standing exactly on a threat's cell. The sense
    radius is a separate, larger range in which the fly gets a warning
    signal but is still safe -- this is what gives the escape circuit
    something to react to before it's too late.

Curriculum-friendly by construction: food/threats can each be switched off
via food_enabled/threats_enabled, so training/curriculum.py can build
different stages by configuring this one class rather than writing separate
environments.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .entities import Food, Position, Threat


class Action(enum.IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


_MOVES: dict[Action, tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


@dataclass
class Observation:
    food_signal: float  # 0 (out of range) .. 1 (right on top of it)
    food_dx: float  # unit direction to nearest in-range food; 0 if none
    food_dy: float
    threat_signal: float  # 0 (out of range) .. 1 (right on top of it)
    threat_dx: float  # unit direction to nearest in-range threat; 0 if none
    threat_dy: float
    hunger: float  # current_hunger / max_hunger -- always known, not sensed

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.food_signal, self.food_dx, self.food_dy,
                self.threat_signal, self.threat_dx, self.threat_dy,
                self.hunger,
            ],
            dtype=np.float32,
        )


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    cause: str | None = None  # "starved" | "threat" | "timeout" | None


class Environment:
    def __init__(
        self,
        grid_size: int = 20,
        max_hunger: int = 100,
        max_ticks: int = 300,
        num_food: int = 3,
        num_threats: int = 2,
        food_radius: float = 2.0,
        threat_radius: float = 3.0,
        food_enabled: bool = True,
        threats_enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.max_hunger = max_hunger
        self.max_ticks = max_ticks
        self.num_food = num_food if food_enabled else 0
        self.num_threats = num_threats if threats_enabled else 0
        self.food_radius = food_radius
        self.threat_radius = threat_radius
        self._rng = np.random.default_rng(seed)

        self.fly_pos: Position
        self.hunger: int
        self.tick: int
        self.food: list[Food]
        self.threats: list[Threat]
        self.reset()

    def _random_empty_cell(self, taken: set[tuple[int, int]]) -> Position:
        while True:
            x = int(self._rng.integers(0, self.grid_size))
            y = int(self._rng.integers(0, self.grid_size))
            if (x, y) not in taken:
                taken.add((x, y))
                return Position(x, y)

    def reset(self) -> Observation:
        self.tick = 0
        self.hunger = self.max_hunger
        taken: set[tuple[int, int]] = set()
        self.fly_pos = self._random_empty_cell(taken)
        self.food = [
            Food(self._random_empty_cell(taken), self.food_radius)
            for _ in range(self.num_food)
        ]
        self.threats = [
            Threat(self._random_empty_cell(taken), self.threat_radius)
            for _ in range(self.num_threats)
        ]
        return self._observe()

    def _nearest_in_range(self, positions: list[Position], radius: float):
        best_dist, best_pos = None, None
        for pos in positions:
            d = self.fly_pos.distance_to(pos)
            if d <= radius and (best_dist is None or d < best_dist):
                best_dist, best_pos = d, pos
        return best_dist, best_pos

    def _signal_and_direction(self, dist, pos, radius: float):
        if pos is None:
            return 0.0, 0.0, 0.0
        signal = 1.0 - dist / radius if radius > 0 else 1.0
        dx, dy = pos.x - self.fly_pos.x, pos.y - self.fly_pos.y
        norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        return signal, dx / norm, dy / norm

    def _observe(self) -> Observation:
        f_dist, f_pos = self._nearest_in_range(
            [f.position for f in self.food], self.food_radius
        )
        t_dist, t_pos = self._nearest_in_range(
            [t.position for t in self.threats], self.threat_radius
        )
        food_signal, food_dx, food_dy = self._signal_and_direction(f_dist, f_pos, self.food_radius)
        threat_signal, threat_dx, threat_dy = self._signal_and_direction(t_dist, t_pos, self.threat_radius)

        return Observation(
            food_signal=food_signal, food_dx=food_dx, food_dy=food_dy,
            threat_signal=threat_signal, threat_dx=threat_dx, threat_dy=threat_dy,
            hunger=self.hunger / self.max_hunger,
        )

    def step(self, action: Action) -> StepResult:
        dx, dy = _MOVES[Action(action)]
        self.fly_pos = Position(
            x=min(max(self.fly_pos.x + dx, 0), self.grid_size - 1),
            y=min(max(self.fly_pos.y + dy, 0), self.grid_size - 1),
        )
        self.tick += 1
        self.hunger -= 1

        for f in self.food:
            if self.fly_pos.distance_to(f.position) <= f.pickup_radius:
                self.hunger = self.max_hunger
                self.food.remove(f)
                break

        cause = None
        done = False
        if any(
            self.fly_pos.x == t.position.x and self.fly_pos.y == t.position.y
            for t in self.threats
        ):
            done, cause = True, "threat"
        elif self.hunger <= 0:
            done, cause = True, "starved"
        elif self.tick >= self.max_ticks:
            done, cause = True, "timeout"

        return StepResult(observation=self._observe(), reward=1.0, done=done, cause=cause)
