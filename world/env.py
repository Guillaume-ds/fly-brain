"""Grid-world survival environment for the fly.

Food and threats spawn continuously and threats wander (see
wiki/decisions.md #14, #15 for why -- a static placement is permanently
dodgeable). Exposes a Gym-style reset()/step() interface; food_enabled/
threats_enabled let training/curriculum.py configure different stages
from this one class. increase_/decrease_spider_rate and
increase_/decrease_food_rate are director/'s control surface.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .entities import Food, Position, Threat

SPAWN_RATE_BOUNDS = (0.0, 0.2)  # min/max per-tick spawn probability
SPAWN_RATE_STEP = 0.02  # fixed nudge used by increase_*/decrease_*


class Action(enum.IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


MOVES: dict[Action, tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


@dataclass
class SensorReading:
    signal: float  # 0 (out of range) .. 1 (right on top of it)
    dx: float  # unit direction to the sensed entity; 0 if out of range
    dy: float


@dataclass
class Observation:
    food_signal: float
    food_dx: float
    food_dy: float
    threat_signal: float
    threat_dx: float
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
        spider_spawn_rate: float = 0.03,
        food_spawn_rate: float = 0.03,
        max_spiders: int = 10,
        max_food: int = 10,
        food_radius: float = 2.0,
        threat_radius: float = 3.0,
        threat_move_probability: float = 0.3,
        food_enabled: bool = True,
        threats_enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.max_hunger = max_hunger
        self.max_ticks = max_ticks
        self.spider_spawn_rate = clamp_rate(spider_spawn_rate if threats_enabled else 0.0)
        self.food_spawn_rate = clamp_rate(food_spawn_rate if food_enabled else 0.0)
        self.max_spiders = max_spiders
        self.max_food = max_food
        self.food_radius = food_radius
        self.threat_radius = threat_radius
        self.threat_move_probability = threat_move_probability
        self.rng = np.random.default_rng(seed)

        self.fly_pos: Position
        self.hunger: int
        self.tick: int
        self.food: list[Food]
        self.threats: list[Threat]
        self.reset()

    # director-facing controls -- see wiki/decisions.md #14 for why these
    # are fixed, bounded nudges rather than taking a magnitude argument.
    def increase_spider_rate(self) -> None:
        self.spider_spawn_rate = clamp_rate(self.spider_spawn_rate + SPAWN_RATE_STEP)

    def decrease_spider_rate(self) -> None:
        self.spider_spawn_rate = clamp_rate(self.spider_spawn_rate - SPAWN_RATE_STEP)

    def increase_food_rate(self) -> None:
        self.food_spawn_rate = clamp_rate(self.food_spawn_rate + SPAWN_RATE_STEP)

    def decrease_food_rate(self) -> None:
        self.food_spawn_rate = clamp_rate(self.food_spawn_rate - SPAWN_RATE_STEP)

    def reset(self) -> Observation:
        self.tick = 0
        self.hunger = self.max_hunger
        self.fly_pos = self.random_empty_cell(set())
        self.food = []
        self.threats = []
        return self.observe()

    def step(self, action: Action) -> StepResult:
        self.move_fly(Action(action))
        self.tick += 1
        self.hunger -= 1
        self.move_threats()
        self.spawn_entities()
        self.resolve_food_pickup()

        cause = self.determine_episode_end()
        return StepResult(observation=self.observe(), reward=1.0, done=cause is not None, cause=cause)

    def clamp_to_grid(self, x: int, y: int) -> Position:
        return Position(x=min(max(x, 0), self.grid_size - 1), y=min(max(y, 0), self.grid_size - 1))

    def random_empty_cell(self, taken: set[tuple[int, int]]) -> Position:
        while True:
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            if (x, y) not in taken:
                taken.add((x, y))
                return Position(x, y)

    def occupied_cells(self) -> set[tuple[int, int]]:
        cells = {(self.fly_pos.x, self.fly_pos.y)}
        cells.update((f.position.x, f.position.y) for f in self.food)
        cells.update((t.position.x, t.position.y) for t in self.threats)
        return cells

    def move_fly(self, action: Action) -> None:
        dx, dy = MOVES[action]
        self.fly_pos = self.clamp_to_grid(self.fly_pos.x + dx, self.fly_pos.y + dy)

    def move_threats(self) -> None:
        possible_steps = list(MOVES.values())
        for threat in self.threats:
            if self.rng.random() < self.threat_move_probability:
                dx, dy = possible_steps[int(self.rng.integers(0, len(possible_steps)))]
                threat.position = self.clamp_to_grid(threat.position.x + dx, threat.position.y + dy)

    def spawn_entities(self) -> None:
        if len(self.threats) < self.max_spiders and self.rng.random() < self.spider_spawn_rate:
            self.threats.append(Threat(self.random_empty_cell(self.occupied_cells()), self.threat_radius))
        if len(self.food) < self.max_food and self.rng.random() < self.food_spawn_rate:
            self.food.append(Food(self.random_empty_cell(self.occupied_cells()), self.food_radius))

    def resolve_food_pickup(self) -> None:
        for food in self.food:
            if self.fly_pos.distance_to(food.position) <= food.pickup_radius:
                self.hunger = self.max_hunger
                self.food.remove(food)
                break

    def determine_episode_end(self) -> str | None:
        fly_on_a_threat = any(
            self.fly_pos.x == t.position.x and self.fly_pos.y == t.position.y
            for t in self.threats
        )
        if fly_on_a_threat:
            return "threat"
        if self.hunger <= 0:
            return "starved"
        if self.tick >= self.max_ticks:
            return "timeout"
        return None

    def sense_nearest(self, positions: list[Position], radius: float) -> SensorReading:
        nearest_dist, nearest_pos = None, None
        for pos in positions:
            dist = self.fly_pos.distance_to(pos)
            if dist <= radius and (nearest_dist is None or dist < nearest_dist):
                nearest_dist, nearest_pos = dist, pos

        if nearest_pos is None:
            return SensorReading(signal=0.0, dx=0.0, dy=0.0)

        signal = 1.0 - nearest_dist / radius if radius > 0 else 1.0
        dx, dy = nearest_pos.x - self.fly_pos.x, nearest_pos.y - self.fly_pos.y
        norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        return SensorReading(signal=signal, dx=dx / norm, dy=dy / norm)

    def observe(self) -> Observation:
        food = self.sense_nearest([f.position for f in self.food], self.food_radius)
        threat = self.sense_nearest([t.position for t in self.threats], self.threat_radius)
        return Observation(
            food_signal=food.signal, food_dx=food.dx, food_dy=food.dy,
            threat_signal=threat.signal, threat_dx=threat.dx, threat_dy=threat.dy,
            hunger=self.hunger / self.max_hunger,
        )


def clamp_rate(rate: float) -> float:
    return min(max(rate, SPAWN_RATE_BOUNDS[0]), SPAWN_RATE_BOUNDS[1])
