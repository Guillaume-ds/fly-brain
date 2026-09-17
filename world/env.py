"""Grid-world survival environment for a fly colony.

Food and threats spawn continuously and threats wander (see
wiki/decisions.md #14, #15 -- a static placement is permanently
dodgeable). Multiple flies share one world and can reproduce (see
wiki/decisions.md #20). A population of 1 makes reproduction
structurally impossible (mate_availability is exactly 0 at population 1)
-- that's exactly the single-fly curriculum-training case, so
training/ uses this same class rather than a separate one.

increase_/decrease_spider_rate and increase_/decrease_food_rate are
director/'s control surface.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .entities import Fly, Food, Position, Threat

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
class ColonyStepResult:
    observations: dict[int, Observation]  # fly id -> observation, alive flies only
    deaths: dict[int, str]  # fly id -> cause ("threat" | "starved")
    births: dict[int, int]  # new fly id -> parent fly id
    colony_extinct: bool
    timed_out: bool


class Environment:
    def __init__(
        self,
        grid_size: int = 20,
        max_hunger: int = 100,
        max_ticks: int = 300,
        initial_population: int = 1,
        spider_spawn_rate: float = 0.03,
        food_spawn_rate: float = 0.03,
        max_spiders: int = 10,
        max_food: int = 10,
        food_radius: float = 2.0,
        threat_radius: float = 3.0,
        threat_move_probability: float = 0.3,
        max_population: int = 20,
        reproduction_hunger_threshold: float = 0.8,
        base_reproduction_rate: float = 0.05,
        max_mate_availability: float = 0.8,
        reproduction_hunger_cost_fraction: float = 0.4,
        reproduction_vulnerability_ticks: int = 3,
        offspring_spawn_radius: int = 2,
        food_enabled: bool = True,
        threats_enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.max_hunger = max_hunger
        self.max_ticks = max_ticks
        self.initial_population = initial_population
        self.spider_spawn_rate = clamp_rate(spider_spawn_rate if threats_enabled else 0.0)
        self.food_spawn_rate = clamp_rate(food_spawn_rate if food_enabled else 0.0)
        self.max_spiders = max_spiders
        self.max_food = max_food
        self.food_radius = food_radius
        self.threat_radius = threat_radius
        self.threat_move_probability = threat_move_probability
        self.max_population = max_population
        self.reproduction_hunger_threshold = reproduction_hunger_threshold
        self.base_reproduction_rate = base_reproduction_rate
        self.max_mate_availability = max_mate_availability
        self.reproduction_hunger_cost_fraction = reproduction_hunger_cost_fraction
        self.reproduction_vulnerability_ticks = reproduction_vulnerability_ticks
        self.offspring_spawn_radius = offspring_spawn_radius
        self.rng = np.random.default_rng(seed)

        self.flies: list[Fly]
        self.food: list[Food]
        self.threats: list[Threat]
        self.tick: int
        self.next_fly_id: int
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

    def reset(self) -> dict[int, Observation]:
        self.tick = 0
        self.next_fly_id = 0
        self.food = []
        self.threats = []
        self.flies = []
        for _ in range(self.initial_population):
            self.flies.append(self.spawn_fly(self.random_empty_cell(self.occupied_cells())))
        return {fly.id: self.observe(fly) for fly in self.flies}

    def spawn_fly(self, position: Position) -> Fly:
        fly = Fly(id=self.next_fly_id, position=position, hunger=self.max_hunger)
        self.next_fly_id += 1
        return fly

    def step(self, actions: dict[int, Action]) -> ColonyStepResult:
        missing = [fly.id for fly in self.flies if fly.id not in actions]
        if missing:
            raise ValueError(f"missing action for living flies: {missing}")

        self.move_flies(actions)
        self.tick += 1
        for fly in self.flies:
            fly.hunger -= 1
        self.move_threats()
        self.spawn_entities()
        self.resolve_food_pickup()
        births = self.resolve_reproduction()
        deaths = self.resolve_deaths()

        return ColonyStepResult(
            observations={fly.id: self.observe(fly) for fly in self.flies},
            deaths=deaths,
            births=births,
            colony_extinct=not self.flies,
            timed_out=self.tick >= self.max_ticks,
        )

    def clamp_to_grid(self, x: int, y: int) -> Position:
        return Position(x=min(max(x, 0), self.grid_size - 1), y=min(max(y, 0), self.grid_size - 1))

    def random_empty_cell(self, taken: set[tuple[int, int]]) -> Position:
        while True:
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            if (x, y) not in taken:
                taken.add((x, y))
                return Position(x, y)

    def random_nearby_empty_cell(self, center: Position, radius: int, attempts: int = 10) -> Position:
        taken = self.occupied_cells()
        for _ in range(attempts):
            x = center.x + int(self.rng.integers(-radius, radius + 1))
            y = center.y + int(self.rng.integers(-radius, radius + 1))
            x, y = min(max(x, 0), self.grid_size - 1), min(max(y, 0), self.grid_size - 1)
            if (x, y) not in taken:
                return Position(x, y)
        return self.random_empty_cell(taken)

    def occupied_cells(self) -> set[tuple[int, int]]:
        cells = {(fly.position.x, fly.position.y) for fly in self.flies}
        cells.update((f.position.x, f.position.y) for f in self.food)
        cells.update((t.position.x, t.position.y) for t in self.threats)
        return cells

    def move_flies(self, actions: dict[int, Action]) -> None:
        for fly in self.flies:
            if fly.vulnerable_ticks_left > 0:
                fly.vulnerable_ticks_left -= 1
                action = Action.STAY
            else:
                action = actions[fly.id]
            dx, dy = MOVES[action]
            fly.position = self.clamp_to_grid(fly.position.x + dx, fly.position.y + dy)

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
        for fly in self.flies:
            for food in self.food:
                if fly.position.distance_to(food.position) <= food.pickup_radius:
                    fly.hunger = self.max_hunger
                    self.food.remove(food)
                    break

    def resolve_reproduction(self) -> dict[int, int]:
        """See wiki/decisions.md #20 for the formula and why it's shaped
        this way (zero at population 1, never certain even at capacity).
        """
        births: dict[int, int] = {}
        starting_population = len(self.flies)
        if starting_population >= self.max_population or self.max_population <= 1:
            return births

        mate_availability = min(
            self.max_mate_availability,
            (starting_population - 1) / (self.max_population - 1),
        )
        probability = self.base_reproduction_rate * mate_availability

        for fly in list(self.flies):
            if len(self.flies) >= self.max_population:
                break
            if fly.hunger / self.max_hunger <= self.reproduction_hunger_threshold:
                continue
            if self.rng.random() >= probability:
                continue

            fly.hunger = int(fly.hunger * (1 - self.reproduction_hunger_cost_fraction))
            fly.vulnerable_ticks_left = self.reproduction_vulnerability_ticks

            offspring = self.spawn_fly(self.random_nearby_empty_cell(fly.position, self.offspring_spawn_radius))
            self.flies.append(offspring)
            births[offspring.id] = fly.id

        return births

    def resolve_deaths(self) -> dict[int, str]:
        deaths: dict[int, str] = {}
        survivors = []
        for fly in self.flies:
            cause = self.determine_fly_death(fly)
            if cause is None:
                survivors.append(fly)
            else:
                deaths[fly.id] = cause
        self.flies = survivors
        return deaths

    def determine_fly_death(self, fly: Fly) -> str | None:
        fly_on_a_threat = any(
            fly.position.x == t.position.x and fly.position.y == t.position.y
            for t in self.threats
        )
        if fly_on_a_threat:
            return "threat"
        if fly.hunger <= 0:
            return "starved"
        return None

    def sense_nearest(self, origin: Position, positions: list[Position], radius: float) -> SensorReading:
        nearest_dist, nearest_pos = None, None
        for pos in positions:
            dist = origin.distance_to(pos)
            if dist <= radius and (nearest_dist is None or dist < nearest_dist):
                nearest_dist, nearest_pos = dist, pos

        if nearest_pos is None:
            return SensorReading(signal=0.0, dx=0.0, dy=0.0)

        signal = 1.0 - nearest_dist / radius if radius > 0 else 1.0
        dx, dy = nearest_pos.x - origin.x, nearest_pos.y - origin.y
        norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        return SensorReading(signal=signal, dx=dx / norm, dy=dy / norm)

    def observe(self, fly: Fly) -> Observation:
        food = self.sense_nearest(fly.position, [f.position for f in self.food], self.food_radius)
        threat = self.sense_nearest(fly.position, [t.position for t in self.threats], self.threat_radius)
        return Observation(
            food_signal=food.signal, food_dx=food.dx, food_dy=food.dy,
            threat_signal=threat.signal, threat_dx=threat.dx, threat_dy=threat.dy,
            hunger=fly.hunger / self.max_hunger,
        )


def clamp_rate(rate: float) -> float:
    return min(max(rate, SPAWN_RATE_BOUNDS[0]), SPAWN_RATE_BOUNDS[1])
