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

A fly's observation is an anonymous list of Percepts (attribute vector +
relative position, no name/type/id) rather than named per-type fields --
see wiki/decisions.md #22. What an item actually does to a fly (heal,
hurt, immobilize) comes from the Result registry (world/results.py), a
similarity blend against the item's attribute vector, never a hardcoded
per-type constant.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from .entities import Fly, Food, Item, Position, Threat
from .items import HashingItemEncoder, ItemEncoder, ItemType, jitter
from .results import ResultConcept, build_default_results, blend_deltas

SPAWN_RATE_BOUNDS = (0.0, 0.2)  # min/max per-tick spawn probability
SPAWN_RATE_STEP = 0.02  # fixed nudge used by increase_*/decrease_*
MAX_ITEM_TYPES = 20  # cap on how many distinct player-created item types can exist, see decisions.md #27


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
class Percept:
    attributes: np.ndarray  # unit-norm item vector -- no name/type/id, see decisions.md #22 part 1
    dx: float  # unit direction to the percept; 0 if distance is ~0
    dy: float
    distance: float  # raw distance -- consumers decide their own proximity weighting


@dataclass
class Observation:
    nearby: list[Percept] = field(default_factory=list)
    hunger: float = 0.0  # current_hunger / max_hunger -- always known, not sensed


@dataclass
class ColonyStepResult:
    observations: dict[int, Observation]  # fly id -> observation, alive flies only
    deaths: dict[int, str]  # fly id -> cause ("threat" | "starved" | "damage")
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
        max_health: int = 100,
        max_stuck_ticks: int = 10,
        encoder: ItemEncoder | None = None,
        food_description: str = "a nourishing piece of food",
        threat_description: str = "a fast, venomous spider",
        item_jitter_sigma: float = 0.05,
        item_radius: float = 2.0,
        item_spawn_rate: float = 0.02,
        max_items: int = 15,
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
        self.max_health = max_health
        self.max_stuck_ticks = max_stuck_ticks
        self.encoder = encoder or HashingItemEncoder()
        self.item_jitter_sigma = item_jitter_sigma
        self.item_radius = item_radius
        self.item_spawn_rate = item_spawn_rate
        self.max_items = max_items
        self.results: list[ResultConcept] = build_default_results(
            self.encoder, max_hunger=self.max_hunger, max_health=self.max_health, max_stuck_ticks=self.max_stuck_ticks,
        )
        # prototypes encoded once -- spawning only jitters them, never
        # re-encodes the same fixed description from scratch every time
        self.food_type = ItemType(food_description, self.encoder.encode(food_description))
        self.threat_type = ItemType(threat_description, self.encoder.encode(threat_description))
        self.item_types: list[ItemType] = []  # player-created, see add_item_type()
        self.rng = np.random.default_rng(seed)

        self.flies: list[Fly]
        self.food: list[Food]
        self.threats: list[Threat]
        self.items: list[Item]
        self.tick: int
        self.next_fly_id: int
        self.reset()

    def add_item_type(self, description: str) -> None:
        """The player-facing item-creation entry point (decisions.md
        #27) -- director/'s create_item action wraps this directly. How
        flies perceive and react to the resulting item is determined
        entirely by `description`'s attribute vector, via the same
        Percept/Result-registry mechanism as food -- nothing else about
        it is configurable, on purpose (decisions.md #27: keep it
        simple). Silently capped at MAX_ITEM_TYPES.
        """
        if len(self.item_types) >= MAX_ITEM_TYPES:
            return
        self.item_types.append(ItemType(description, self.encoder.encode(description)))

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
        self.items = []
        self.flies = []
        for _ in range(self.initial_population):
            self.flies.append(self.spawn_fly(self.random_empty_cell(self.occupied_cells())))
        return {fly.id: self.observe(fly) for fly in self.flies}

    def spawn_fly(self, position: Position) -> Fly:
        fly = Fly(id=self.next_fly_id, position=position, hunger=self.max_hunger, health=self.max_health)
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
        self.resolve_item_pickup()
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
        cells.update((i.position.x, i.position.y) for i in self.items)
        return cells

    def move_flies(self, actions: dict[int, Action]) -> None:
        for fly in self.flies:
            if fly.vulnerable_ticks_left > 0:
                fly.vulnerable_ticks_left -= 1
                action = Action.STAY
            elif fly.stuck_ticks > 0:
                fly.stuck_ticks -= 1
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
            attributes = jitter(self.threat_type.attributes, self.item_jitter_sigma, self.rng)
            self.threats.append(Threat(self.random_empty_cell(self.occupied_cells()), self.threat_radius, attributes))
        if len(self.food) < self.max_food and self.rng.random() < self.food_spawn_rate:
            attributes = jitter(self.food_type.attributes, self.item_jitter_sigma, self.rng)
            self.food.append(Food(self.random_empty_cell(self.occupied_cells()), self.food_radius, attributes))
        if self.item_types and len(self.items) < self.max_items and self.rng.random() < self.item_spawn_rate:
            item_type = self.item_types[int(self.rng.integers(0, len(self.item_types)))]
            attributes = jitter(item_type.attributes, self.item_jitter_sigma, self.rng)
            self.items.append(Item(self.random_empty_cell(self.occupied_cells()), self.item_radius, attributes))

    def resolve_food_pickup(self) -> None:
        for fly in self.flies:
            for food in self.food:
                if fly.position.distance_to(food.position) <= food.pickup_radius:
                    self.apply_result(fly, food.attributes)
                    self.food.remove(food)
                    break

    def resolve_item_pickup(self) -> None:
        for fly in self.flies:
            for item in self.items:
                if fly.position.distance_to(item.position) <= item.interaction_radius:
                    self.apply_result(fly, item.attributes)
                    self.items.remove(item)
                    break

    def apply_result(self, fly: Fly, attributes: np.ndarray) -> None:
        """The real mechanical effect of an item on a fly -- a
        clipped-cosine-similarity blend across the Result registry, see
        world/results.py and decisions.md #22 part 5.
        """
        deltas = blend_deltas(attributes, self.results)
        fly.hunger = int(min(self.max_hunger, max(0, fly.hunger + deltas.get("hunger", 0.0))))
        fly.health = int(min(self.max_health, max(0, fly.health + deltas.get("health", 0.0))))
        fly.stuck_ticks = min(self.max_stuck_ticks, max(0, fly.stuck_ticks + round(deltas.get("stuck_ticks", 0.0))))

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
        if fly.health <= 0:
            return "damage"
        if fly.hunger <= 0:
            return "starved"
        return None

    def perceive(self, origin: Position, target: Position, attributes: np.ndarray, radius: float) -> Percept | None:
        """A percept is anonymous by construction: only the entity's
        attribute vector and relative position ever get returned here,
        nothing that reveals what world-internal type it came from (see
        decisions.md #22 part 1). `radius` decides visibility only --
        it's a world-internal cutoff, not part of what the fly receives.
        """
        distance = origin.distance_to(target)
        if distance > radius:
            return None
        dx, dy = target.x - origin.x, target.y - origin.y
        norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        return Percept(attributes=attributes, dx=dx / norm, dy=dy / norm, distance=distance)

    def observe(self, fly: Fly) -> Observation:
        nearby: list[Percept] = []
        for food in self.food:
            percept = self.perceive(fly.position, food.position, food.attributes, self.food_radius)
            if percept is not None:
                nearby.append(percept)
        for threat in self.threats:
            percept = self.perceive(fly.position, threat.position, threat.attributes, self.threat_radius)
            if percept is not None:
                nearby.append(percept)
        for item in self.items:
            percept = self.perceive(fly.position, item.position, item.attributes, self.item_radius)
            if percept is not None:
                nearby.append(percept)
        return Observation(nearby=nearby, hunger=fly.hunger / self.max_hunger)


def clamp_rate(rate: float) -> float:
    return min(max(rate, SPAWN_RATE_BOUNDS[0]), SPAWN_RATE_BOUNDS[1])
