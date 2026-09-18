"""Grid-world survival environment for a fly colony.

Items/tiles/mobs spawn continuously and mobs wander (see wiki/decisions.md
#14, #15 -- a static placement is permanently dodgeable). Multiple flies
share one world and can reproduce (see wiki/decisions.md #20). A
population of 1 makes reproduction structurally impossible
(mate_availability is exactly 0 at population 1) -- that's exactly the
single-fly curriculum-training case, so training/ uses this same class
rather than a separate one.

increase_/decrease_spider_rate and increase_/decrease_food_rate are
director/'s control surface, alongside create_item/create_tile/create_mob
(decisions.md #27, #30-#32).

There is exactly one item entity (decisions.md #28): food is a
registered ItemType like any player-created one, spawning through the
same path into the same `self.items` list. A `Tile` is exactly the same
shape, spawning through the same path into `self.tiles`, differing only
in never being consumed on contact (decisions.md #31). `Mob` is the
deliberate exception -- perceived like an item, but it moves and its
effect is authored (`effect`/`strength`), never derived from the Result
registry, and the built-in spider still kills unconditionally through
`determine_fly_death()`; see `entities.Mob` for why.

A fly's observation is an anonymous list of Percepts (attribute vector +
relative position, no name/type/id) rather than named per-type fields --
see wiki/decisions.md #22. What an item/tile actually does to a fly
(heal, hurt, immobilize) comes from the Result registry (world/results.py),
a similarity blend against its attribute vector scaled by `strength`
(decisions.md #32), never a hardcoded per-type constant.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .entities import Fly, Item, Mob, Position, Tile
from .items import HashingItemEncoder, ItemEncoder, ItemType, MobType, jitter
from .results import (
    MID_STRENGTH,
    TILE_EFFECT_FRACTION,
    Channel,
    Effect,
    ResultConcept,
    build_default_results,
    blend_deltas,
    clamp_strength,
    compose_mob_description,
    mob_effect_delta,
    strength_magnitude,
)

SPAWN_RATE_BOUNDS = (0.0, 0.2)  # min/max per-tick spawn probability
SPAWN_RATE_STEP = 0.02  # fixed nudge used by increase_*/decrease_*
MAX_ITEM_TYPES = 20  # cap on how many distinct player-created item types can exist, see decisions.md #27
MAX_TILE_TYPES = 20  # same reasoning as MAX_ITEM_TYPES -- an unbounded number of lingering area effects is a bigger exploit than an unbounded number of consumed items (decisions.md #31)
MAX_MOB_TYPES = 20
FOOD_TYPE_NAME = "food"  # the built-in ItemType director/ is allowed to tune
SPIDER_TYPE_NAME = "spider"  # the built-in MobType -- renamed from THREAT_TYPE_NAME (decisions.md #31): it's specifically the spider now that a Mob isn't always a threat


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
    nearby: list[Percept]
    hunger: float  # current_hunger / max_hunger -- always known, not sensed


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
        max_mobs: int = 10,
        mob_radius: float = 3.0,
        mob_move_probability: float = 0.3,
        max_population: int = 20,
        reproduction_hunger_threshold: float = 0.8,
        base_reproduction_rate: float = 0.05,
        max_mate_availability: float = 0.8,
        reproduction_hunger_cost_fraction: float = 0.4,
        reproduction_vulnerability_ticks: int = 3,
        offspring_spawn_radius: int = 2,
        food_enabled: bool = True,
        spider_enabled: bool = True,
        max_health: int = 100,
        max_stuck_ticks: int = 10,
        encoder: ItemEncoder | None = None,
        food_description: str = "a nourishing piece of food",
        spider_description: str = "a fast, venomous spider",
        item_jitter_sigma: float = 0.05,
        item_radius: float = 2.0,
        item_spawn_rate: float = 0.02,
        max_items: int = 15,
        mob_spawn_rate: float = 0.02,
        tile_radius: float = 2.5,
        tile_spawn_rate: float = 0.01,
        max_tiles: int = 8,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.max_hunger = max_hunger
        self.max_ticks = max_ticks
        self.initial_population = initial_population
        self.max_mobs = max_mobs
        self.mob_radius = mob_radius
        self.mob_move_probability = mob_move_probability
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
        self.mob_spawn_rate = mob_spawn_rate
        self.tile_radius = tile_radius
        self.tile_spawn_rate = tile_spawn_rate
        self.max_tiles = max_tiles
        self.results: list[ResultConcept] = build_default_results(
            self.encoder, max_hunger=self.max_hunger, max_health=self.max_health, max_stuck_ticks=self.max_stuck_ticks,
        )
        # Every Channel a Result can move needs a bound here -- apply_result()
        # reads this dict rather than branching per channel, so a Result on an
        # unregistered channel raises loudly instead of being silently dropped.
        self.channel_limits: dict[Channel, int] = {
            Channel.HUNGER: self.max_hunger,
            Channel.HEALTH: self.max_health,
            Channel.STUCK_TICKS: self.max_stuck_ticks,
        }

        # Prototypes are encoded once here -- spawning only jitters them,
        # never re-encodes the same description from scratch every tick.
        # Food is an ordinary ItemType; it just has a name so director/
        # can tune its rate (decisions.md #28). strength=MID_STRENGTH is
        # the unscaled baseline (decisions.md #32) -- food behaves exactly
        # as it always has.
        self.food_type = ItemType(
            name=FOOD_TYPE_NAME,
            description=food_description,
            attributes=self.encoder.encode(food_description),
            spawn_rate=clamp_rate(food_spawn_rate if food_enabled else 0.0),
            radius=item_radius,
            strength=MID_STRENGTH,
        )
        self.item_types: list[ItemType] = []  # player-created, see add_item_type()
        self.tile_types: list[ItemType] = []  # player-created, see add_tile_type()
        # The spider is registered the same way, so its rate/radius live in
        # one place like everything else -- but it spawns on its own path
        # and still kills unconditionally (effect=None, see entities.Mob).
        self.spider_type = MobType(
            name=SPIDER_TYPE_NAME,
            description=spider_description,
            attributes=self.encoder.encode(spider_description),
            spawn_rate=clamp_rate(spider_spawn_rate if spider_enabled else 0.0),
            radius=self.mob_radius,
        )
        self.mob_types: list[MobType] = []  # player-created, see add_mob_type()
        self.rng = np.random.default_rng(seed)

        self.flies: list[Fly]
        self.mobs: list[Mob]
        self.items: list[Item]
        self.tiles: list[Tile]
        self.tick: int
        self.next_fly_id: int
        self.reset()

    @property
    def spawnable_item_types(self) -> list[ItemType]:
        """Food and every player-created type, in one list -- they spawn
        through the same path and are the same kind of thing.
        """
        return [self.food_type, *self.item_types]

    @property
    def spawnable_tile_types(self) -> list[ItemType]:
        """No built-in tile exists today -- just every player-created
        one, kept as its own property for symmetry with the other two
        and so a future built-in tile has somewhere to slot in.
        """
        return list(self.tile_types)

    @property
    def spawnable_mob_types(self) -> list[MobType]:
        """The spider and every player-created type, in one list --
        mirrors spawnable_item_types."""
        return [self.spider_type, *self.mob_types]

    def add_item_type(self, name: str, description: str, strength: int) -> None:
        """The player-facing item-creation entry point (decisions.md
        #27, #30, #32) -- director/'s create_item action wraps this
        directly. How flies perceive and react to the resulting item is
        determined entirely by `name`+`description`'s attribute vector,
        via the same Percept/Result-registry mechanism as food;
        `strength` scales the resulting effect's overall magnitude only,
        never which channel moves or in which direction. Silently capped
        at MAX_ITEM_TYPES.
        """
        if len(self.item_types) >= MAX_ITEM_TYPES:
            return
        self.item_types.append(
            ItemType(
                name=name,
                description=description,
                attributes=self.encoder.encode(f"{name}, {description}"),
                spawn_rate=self.item_spawn_rate,
                radius=self.item_radius,
                strength=clamp_strength(strength),
            )
        )

    def add_tile_type(self, name: str, description: str, strength: int) -> None:
        """The player-facing tile-creation entry point (decisions.md
        #31, #32) -- director/'s create_tile action wraps this directly.
        Identical mechanism to add_item_type -- same encoder-driven
        attributes, same Result-registry effect, same `strength` scaling
        -- the only difference is downstream, in how a spawned Tile is
        resolved (never consumed, a fraction applied every tick, see
        resolve_tile_effects()). Silently capped at MAX_TILE_TYPES.
        """
        if len(self.tile_types) >= MAX_TILE_TYPES:
            return
        self.tile_types.append(
            ItemType(
                name=name,
                description=description,
                attributes=self.encoder.encode(f"{name}, {description}"),
                spawn_rate=self.tile_spawn_rate,
                radius=self.tile_radius,
                strength=clamp_strength(strength),
            )
        )

    def add_mob_type(self, name: str, effect: str, strength: int) -> None:
        """The player-facing mob-creation entry point (decisions.md #30-
        #32) -- director/'s create_mob action wraps this directly.
        Unlike add_item_type/add_tile_type, `effect` is authored and read
        directly -- never derived from the embedding, which drives
        perception only (decisions.md #30: the encoder can't be trusted
        with anything that could invalidate the frozen escape circuit's
        training). An `effect` outside the real vocabulary has nowhere
        to land and is silently dropped, same principle as everywhere
        else in this pipeline. Silently capped at MAX_MOB_TYPES.
        """
        if len(self.mob_types) >= MAX_MOB_TYPES:
            return
        try:
            effect_enum = Effect(effect)
        except ValueError:
            return
        strength = clamp_strength(strength)
        description = compose_mob_description(name, effect_enum, strength)
        self.mob_types.append(
            MobType(
                name=name,
                description=description,
                attributes=self.encoder.encode(description),
                spawn_rate=self.mob_spawn_rate,
                radius=self.mob_radius,
                effect=effect_enum,
                strength=strength,
            )
        )

    # director-facing controls -- see wiki/decisions.md #14 for why these
    # are fixed, bounded nudges rather than taking a magnitude argument.
    def increase_spider_rate(self) -> None:
        self.spider_type.spawn_rate = clamp_rate(self.spider_type.spawn_rate + SPAWN_RATE_STEP)

    def decrease_spider_rate(self) -> None:
        self.spider_type.spawn_rate = clamp_rate(self.spider_type.spawn_rate - SPAWN_RATE_STEP)

    def increase_food_rate(self) -> None:
        self.food_type.spawn_rate = clamp_rate(self.food_type.spawn_rate + SPAWN_RATE_STEP)

    def decrease_food_rate(self) -> None:
        self.food_type.spawn_rate = clamp_rate(self.food_type.spawn_rate - SPAWN_RATE_STEP)

    def reset(self) -> dict[int, Observation]:
        self.tick = 0
        self.next_fly_id = 0
        self.mobs = []
        self.items = []
        self.tiles = []
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
        self.move_mobs()
        self.spawn_entities()
        self.resolve_item_pickup()
        self.resolve_tile_effects()
        self.resolve_mob_contact()
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
        # Tiles are deliberately excluded -- a zone, not a thing you can't
        # share a cell with (decisions.md #31).
        cells = {(fly.position.x, fly.position.y) for fly in self.flies}
        cells.update((m.position.x, m.position.y) for m in self.mobs)
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

    def move_mobs(self) -> None:
        possible_steps = list(MOVES.values())
        for mob in self.mobs:
            if self.rng.random() < self.mob_move_probability:
                dx, dy = possible_steps[int(self.rng.integers(0, len(possible_steps)))]
                mob.position = self.clamp_to_grid(mob.position.x + dx, mob.position.y + dy)

    def spawn_entities(self) -> None:
        for mob_type in self.spawnable_mob_types:
            if len(self.mobs) >= self.max_mobs:
                break
            if self.rng.random() < mob_type.spawn_rate:
                self.mobs.append(self.spawn_mob_of(mob_type))
        for item_type in self.spawnable_item_types:
            if len(self.items) >= self.max_items:
                break
            if self.rng.random() < item_type.spawn_rate:
                self.items.append(self.spawn_of(Item, item_type))
        for tile_type in self.spawnable_tile_types:
            if len(self.tiles) >= self.max_tiles:
                break
            if self.rng.random() < tile_type.spawn_rate:
                self.tiles.append(self.spawn_of(Tile, tile_type))

    def spawn_of(self, entity_class: type, item_type: ItemType):
        """One spawn path for Item and Tile -- an empty cell, the type's
        radius, a per-instance jitter of its prototype vector, and the
        type's strength (decisions.md #32). They're built identically;
        what differs is only what the world does with them afterwards.
        """
        return entity_class(
            self.random_empty_cell(self.occupied_cells()),
            item_type.radius,
            jitter(item_type.attributes, self.item_jitter_sigma, self.rng),
            item_type.strength,
        )

    def spawn_mob_of(self, mob_type: MobType) -> Mob:
        """Mob's own spawn path -- same shape otherwise, but a Mob also
        carries `effect`/`strength` through from its MobType so contact
        resolution knows what to do without looking the type back up
        (decisions.md #31).
        """
        return Mob(
            self.random_empty_cell(self.occupied_cells()),
            mob_type.radius,
            jitter(mob_type.attributes, self.item_jitter_sigma, self.rng),
            mob_type.effect,
            mob_type.strength,
        )

    def resolve_item_pickup(self) -> None:
        for fly in self.flies:
            for item in self.items:
                if fly.position.distance_to(item.position) <= item.radius:
                    self.apply_result(fly, item.attributes, item.strength)
                    self.items.remove(item)
                    break

    def resolve_tile_effects(self) -> None:
        """Same mechanism as an item pickup, but never consumed, and only
        a fraction of it applied per tick -- a tile is a place a fly can
        stay in, not a thing it touches once (decisions.md #31, #32).
        """
        for fly in self.flies:
            for tile in self.tiles:
                if fly.position.distance_to(tile.position) <= tile.radius:
                    self.apply_result(fly, tile.attributes, tile.strength, tick_fraction=TILE_EFFECT_FRACTION)

    def resolve_mob_contact(self) -> None:
        """A created mob's effect: authored from `effect`/`strength`,
        graded per tick of contact, never blend_deltas() (decisions.md
        #30, #31). The built-in spider (effect=None) is untouched here --
        it still kills unconditionally through determine_fly_death().
        """
        for fly in self.flies:
            for mob in self.mobs:
                if mob.effect is None:
                    continue
                if fly.position.x == mob.position.x and fly.position.y == mob.position.y:
                    channel, delta = mob_effect_delta(mob.effect, mob.strength, self.channel_limits)
                    _write_channel_delta(fly, channel, delta, self.channel_limits[channel])

    def apply_result(self, fly: Fly, attributes: np.ndarray, strength: int, tick_fraction: float = 1.0) -> None:
        """The real mechanical effect of an item/tile on a fly -- a
        clipped-cosine-similarity blend across the Result registry, see
        world/results.py and decisions.md #22 part 5, scaled by
        `strength`'s overall magnitude (decisions.md #32) and by
        `tick_fraction` (a Tile applies a fraction of this per tick
        instead of once, decisions.md #31; an Item always passes the
        default 1.0).

        Generic over `Channel`: each channel's value names the `Fly`
        field it writes, and `channel_limits` bounds it. A Result on a
        channel with no registered limit raises here rather than being
        silently dropped (decisions.md #28).
        """
        magnitude = strength_magnitude(strength) * tick_fraction
        for channel, delta in blend_deltas(attributes, self.results).items():
            _write_channel_delta(fly, channel, delta * magnitude, self.channel_limits[channel])

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
        # Only the built-in spider (effect=None) still kills unconditionally
        # on contact -- a created mob's damage flows through resolve_mob_contact()
        # into `health`, so it's caught by the health<=0 check below instead
        # (decisions.md #31).
        fly_on_the_spider = any(
            fly.position.x == m.position.x and fly.position.y == m.position.y and m.effect is None
            for m in self.mobs
        )
        if fly_on_the_spider:
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
        """Items, tiles, and mobs are perceived identically -- each
        carries its own radius and attribute vector, and nothing that
        distinguishes them survives into the Percept (decisions.md #22
        part 1, #28, #31).
        """
        nearby = [
            percept
            for entity in (*self.items, *self.tiles, *self.mobs)
            if (percept := self.perceive(fly.position, entity.position, entity.attributes, entity.radius))
        ]
        return Observation(nearby=nearby, hunger=fly.hunger / self.max_hunger)


def _write_channel_delta(fly: Fly, channel: Channel, delta: float, limit: int) -> None:
    """The one place a Channel delta actually gets written to a Fly --
    shared by apply_result() (item/tile) and resolve_mob_contact() (mob)
    so the clamping logic exists exactly once.
    """
    current = getattr(fly, channel.value)
    setattr(fly, channel.value, int(min(limit, max(0, round(current + delta)))))


def clamp_rate(rate: float) -> float:
    return min(max(rate, SPAWN_RATE_BOUNDS[0]), SPAWN_RATE_BOUNDS[1])
