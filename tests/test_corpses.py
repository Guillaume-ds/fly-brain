"""Corpses (wiki/decisions.md #41).

Replaces the old fly-to-fly kill-transfer (decisions.md #35), which had
a real order-dependence bug: in a mutual kill (both flies reach 0 health
the same tick -- the normal outcome of symmetric 1v1 combat between
equal-health flies), `resolve_kill_transfers()` iterated `self.flies`
and mutated a dying fly's hunger mid-loop, so whichever fly was
processed second received a transfer computed from an already-inflated
hunger value. Verified live before this entry: 60/60 pre-death hunger
produced a 45/30 split, not the expected symmetric 30/30 -- 75 total
transferred, more than either fly's actual hunger.

What's pinned here: every fly death (combat, starvation, or a mob --
not just a kill) leaves a `Corpse` at its death position, worth a
fraction of what that fly had left. Any fly, any owner, can eat it --
same mechanism as an item pickup (sense via WORLD_SENSING_RADIUS,
consumed within a small interaction radius, removed after). No more
fly-to-fly mutation, so no more order dependence.
"""

from __future__ import annotations

import pytest

from world.entities import Position
from world.env import CORPSE_HUNGER_FRACTION, Action, DEFAULT_OWNER, Environment
from world.results import Channel

OWNER = DEFAULT_OWNER


@pytest.fixture
def env() -> Environment:
    return Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_spawn_rate=0.0,
        tile_spawn_rate=0.0,
        mob_move_probability=0.0,
        starting_energy=1000.0,
        seed=1,
    )


# --- a corpse spawns on death, worth a fraction of the fly's own hunger ---

def test_death_leaves_a_corpse_worth_a_fraction_of_its_own_hunger(env):
    fly = env.flies[0]
    fly.health = 0
    fly.hunger = 80

    deaths = env.resolve_deaths()

    assert deaths
    assert len(env.corpses) == 1
    assert env.corpses[0].position == fly.position
    assert env.corpses[0].hunger_value == pytest.approx(80 * CORPSE_HUNGER_FRACTION)


def test_a_starved_fly_leaves_a_worthless_corpse(env):
    fly = env.flies[0]
    fly.hunger = 0  # starvation trigger

    env.resolve_deaths()

    assert env.corpses[0].hunger_value == pytest.approx(0.0)


def test_every_death_cause_spawns_a_corpse_not_just_combat(env):
    """Decision: uniform across combat, starvation, and mobs -- not
    scoped to kills the way the old kill-transfer was.
    """
    fly = env.flies[0]
    fly.hunger = 0  # starves, not killed by anything

    env.resolve_deaths()

    assert len(env.corpses) == 1


# --- eating a corpse: any owner, single-use, distance-gated ---------------

def test_any_fly_any_owner_can_eat_a_corpse(env):
    env.spawn_colony("rival", 1)
    dying = env.flies[0]
    dying.health = 0
    dying.hunger = 80
    env.resolve_deaths()
    corpse_position = env.corpses[0].position

    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = corpse_position
    rival.hunger = 50  # room to grow -- default spawn hunger is already at the max
    hunger_before = rival.hunger

    env.resolve_corpse_pickup()

    assert rival.hunger > hunger_before


def test_a_flys_own_colony_can_eat_its_corpse_too(env):
    """Deliberately not rival-restricted: a starved colony-mate's corpse
    can feed its own colony.
    """
    env.spawn_colony(OWNER, 1)  # a second same-owner fly
    dying, survivor = env.flies[0], env.flies[1]
    dying.health = 0
    dying.hunger = 80
    survivor.hunger = 50  # room to grow -- default spawn hunger is already at the max
    env.resolve_deaths()
    corpse_position = env.corpses[0].position

    survivor.position = corpse_position
    hunger_before = survivor.hunger

    env.resolve_corpse_pickup()

    assert survivor.hunger > hunger_before


def test_corpse_removed_after_being_eaten(env):
    fly = env.flies[0]
    fly.health = 0
    fly.hunger = 80
    env.resolve_deaths()
    survivor_position = env.corpses[0].position
    env.spawn_colony("rival", 1)
    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = survivor_position

    env.resolve_corpse_pickup()

    assert env.corpses == []


def test_a_fly_far_away_does_not_eat_it(env):
    env.spawn_colony("rival", 1)
    dying = env.flies[0]
    dying.position = Position(0, 0)
    dying.health = 0
    dying.hunger = 80
    env.resolve_deaths()

    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = Position(19, 19)  # far outside corpse_radius

    env.resolve_corpse_pickup()

    assert len(env.corpses) == 1  # still there, uneaten


def test_a_fly_only_eats_one_corpse_per_tick(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    mine.position = Position(5, 5)
    mine.health = 0
    mine.hunger = 80
    env.resolve_deaths()
    # a second corpse, same spot, same tick
    env.spawn_colony("rival_b", 1)
    second = env.flies[-1]
    second.position = Position(5, 5)
    second.health = 0
    second.hunger = 80
    env.resolve_deaths()
    assert len(env.corpses) == 2

    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = Position(5, 5)

    env.resolve_corpse_pickup()

    assert len(env.corpses) == 1  # only one eaten this tick


# --- the max_corpses cap -----------------------------------------------

def test_corpses_are_capped(env):
    env.max_corpses = 2
    for i in range(4):
        env.spawn_colony(f"owner_{i}", 1)
        dying = env.flies[-1]
        dying.health = 0
        dying.hunger = 80
        env.resolve_deaths()

    assert len(env.corpses) == 2


# --- perceived like anything else, via the shared sensing radius ---------

def test_a_corpse_is_perceived_like_any_other_entity(env):
    from world.env import WORLD_SENSING_RADIUS

    fly = env.flies[0]
    fly.health = 0
    fly.hunger = 80
    env.resolve_deaths()
    env.spawn_colony("rival", 1)
    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = Position(env.corpses[0].position.x, env.corpses[0].position.y)

    percepts = env.observe(rival).nearby

    assert len(percepts) >= 1
    assert set(vars(percepts[0])) == {"attributes", "dx", "dy", "distance"}


# --- the order-dependence bug this entry exists to fix --------------------

def test_a_mutual_kill_no_longer_inflates_or_skews_the_reward(env):
    """The regression this entry exists to fix: two equal-hunger flies
    dying the same tick must each produce a corpse worth exactly their
    OWN pre-death hunger fraction -- no fly-to-fly mutation, so no
    order-dependent inflation (the old bug: 60/60 in, 45/30 out).
    """
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = rival.position = Position(5, 5)
    mine.health = 0
    rival.health = 0
    mine.hunger = 60
    rival.hunger = 60

    env.resolve_deaths()

    assert len(env.corpses) == 2
    values = sorted(c.hunger_value for c in env.corpses)
    expected = 60 * CORPSE_HUNGER_FRACTION
    assert values == pytest.approx([expected, expected])  # symmetric, not 45/30
    assert sum(values) == pytest.approx(2 * expected)  # no value created from nothing


def test_a_mutual_1v1_kill_really_happens_live(env):
    """Confirms the mechanical claim in this file's module docstring: two
    full-health flies fighting to a standstill both die the same tick,
    under today's symmetric FLY_COMBAT_DAMAGE.
    """
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = rival.position = Position(5, 5)
    mine.hunger = rival.hunger = 80

    for _ in range(25):
        result = env.step({f.id: Action.STAY for f in env.flies})
        if result.deaths:
            break

    assert mine.id in result.deaths
    assert rival.id in result.deaths
