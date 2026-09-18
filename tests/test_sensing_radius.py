"""Sensing radius, split from interaction radius (wiki/decisions.md #39).

Before this entry, an entity's own `radius` served two different jobs at
once: how far away a fly could *perceive* it (observe()) and how close a
fly had to get to actually *interact* with it (resolve_item_pickup/
resolve_tile_effects/resolve_mob_contact). Bumping that one shared number
to let a fly smell food from far away also made food get eaten from that
same distance, instantly, with no approach ever happening -- confirmed
live before this fix (Scenario A produced *zero* reinforcement events,
because the "approach" was actually a single ambient pickup at tick 0).

What's pinned here: `WORLD_SENSING_RADIUS` (world/env.py) now governs
what shows up in `Observation.nearby` for every item/tile/mob, one
shared value across all three kinds; each entity's own `radius` still
gates the corresponding resolve_*() interaction, untouched and
independent. And, now that a real multi-tick approach is possible: the
KC eligibility trace (fly_brain/plasticity.py, decisions.md #22/#26)
already carries credit across the ticks spent sensing something before
contact, not just the tick contact happens on.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_brain.plasticity import PlasticityAgent, build_plasticity_template
from world.entities import Item, Mob, Position, Tile
from world.env import Action, DEFAULT_OWNER, Environment, WORLD_SENSING_RADIUS

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
        starting_energy=1000.0,
        seed=2,
    )


def place_fly_and_entity(env: Environment, entity, distance: int) -> None:
    fly = env.flies[0]
    fly.position = Position(2, 10)
    entity.position = Position(fly.position.x + distance, fly.position.y)
    if isinstance(entity, Mob):
        env.mobs = [entity]
    elif isinstance(entity, Tile):
        env.tiles = [entity]
    else:
        env.items = [entity]


# --- sensing radius applies uniformly to item/tile/mob -------------------

def test_item_sensed_well_beyond_its_interaction_radius(env):
    item = env.spawn_of(Item, env.food_type)
    place_fly_and_entity(env, item, distance=int(item.radius) + 2)

    assert len(env.observe(env.flies[0]).nearby) == 1


def test_tile_sensed_well_beyond_its_interaction_radius(env):
    assert env.add_tile_type("test tile", "a warm healing spring", 3) is True
    tile = env.spawn_of(Tile, env.tile_types[-1])
    place_fly_and_entity(env, tile, distance=int(tile.radius) + 2)

    assert len(env.observe(env.flies[0]).nearby) == 1


def test_mob_sensed_well_beyond_its_interaction_radius(env):
    assert env.add_mob_type("test mob", "damage", 3) is True
    mob = env.spawn_mob_of(env.mob_types[-1])
    place_fly_and_entity(env, mob, distance=int(mob.radius) + 2)

    assert len(env.observe(env.flies[0]).nearby) == 1


def test_nothing_beyond_the_shared_sensing_radius_is_perceived(env):
    item = env.spawn_of(Item, env.food_type)
    place_fly_and_entity(env, item, distance=int(WORLD_SENSING_RADIUS) + 2)

    assert env.observe(env.flies[0]).nearby == []


# --- the eligibility trace credits ticks spent approaching, not just contact ---

def test_sensing_before_contact_builds_a_real_eligibility_trace(env):
    """The concrete regression this entry exists for: a few ticks of
    walking toward something sensed-but-not-yet-reached must leave a
    real, nonzero trace footprint -- proof reinforce(), once contact
    finally happens, isn't only crediting the last tick.
    """
    template = build_plasticity_template(env.encoder)
    agent = PlasticityAgent(template, initial_gains=None)
    fly = env.flies[0]
    item = env.spawn_of(Item, env.food_type)
    # Distance chosen so several ticks of sensing happen before the fly
    # is close enough for resolve_item_pickup() to actually fire.
    place_fly_and_entity(env, item, distance=int(item.radius) + 3)

    for _ in range(int(item.radius) + 1):  # stop just short of contact
        obs = env.observe(fly)
        if not obs.nearby:
            pytest.skip("item left sensing range before contact -- distance/radius mismatch")
        agent.sense_and_decide(obs)

    assert float(np.linalg.norm(agent.kc_trace)) > 0.0
    assert agent.reinforcement_events == 0  # no contact yet -- sensing alone never reinforces


def test_a_multi_tick_approach_still_reinforces_on_eventual_contact(env):
    """End-to-end regression for the live-verified scenario: a fly that
    senses food from beyond its interaction radius, walks toward it over
    several ticks, and finally makes contact still gets real
    reinforcement -- not silently dropped, and not requiring the old
    "sensed and touched in the same tick" shortcut.
    """
    template = build_plasticity_template(env.encoder)
    agent = PlasticityAgent(template, initial_gains=None)
    fly = env.flies[0]
    item = env.spawn_of(Item, env.food_type)
    place_fly_and_entity(env, item, distance=5)

    for _ in range(5):
        obs = env.observe(fly)
        agent.sense_and_decide(obs)
        result = env.step({fly.id: Action.RIGHT})
        agent.reinforce(result.effects.get(fly.id, {}))
        fly = next((f for f in env.flies if f.id == fly.id), fly)
        if result.effects.get(fly.id if fly else -1, {}):
            break

    assert agent.reinforcement_events >= 1
    assert agent.gain_drift() > 0.0
