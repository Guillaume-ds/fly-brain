"""The resource-cost system (wiki/decisions.md #33).

Not part of the item contract (tests/test_item_contract.py) -- this is
economy bookkeeping, not "what is an item." What's pinned here:

  - cost is a function of (kind, strength) only, per creation_cost()
  - a request that can't be afforded is rejected -- no type registered,
    no energy spent -- and the caller can tell that apart from success
  - a request that clears energy but fails some other guard (the type
    cap, an invalid mob effect) never spends energy either
  - energy regenerates on a fixed per-tick schedule, capped at max_energy
"""

from __future__ import annotations

import pytest

from world.env import (
    KIND_BASE_COST,
    MAX_ITEM_TYPES,
    Action,
    Environment,
    creation_cost,
)


@pytest.fixture
def env() -> Environment:
    return Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        starting_energy=20.0,
        max_energy=50.0,
        energy_regen_per_tick=0.1,
        seed=1,
    )


# --- the cost formula itself --------------------------------------------

@pytest.mark.parametrize("kind", ["item", "tile", "mob"])
@pytest.mark.parametrize("strength", [1, 2, 3, 4, 5])
def test_creation_cost_is_kind_base_times_strength(kind, strength):
    assert creation_cost(kind, strength) == KIND_BASE_COST[kind] * strength


def test_tile_costs_more_than_item_or_mob_at_equal_strength():
    """Tiles keep paying out every tick they're stood in -- that's the
    whole reason they're priced higher (decisions.md #33), pinned here
    so a careless constant edit can't quietly erase it.
    """
    assert creation_cost("tile", 3) > creation_cost("item", 3)
    assert creation_cost("tile", 3) > creation_cost("mob", 3)


def test_creation_cost_clamps_out_of_range_strength():
    """The [1, 5] bound is advisory in the director schema, not
    enforced by anything upstream -- creation_cost() has to be the
    real enforcement, same principle as clamp_strength() elsewhere.
    """
    assert creation_cost("item", 999) == creation_cost("item", 5)
    assert creation_cost("item", -3) == creation_cost("item", 1)


# --- spending -------------------------------------------------------------

def test_affordable_creation_succeeds_and_deducts_exactly_its_cost(env):
    before = env.energy
    cost = creation_cost("item", 3)

    created = env.add_item_type("berry", "a sweet ripe berry", 3)

    assert created is True
    assert env.energy == pytest.approx(before - cost)
    assert len(env.item_types) == 1


def test_unaffordable_creation_is_rejected_and_spends_nothing(env):
    env.energy = 1.0  # below even the cheapest item at strength 1
    before = env.energy

    created = env.add_item_type("berry", "a sweet ripe berry", 1)

    assert created is False
    assert env.energy == before
    assert env.item_types == []


@pytest.mark.parametrize(
    "kind,create",
    [
        ("item", lambda env: env.add_item_type("berry", "a sweet ripe berry", 5)),
        ("tile", lambda env: env.add_tile_type("spring", "a warm healing spring", 5)),
        ("mob", lambda env: env.add_mob_type("spider", "damage", 5)),
    ],
)
def test_each_kind_is_gated_by_its_own_cost(env, kind, create):
    """Not just create_item -- create_tile and create_mob are gated the
    same way, at their own (higher) cost.
    """
    env.energy = creation_cost(kind, 5) - 0.01  # just short of affordable

    created = create(env)

    assert created is False
    assert env.energy == pytest.approx(creation_cost(kind, 5) - 0.01)


def test_type_cap_rejection_spends_no_energy(env):
    """Cap check happens before affordability -- a request that would
    be rejected anyway must never cost anything (decisions.md #33's
    guard order). Energy set high so only the cap, not affordability,
    is what's under test here.
    """
    env.energy = 1000.0
    for i in range(MAX_ITEM_TYPES):
        assert env.add_item_type(f"item {i}", "a plain object", 1) is True
    before = env.energy

    created = env.add_item_type("one too many", "a plain object", 1)

    assert created is False
    assert env.energy == before
    assert len(env.item_types) == MAX_ITEM_TYPES


def test_invalid_mob_effect_spends_no_energy(env):
    """Same principle as the cap: a request that was never going to
    succeed (decisions.md #30's "nowhere to land") must never be
    charged for, regardless of whether energy was available.
    """
    before = env.energy

    created = env.add_mob_type("dragon", "breathe_fire", 5)

    assert created is False
    assert env.energy == before
    assert env.mob_types == []


# --- regeneration -----------------------------------------------------

def test_energy_regenerates_each_tick(env):
    env.energy = 10.0
    actions = {fly.id: Action.STAY for fly in env.flies}

    env.step(actions)

    assert env.energy == pytest.approx(10.0 + env.energy_regen_per_tick)


def test_energy_regen_is_capped_at_max_energy(env):
    env.energy = env.max_energy
    actions = {fly.id: Action.STAY for fly in env.flies}

    env.step(actions)

    assert env.energy == env.max_energy


def test_energy_resets_to_starting_energy_on_reset(env):
    env.energy = 2.0

    env.reset()

    assert env.energy == env.starting_energy
