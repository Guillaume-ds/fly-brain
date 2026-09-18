"""Multi-colony ownership (wiki/decisions.md #34, #35).

What's pinned here, none of it covered by the item contract or the
creation-cost tests:

  - Fly.owner, inherited by offspring
  - per-owner extinction (ColonyStepResult.colony_extinct)
  - fly-vs-fly perception: every fly perceives every other fly through
    a fixed, exactly orthogonal per-owner vector -- and the Percept
    itself stays exactly attributes/dx/dy/distance, same as every other
    pathway (the item contract's own invariant, extended here)
  - combat: authored, symmetric HEALTH damage between different-owner
    flies on contact; same-owner contact does nothing
  - the kill-transfer reward: a dying fly's hunger moves to the rival(s)
    on its cell, same tick
  - target-based spawn placement for create_item/create_tile/create_mob
"""

from __future__ import annotations

import numpy as np
import pytest

from world.entities import Fly, Item, Position
from world.env import TARGET_SPAWN_RADIUS, Action, DEFAULT_OWNER, Environment

OWNER = DEFAULT_OWNER


@pytest.fixture
def env() -> Environment:
    """A quiet world: nothing spawns on its own, so each test controls
    exactly which flies exist and where.
    """
    return Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_move_probability=0.0,  # no created mobs here, but keep movement off for determinism generally
        seed=1,
    )


def place(env: Environment, owner: str, position: Position, **kwargs) -> Fly:
    """Adds one fly for `owner` at an exact position, bypassing spawn
    randomness entirely -- these tests are about ownership mechanics,
    not placement.
    """
    fly = env.spawn_fly(position, owner=owner)
    fly.__dict__.update(kwargs)
    env.flies.append(fly)
    return fly


# --- ownership itself ----------------------------------------------------

def test_every_fly_has_an_owner(env):
    assert env.flies[0].owner == OWNER


class _AlwaysReproduces:
    """A stand-in for np.random.Generator whose .random() always returns
    0.0, so resolve_reproduction()'s probability roll can never fail --
    deterministic without needing to hunt for a lucky seed. Everything
    else (random_nearby_empty_cell's placement) delegates to a real
    generator.
    """

    def __init__(self, real_rng: np.random.Generator) -> None:
        self._real = real_rng

    def random(self) -> float:
        return 0.0

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_offspring_inherits_the_parents_owner(env):
    env.spawn_colony("rival", 1)
    parent = next(f for f in env.flies if f.owner == "rival")
    parent.hunger = env.max_hunger  # clears reproduction_hunger_threshold
    env.flies[0].hunger = 0  # the default owner's own fly stays ineligible -- only "rival" may reproduce here
    env.rng = _AlwaysReproduces(np.random.default_rng(0))
    env.base_reproduction_rate = 1.0
    env.max_population = 50

    births = env.resolve_reproduction()

    assert births, "expected at least one birth to actually test inheritance"
    for new_id in births:
        offspring = next(f for f in env.flies if f.id == new_id)
        assert offspring.owner == "rival"


def test_register_owner_is_idempotent(env):
    env.register_owner("rival")
    vector_before, home_before, energy_before = (
        env.owner_vectors["rival"].copy(), env.owner_home["rival"], env.energy["rival"],
    )

    env.register_owner("rival")

    np.testing.assert_array_equal(env.owner_vectors["rival"], vector_before)
    assert env.owner_home["rival"] == home_before
    assert env.energy["rival"] == energy_before
    assert env.owners.count("rival") == 1


# --- per-owner extinction -------------------------------------------------

def test_colony_extinct_is_false_while_an_owner_has_flies(env):
    result = env.step({f.id: Action.STAY for f in env.flies})
    assert result.colony_extinct[OWNER] is False


def test_colony_extinct_is_true_only_for_the_owner_with_no_flies_left(env):
    env.spawn_colony("rival", 1)
    for fly in list(env.flies):
        if fly.owner == "rival":
            fly.health = 0  # dies this step via determine_fly_death's health<=0 check

    result = env.step({f.id: Action.STAY for f in env.flies})

    assert result.colony_extinct["rival"] is True
    assert result.colony_extinct[OWNER] is False


# --- fly-vs-fly perception -------------------------------------------------

def test_a_fly_perceives_a_nearby_fly_of_any_owner(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = Position(5, 5)
    rival.position = Position(5, 6)

    percepts = env.observe(mine).nearby

    assert len(percepts) == 1
    assert set(vars(percepts[0])) == {"attributes", "dx", "dy", "distance"}


def test_different_owners_perceive_as_different_vectors(env):
    """The whole point of decisions.md #35: a colony can tell rivals
    apart, unlike #34's original shared-vector draft.
    """
    env.register_owner("rival_a")
    env.register_owner("rival_b")

    assert not np.array_equal(env.owner_vectors["rival_a"], env.owner_vectors["rival_b"])
    assert np.dot(env.owner_vectors["rival_a"], env.owner_vectors["rival_b"]) == pytest.approx(0.0)


def test_a_colonys_own_flies_share_one_owner_vector(env):
    env.spawn_colony(OWNER, 1)  # a second fly for the same default owner
    same_owner_flies = [f for f in env.flies if f.owner == OWNER]
    assert len(same_owner_flies) == 2

    vectors = [env.owner_vectors[f.owner] for f in same_owner_flies]
    np.testing.assert_array_equal(vectors[0], vectors[1])


def test_fly_perception_respects_its_own_radius(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = Position(0, 0)
    rival.position = Position(19, 19)  # far outside FLY_PERCEPTION_RADIUS

    assert env.observe(mine).nearby == []


# --- combat -----------------------------------------------------------

def test_contact_between_different_owners_damages_both_symmetrically(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = rival.position = Position(5, 5)
    mine.health = rival.health = 60

    env.resolve_fly_combat()

    assert mine.health < 60
    assert rival.health < 60
    assert mine.health == rival.health  # symmetric, same fixed damage to each


def test_contact_between_the_same_owner_does_nothing(env):
    second = place(env, OWNER, env.flies[0].position)
    env.flies[0].health = second.health = 60

    env.resolve_fly_combat()

    assert env.flies[0].health == 60
    assert second.health == 60


def test_flies_far_apart_do_not_fight(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position, rival.position = Position(0, 0), Position(19, 19)
    mine.health = rival.health = 60

    env.resolve_fly_combat()

    assert mine.health == 60
    assert rival.health == 60


# --- kill-transfer reward -------------------------------------------------

def test_a_dying_fly_transfers_hunger_to_the_rival_on_its_cell(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = rival.position = Position(5, 5)
    mine.health = 0  # already dying this tick
    mine.hunger = 80
    rival.hunger = 20

    env.resolve_kill_transfers()

    assert rival.hunger > 20
    assert mine.hunger == 80  # the dying fly's own hunger is read, not spent from


def test_kill_transfer_splits_evenly_across_multiple_rivals(env):
    env.spawn_colony("rival_a", 1)
    env.spawn_colony("rival_b", 1)
    mine = env.flies[0]
    rival_a = next(f for f in env.flies if f.owner == "rival_a")
    rival_b = next(f for f in env.flies if f.owner == "rival_b")
    mine.position = rival_a.position = rival_b.position = Position(5, 5)
    mine.health = 0
    mine.hunger = 80
    rival_a.hunger = rival_b.hunger = 10

    env.resolve_kill_transfers()

    assert rival_a.hunger == rival_b.hunger
    assert rival_a.hunger > 10


def test_a_fly_that_isnt_dying_transfers_nothing(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    mine.position = rival.position = Position(5, 5)
    mine.health = 50  # not dying
    rival.hunger = 20

    env.resolve_kill_transfers()

    assert rival.hunger == 20


def test_no_transfer_without_a_rival_on_the_same_cell(env):
    mine = env.flies[0]
    mine.health = 0
    mine.hunger = 80

    env.resolve_kill_transfers()  # should not raise, nothing to do

    assert mine.hunger == 80


# --- target-based spawn placement ------------------------------------

def test_no_target_spawns_anywhere_unchanged_from_before(env):
    """The default -- omitting `target` entirely -- must behave exactly
    like it always has: no bias toward any owner's home region.
    """
    assert env.add_item_type("berry", "a sweet ripe berry", 3) is True
    assert env.item_types[0].spawn_near is None


def test_target_own_biases_toward_the_issuing_owners_home(env):
    assert env.add_item_type("berry", "a sweet ripe berry", 3, owner=OWNER, target="own") is True

    assert env.item_types[0].spawn_near == env.owner_home[OWNER]


def test_target_a_named_rival_biases_toward_their_home(env):
    assert env.add_item_type("trap", "a sweet ripe berry", 3, owner=OWNER, target="rival") is True

    assert env.item_types[0].spawn_near == env.owner_home["rival"]
    assert "rival" in env.owners  # registered even though they own no flies yet


def test_spawned_instance_lands_near_its_types_target(env):
    assert env.add_item_type("berry", "a sweet ripe berry", 3, owner=OWNER, target="own") is True
    home = env.owner_home[OWNER]

    item = env.spawn_of(Item, env.item_types[0])

    # random_nearby_empty_cell samples x/y independently within radius --
    # a square neighborhood, not a circular one (Euclidean distance can
    # reach radius * sqrt(2)) -- so the bound to check is per-axis.
    assert abs(item.position.x - home.x) <= TARGET_SPAWN_RADIUS
    assert abs(item.position.y - home.y) <= TARGET_SPAWN_RADIUS
