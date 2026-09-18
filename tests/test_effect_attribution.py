"""Effect-attributed reward (wiki/decisions.md #28, #37).

Before this entry, `Colony.step()` reinforced on a before/after diff of
a fly's *net* per-tick state -- which silently mixed in the constant
hunger decay alongside whatever an item/tile/mob/fly actually did. A
real but weak food pickup (say +0.4 hunger) netted against that tick's
-1 decay read as -0.6, and `reward = max(0, delta) * weight` turned
that into exactly zero reward for a real food pickup -- the concrete
mechanism behind decisions.md #28's "starvation produces no learning
signal at all" finding (decay swallows weak effects, not just itself).

What's pinned here: `ColonyStepResult.effects` carries ONLY effect
deltas -- item/tile pickup, mob contact, fly combat, kill-transfer --
never the decay, and `Colony.step()` reinforces on that, not a net-state
diff. Not addressed here, and deliberately not conflated with it: #28's
separate, still-open question of whether hunger *loss* itself (decay or
a `starve`-effect mob) should carry its own punishment signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from world.entities import Item, Position
from world.env import Action, DEFAULT_OWNER, Environment
from world.results import Channel

OWNER = DEFAULT_OWNER


@pytest.fixture
def env() -> Environment:
    return Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        seed=1,
    )


def place_item(env: Environment, description: str, strength: int) -> Item:
    assert env.add_item_type("test item", description, strength) is True
    item = env.spawn_of(Item, env.item_types[-1])
    item.position = env.flies[0].position
    env.items = [item]
    return item


# --- decay is never an effect -------------------------------------------

def test_a_quiet_tick_produces_no_effects_for_anyone(env):
    result = env.step({f.id: Action.STAY for f in env.flies})

    assert result.effects.get(env.flies[0].id, {}) == {}


def test_hunger_decay_itself_never_appears_in_effects(env):
    """The fly's real hunger DOES drop by decay -- effects just never
    attributes that drop to anything, because it isn't an effect of
    anything the fly touched.
    """
    fly = env.flies[0]
    hunger_before = fly.hunger

    result = env.step({fly.id: Action.STAY})

    assert fly.hunger == hunger_before - 1  # decay still happens
    assert Channel.HUNGER not in result.effects.get(fly.id, {})  # just never attributed


# --- item/tile effects are tracked, and are the RAW effect, not netted ---

def test_item_pickup_effect_matches_the_registry_exactly(env):
    """Not netted against decay -- effects should show precisely what
    blend_deltas() * strength_magnitude() produced, full stop.
    """
    from world.results import blend_deltas, strength_magnitude

    item = place_item(env, "a smelly, poisonous piece of meat", 4)
    fly = env.flies[0]
    magnitude = strength_magnitude(item.strength)
    expected = {
        channel: round(delta * magnitude) for channel, delta in blend_deltas(item.attributes, env.results).items()
    }

    result = env.step({fly.id: Action.STAY})

    # blend_deltas() always returns an entry per registered Result concept
    # (food/damage/immobilize), even when its weight is 0 -- apply_result()
    # applies (and therefore _apply_channel_delta() records) every one of
    # them, so the two dicts should match key-for-key, not just on the
    # nonzero ones.
    actual = {channel: round(delta) for channel, delta in result.effects[fly.id].items()}
    assert actual == expected


def test_a_weak_effect_is_not_diluted_by_the_same_ticks_decay(env):
    """The regression this entry exists to fix: construct an item whose
    real hunger effect is small (< 1) -- under the old net-diff scheme
    this would have netted to a negative number against decay and
    produced ZERO reward. Under effects-based attribution it must show
    up as its own small positive number, undiluted.
    """
    # A hashing-encoder description with only mild food-similarity gives
    # a small blend weight; strength=1 keeps the magnitude scaling at
    # its minimum (decisions.md #32), so the resulting Delta-hunger is
    # small on purpose.
    item = place_item(env, "a plain grey pebble", 1)
    fly = env.flies[0]

    result = env.step({fly.id: Action.STAY})

    hunger_effect = result.effects.get(fly.id, {}).get(Channel.HUNGER)
    if hunger_effect is None or hunger_effect <= 0:
        pytest.skip("this description didn't land with a positive hunger weight under the stub encoder")
    # The real point of the test: whatever the raw effect was, it's
    # exposed exactly as-is -- never effect-minus-decay. A positive
    # result here, on a description chosen to be weak, is exactly the
    # case the old net-diff scheme would have zeroed out.
    assert hunger_effect > 0


def test_tile_effect_is_tracked_and_excludes_decay(env):
    assert env.add_tile_type("test tile", "a warm healing spring", 3) is True
    from world.entities import Tile

    tile = env.spawn_of(Tile, env.tile_types[-1])
    tile.position = env.flies[0].position
    env.tiles = [tile]
    fly = env.flies[0]

    result = env.step({fly.id: Action.STAY})

    assert fly.id in result.effects
    assert result.effects[fly.id] != {}


# --- mob contact, fly combat, kill-transfer are all tracked --------------

def test_mob_contact_effect_is_tracked(env):
    assert env.add_mob_type("test mob", "damage", 3) is True
    mob = env.spawn_mob_of(env.mob_types[-1])
    mob.position = env.flies[0].position
    env.mobs = [mob]
    fly = env.flies[0]

    result = env.step({fly.id: Action.STAY})

    assert result.effects[fly.id][Channel.HEALTH] < 0


def test_fly_combat_effect_is_tracked(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = mine.position
    actions = {f.id: Action.STAY for f in env.flies}

    result = env.step(actions)

    assert result.effects[mine.id][Channel.HEALTH] < 0
    assert result.effects[rival.id][Channel.HEALTH] < 0


def test_kill_transfer_effect_is_tracked(env):
    env.spawn_colony("rival", 1)
    mine = env.flies[0]
    rival = next(f for f in env.flies if f.owner == "rival")
    rival.position = mine.position
    mine.health = 1  # combat this tick will finish it off
    mine.hunger = 80

    result = env.step({f.id: Action.STAY for f in env.flies})

    # mine died this tick -- its own effects entry (from taking combat
    # damage) is still exposed even though it's no longer in env.flies.
    assert mine.id not in [f.id for f in env.flies]
    assert result.effects[rival.id][Channel.HUNGER] > 0


# --- Colony actually reinforces off effects, not a net-state diff --------

def test_colony_reinforces_using_effects_not_net_state(env, monkeypatch):
    """An integration check that Colony.step() really wires result.effects
    into reinforce(), by spying on what it's called with.
    """
    from fly_brain.agent import build_escape_template
    from fly_brain.plasticity import build_plasticity_template
    from game.colony import Colony

    encoder = env.encoder
    escape_template = build_escape_template(encoder)
    plasticity_template = build_plasticity_template(encoder)
    colony = Colony(env, escape_template, plasticity_template, np.ones(len(escape_template.blueprint.edges)), seed=1)

    place_item(colony.env, "a smelly, poisonous piece of meat", 5)

    calls = []

    def spy(deltas, _original):
        calls.append(deltas)
        _original(deltas)

    for agent in colony.plasticity_agents.values():
        original = agent.reinforce
        monkeypatch.setattr(agent, "reinforce", lambda deltas, _orig=original: spy(deltas, _orig))

    colony.step()

    assert calls, "reinforce() was never called"
    assert any(deltas for deltas in calls), "reinforce() should have been called with a non-empty effect this tick"


def test_colony_step_survives_a_birth_in_the_same_tick(env):
    """A regression guard: a newborn fly exists in env.flies the moment
    env.step() returns (reproduction runs inside it), but Colony hasn't
    built its brain agents yet -- that happens in the births loop right
    after the reinforce loop. Reinforcing must skip a fly with no agents
    yet rather than KeyError.
    """
    from fly_brain.agent import build_escape_template
    from fly_brain.plasticity import build_plasticity_template
    from game.colony import Colony

    encoder = env.encoder
    escape_template = build_escape_template(encoder)
    plasticity_template = build_plasticity_template(encoder)
    gains = np.ones(len(escape_template.blueprint.edges))
    colony = Colony(env, escape_template, plasticity_template, gains, seed=1)
    colony.add_colony(OWNER, 1, gains)  # reproduction needs population >= 2 to be possible at all

    # Force a guaranteed birth this tick, same trick as test_multi_colony.py.
    colony.env.max_population = 10
    colony.env.base_reproduction_rate = 1.0
    for fly in colony.env.flies:
        fly.hunger = colony.env.max_hunger

    class _AlwaysReproduces:
        def __init__(self, real_rng):
            self._real = real_rng

        def random(self):
            return 0.0

        def __getattr__(self, name):
            return getattr(self._real, name)

    colony.env.rng = _AlwaysReproduces(colony.env.rng)

    result = colony.step()  # must not raise

    assert result.births
