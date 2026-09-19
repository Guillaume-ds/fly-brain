"""Colony.brain_snapshot() / Colony.last_action_source (wiki/decisions.md
#46) -- a per-fly diagnostic bundle for a "brain inspector": which tier
(escape/wander/plasticity) produced a fly's last action, and the real
numbers behind that decision, assembled entirely from state the agents
already keep for auditing (decisions.md #26).
"""

from __future__ import annotations

import numpy as np
import pytest

from world.env import Action


@pytest.fixture(scope="module")
def templates():
    from fly_brain.agent import build_escape_template
    from fly_brain.plasticity import build_plasticity_template
    from world.env import Environment

    env = Environment(initial_population=1, seed=1)
    encoder = env.encoder
    return build_escape_template(encoder), build_plasticity_template(encoder)


def make_colony(templates, **env_kwargs):
    from game.colony import Colony
    from world.env import Environment

    escape_template, plasticity_template = templates
    env = Environment(
        initial_population=1,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_spawn_rate=0.0,
        tile_spawn_rate=0.0,
        starting_energy=1000.0,
        seed=3,
        **env_kwargs,
    )
    gains = np.ones(len(escape_template.blueprint.edges))
    return Colony(env, escape_template, plasticity_template, gains, seed=3)


# --- existence / lifecycle -------------------------------------------------

def test_returns_none_for_a_fly_id_that_never_existed(templates):
    colony = make_colony(templates)
    assert colony.brain_snapshot(999_999) is None


def test_returns_none_after_death(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.env.flies[0].health = 0

    result = colony.step()

    assert fly_id in result.deaths
    assert colony.brain_snapshot(fly_id) is None
    assert fly_id not in colony.last_action_source


# --- action_source precedence (mirrors test_wander.py's monkeypatch style) -

def test_action_source_is_escape_when_escape_fires(templates, monkeypatch):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    monkeypatch.setattr(colony.escape_agents[fly_id], "decide", lambda obs: Action.LEFT)

    colony.step()

    assert colony.last_action_source[fly_id] == "escape"
    assert colony.brain_snapshot(fly_id)["action_source"] == "escape"


def test_action_source_is_wander_when_escape_is_silent_and_wander_fires(templates, monkeypatch):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    monkeypatch.setattr(colony.escape_agents[fly_id], "decide", lambda obs: None)
    monkeypatch.setattr(colony.wander_agents[fly_id], "decide", lambda obs, rng: Action.RIGHT)

    colony.step()

    assert colony.last_action_source[fly_id] == "wander"
    assert colony.brain_snapshot(fly_id)["action_source"] == "wander"


def test_action_source_is_plasticity_when_escape_and_wander_are_both_silent(templates, monkeypatch):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    monkeypatch.setattr(colony.escape_agents[fly_id], "decide", lambda obs: None)
    monkeypatch.setattr(colony.wander_agents[fly_id], "decide", lambda obs, rng: None)

    colony.step()

    assert colony.last_action_source[fly_id] == "plasticity"
    assert colony.brain_snapshot(fly_id)["action_source"] == "plasticity"


# --- snapshot content matches the underlying agents' own state -------------

def test_snapshot_has_the_full_expected_shape(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.step()

    snapshot = colony.brain_snapshot(fly_id)

    assert snapshot["fly_id"] == fly_id
    assert snapshot["action_source"] in {"escape", "wander", "plasticity"}
    assert set(snapshot["escape"]) == {"danger_strength"}
    assert set(snapshot["plasticity"]) == {
        "valence", "gain_drift", "reinforcement_events", "cumulative_dopamine", "probe_food", "probe_danger",
    }
    assert set(snapshot["wander"]) == {"persistence", "current_direction"}


def test_escape_danger_strength_matches_the_escape_agent(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.step()

    escape_agent = colony.escape_agents[fly_id]
    snapshot = colony.brain_snapshot(fly_id)

    assert snapshot["escape"]["danger_strength"] == escape_agent.last_danger_strength


def test_plasticity_valence_matches_the_plasticity_agent(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.step()

    plasticity_agent = colony.plasticity_agents[fly_id]
    snapshot = colony.brain_snapshot(fly_id)

    assert snapshot["plasticity"]["valence"] == plasticity_agent.last_valence


def test_plasticity_audit_fields_match_the_plasticity_agent(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    for _ in range(5):
        colony.step()
        if fly_id not in colony.plasticity_agents:
            pytest.skip("fly died -- unrelated to brain_snapshot itself")

    plasticity_agent = colony.plasticity_agents[fly_id]
    snapshot = colony.brain_snapshot(fly_id)

    assert snapshot["plasticity"]["gain_drift"] == plasticity_agent.gain_drift()
    assert snapshot["plasticity"]["reinforcement_events"] == plasticity_agent.reinforcement_events
    assert snapshot["plasticity"]["cumulative_dopamine"] == plasticity_agent.cumulative_dopamine


def test_probe_fields_use_the_real_food_and_danger_reference_vectors(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.step()

    plasticity_agent = colony.plasticity_agents[fly_id]
    expected_food = plasticity_agent.probe(colony.env.food_type.attributes)
    expected_danger = plasticity_agent.probe(colony.escape_template.danger_vector)

    snapshot = colony.brain_snapshot(fly_id)

    assert snapshot["plasticity"]["probe_food"] == pytest.approx(expected_food)
    assert snapshot["plasticity"]["probe_danger"] == pytest.approx(expected_danger)


def test_wander_fields_match_the_wander_agent(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.wander_agents[fly_id].persistence = 4.0
    colony.step()

    wander_agent = colony.wander_agents[fly_id]
    snapshot = colony.brain_snapshot(fly_id)

    assert snapshot["wander"]["persistence"] == wander_agent.persistence
    expected_direction = (
        wander_agent.current_direction.name if wander_agent.current_direction is not None else None
    )
    assert snapshot["wander"]["current_direction"] == expected_direction
