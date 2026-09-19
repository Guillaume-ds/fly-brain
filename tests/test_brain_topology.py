"""Colony.brain_topology() / Colony.neuron_state() (wiki/decisions.md
#47) -- the real connectome subgraph each circuit runs on, plus live
per-neuron voltage/spike state for a watched fly, feeding the brain
inspector's Tier 3 neuron-graph view.
"""

from __future__ import annotations

import numpy as np
import pytest


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


# --- brain_topology() -------------------------------------------------------

def test_topology_has_both_circuits_with_the_expected_shape(templates):
    colony = make_colony(templates)
    topology = colony.brain_topology()

    assert set(topology) == {"escape", "plasticity"}
    for circuit in topology.values():
        assert set(circuit) == {"neuron_ids", "edges", "groups"}
        assert len(circuit["neuron_ids"]) > 0
        for edge in circuit["edges"]:
            assert set(edge) == {"pre", "post", "weight"}


def test_topology_edge_indices_are_valid_neuron_indices(templates):
    colony = make_colony(templates)
    topology = colony.brain_topology()

    for circuit in topology.values():
        n = len(circuit["neuron_ids"])
        for edge in circuit["edges"]:
            assert 0 <= edge["pre"] < n
            assert 0 <= edge["post"] < n


def test_topology_group_indices_are_valid_and_match_the_templates(templates):
    colony = make_colony(templates)
    topology = colony.brain_topology()
    escape_template, plasticity_template = templates

    escape = topology["escape"]
    n_escape = len(escape["neuron_ids"])
    assert set(escape["groups"]) == {"seed", "motor"}
    assert escape["groups"]["seed"] == escape_template.seed_idx
    assert escape["groups"]["motor"] == escape_template.motor_idx
    assert all(0 <= i < n_escape for i in escape["groups"]["seed"] + escape["groups"]["motor"])

    plasticity = topology["plasticity"]
    n_plasticity = len(plasticity["neuron_ids"])
    assert set(plasticity["groups"]) == {"kc", "mbon_approach", "mbon_avoid", "pam", "ppl"}
    assert plasticity["groups"]["kc"] == plasticity_template.kc_idx
    assert plasticity["groups"]["mbon_approach"] == plasticity_template.mbon_approach_idx
    assert plasticity["groups"]["ppl"] == plasticity_template.ppl_idx
    all_indices = [i for group in plasticity["groups"].values() for i in group]
    assert all(0 <= i < n_plasticity for i in all_indices)


def test_topology_is_the_same_regardless_of_which_fly_asks(templates):
    """Topology is colony-wide, not per-fly -- every fly of a given kind
    shares the same connectome subgraph, only synaptic gains differ.
    """
    colony = make_colony(templates, max_population=10, base_reproduction_rate=1.0)
    first = colony.brain_topology()
    second = colony.brain_topology()
    assert first == second


# --- neuron_state() ----------------------------------------------------------

def test_neuron_state_returns_none_for_a_fly_id_that_never_existed(templates):
    colony = make_colony(templates)
    assert colony.neuron_state(999_999) is None


def test_neuron_state_returns_none_after_death(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.env.flies[0].health = 0

    result = colony.step()

    assert fly_id in result.deaths
    assert colony.neuron_state(fly_id) is None


def test_neuron_state_arrays_are_index_aligned_with_topology(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.step()

    topology = colony.brain_topology()
    state = colony.neuron_state(fly_id)

    assert state["fly_id"] == fly_id
    assert len(state["escape"]["v"]) == len(topology["escape"]["neuron_ids"])
    assert len(state["escape"]["spikes"]) == len(topology["escape"]["neuron_ids"])
    assert len(state["plasticity"]["v"]) == len(topology["plasticity"]["neuron_ids"])
    assert len(state["plasticity"]["spikes"]) == len(topology["plasticity"]["neuron_ids"])
    assert all(isinstance(b, bool) for b in state["escape"]["spikes"])


def test_neuron_state_matches_the_real_circuit_state(templates):
    colony = make_colony(templates)
    fly_id = next(iter(colony.observations))
    colony.step()

    escape_circuit = colony.escape_agents[fly_id].circuit
    plasticity_circuit = colony.plasticity_agents[fly_id].circuit
    state = colony.neuron_state(fly_id)

    assert state["escape"]["v"] == pytest.approx(escape_circuit.v.tolist())
    assert state["escape"]["spikes"] == escape_circuit.spikes_prev.tolist()
    assert state["plasticity"]["v"] == pytest.approx(plasticity_circuit.v.tolist())
    assert state["plasticity"]["spikes"] == plasticity_circuit.spikes_prev.tolist()
