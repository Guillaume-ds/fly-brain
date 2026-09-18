"""Hunger-loss punishment (wiki/decisions.md #38).

Before this entry, `reinforce()`'s punishment term only ever looked at
Δhealth and Δstuck_ticks -- a fly losing hunger (the `starve`-effect mob,
decisions.md #30) produced neither reward nor punishment, learning
nothing from an encounter that killed it. This extends the punishment
formula symmetrically with the existing health-loss term:
`max(0, -Δhunger) * CHANNEL_WEIGHTS[HUNGER]`.

Deliberately NOT covered here, because it's a different question already
settled by #37: whether plain per-tick hunger *decay* should itself be
punished. It never reaches this code at all -- `ColonyStepResult.effects`
excludes decay by construction, so `reinforce()` never sees it. This file
only pins the new term's own behavior, given some deltas dict.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_brain.plasticity import PlasticityAgent, build_plasticity_template
from world.items import HashingItemEncoder
from world.results import Channel


@pytest.fixture(scope="module")
def template():
    encoder = HashingItemEncoder(output_dim=16)
    return build_plasticity_template(encoder, seed=0)


def fresh_agent(template) -> PlasticityAgent:
    return PlasticityAgent(template, initial_gains=None)


# --- the new term fires, and moves gains in the punishing direction ------

def test_hunger_loss_alone_triggers_reinforcement(template):
    agent = fresh_agent(template)
    # Drive the eligibility trace first, same as a real tick would via
    # sense_and_decide(), so there's something for the punishment signal
    # to actually act on.
    attrs = np.ones(16)
    _sense(agent, attrs)

    before_events = agent.reinforcement_events
    agent.reinforce({Channel.HUNGER: -10.0})

    assert agent.reinforcement_events == before_events + 1
    assert agent.gain_drift() > 0.0


def test_hunger_loss_depresses_approach_and_potentiates_avoid(template):
    """Mirrors the existing health-loss punishment direction (decisions.md
    #26): for the KCs made eligible this tick, punishment should weaken
    the approach pathway and strengthen the avoid pathway -- the same
    "move away from this" shift a `damage` mob's health loss already
    produces, now also produced by a `starve` mob's hunger loss.
    """
    agent = fresh_agent(template)
    attrs = np.ones(16)
    _sense(agent, attrs)

    approach_idx = template.kc_mbon_approach_edge_idx
    avoid_idx = template.kc_mbon_avoid_edge_idx
    approach_before = agent.circuit.get_params()[approach_idx].copy()
    avoid_before = agent.circuit.get_params()[avoid_idx].copy()

    agent.reinforce({Channel.HUNGER: -10.0})

    approach_after = agent.circuit.get_params()[approach_idx]
    avoid_after = agent.circuit.get_params()[avoid_idx]

    assert approach_after.sum() <= approach_before.sum()
    assert avoid_after.sum() >= avoid_before.sum()
    assert not np.array_equal(approach_before, approach_after) or not np.array_equal(avoid_before, avoid_after)


def test_repeated_hunger_loss_moves_probe_valence_negative(template):
    """The colony.md testing contract: repeatedly reinforcing the same
    percept with a consistent-sign outcome must move probe()'s valence
    for that percept in the matching direction, and leave gain_drift() > 0.
    """
    agent = fresh_agent(template)
    attrs = np.ones(16)

    valence_before = agent.probe(attrs)
    for _ in range(10):
        _sense(agent, attrs)
        agent.reinforce({Channel.HUNGER: -10.0})
    valence_after = agent.probe(attrs)

    assert valence_after < valence_before
    assert agent.gain_drift() > 0.0


# --- symmetry with the existing health-loss term --------------------------

def test_hunger_loss_and_health_loss_punish_in_the_same_direction(template):
    """Not the same magnitude (CHANNEL_WEIGHTS differ), but the same
    *kind* of update: both should be read as pure punishment, shifting
    gain_drift() away from zero with no reward component involved.
    """
    hunger_agent = fresh_agent(template)
    health_agent = fresh_agent(template)
    attrs = np.ones(16)
    _sense(hunger_agent, attrs)
    _sense(health_agent, attrs)

    hunger_agent.reinforce({Channel.HUNGER: -10.0})
    health_agent.reinforce({Channel.HEALTH: -10.0})

    assert hunger_agent.gain_drift() > 0.0
    assert health_agent.gain_drift() > 0.0


# --- no double-counting: a positive hunger delta is reward only ----------

def test_positive_hunger_delta_still_reads_as_reward_not_punishment(template):
    """Regression guard: max(0, -Δhunger) must stay 0 whenever Δhunger is
    positive, so a real food pickup keeps producing reward exactly as
    before this change, never punishment too.
    """
    agent = fresh_agent(template)
    attrs = np.ones(16)

    approach_idx = template.kc_mbon_approach_edge_idx
    avoid_idx = template.kc_mbon_avoid_edge_idx

    _sense(agent, attrs)
    approach_before = agent.circuit.get_params()[approach_idx].sum()
    avoid_before = agent.circuit.get_params()[avoid_idx].sum()

    agent.reinforce({Channel.HUNGER: 10.0})

    approach_after = agent.circuit.get_params()[approach_idx].sum()
    avoid_after = agent.circuit.get_params()[avoid_idx].sum()

    assert approach_after >= approach_before
    assert avoid_after <= avoid_before


# --- empty/zero deltas remain a no-op -------------------------------------

def test_empty_deltas_is_still_a_no_op(template):
    agent = fresh_agent(template)
    before = agent.circuit.get_params().copy()

    agent.reinforce({})

    assert np.array_equal(agent.circuit.get_params(), before)
    assert agent.reinforcement_events == 0


def _sense(agent: PlasticityAgent, attrs: np.ndarray) -> None:
    """Drives the KC eligibility trace the same way a real tick would,
    without needing a full Observation/Environment -- reinforce() acts on
    whatever KCs `kc_trace` marks eligible, so a call to reinforce() with
    nothing sensed first would have no active pathway to reinforce.
    """
    current = np.zeros(agent.circuit.n)
    current[agent.template.kc_idx] += agent.template.projection @ attrs
    spikes = agent.circuit.step(current)
    kc_spike_mask = np.zeros(agent.circuit.n)
    kc_spike_mask[agent.template.kc_idx] = spikes[agent.template.kc_idx].astype(float)
    agent.kc_trace = agent.kc_trace * 0.7 + kc_spike_mask
