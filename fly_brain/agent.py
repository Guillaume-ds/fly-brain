"""EscapeAgent: wraps a Circuit as a policy for the survival task -- turns
a world Observation into input current, steps the circuit, and turns its
output spikes into one of the world's 5 actions.

Only wires up the escape pathway (seeded on DNp01) for now; food/hunger
are part of Observation but unused here (the foraging pathway is a
separate later addition, see wiki/roadmap.md). Whether to flee is the
trained part (TTMn spiking); which direction is plain geometry, not the
circuit's decision -- see wiki/decisions.md for why.

Since #22, Observation carries an anonymous Percept list rather than a
named `threat_signal` field, so DNp01's stimulus is computed the same
way the Result registry computes item effects (see world/results.py):
cosine similarity between each nearby percept's attribute vector and a
`danger` reference vector, encoded with the same ItemEncoder used
everywhere else. This is the resolution to the gap logged in
decisions.md #24 -- reusing the mechanism already agreed on, rather than
reintroducing a named side-channel just for threats.

Circuit-building (loading the connectome, expanding the circuit, finding
seed/motor neurons) is expensive and identical for every fly of the same
seed_type -- separated into EscapeCircuitTemplate so a colony of many
flies builds it once and each fly gets a cheap EscapeAgent instance with
its own Circuit state (genome via set_params(), membrane potential).
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from world.env import Action, Observation, Percept
from world.items import ItemEncoder

from .circuit import Circuit, CircuitBlueprint, build_circuit
from .data import load_connectome_data

logger = logging.getLogger(__name__)

MOTOR_TYPE = "TTMn"
DANGER_DESCRIPTION = "a dangerous, fast predator"


def flee_direction(dx: float, dy: float) -> Action:
    flee_dx, flee_dy = -dx, -dy
    if abs(flee_dx) >= abs(flee_dy):
        return Action.RIGHT if flee_dx > 0 else Action.LEFT
    return Action.DOWN if flee_dy > 0 else Action.UP


def sense_danger(nearby: list[Percept], danger_vector: np.ndarray) -> tuple[float, float, float]:
    """Danger strength sums across every percept that resembles `danger`
    at all (more simultaneous threats should never look less urgent than
    one), each weighted by proximity; flee direction points away from
    the single nearest above-threshold percept, since fleeing needs one
    concrete direction. Bounded in [0, 1] per percept, same scale the
    old fixed `threat_signal` field used.
    """
    total_strength = 0.0
    nearest_distance = None
    flee_dx = flee_dy = 0.0
    for percept in nearby:
        similarity = max(0.0, float(np.dot(percept.attributes, danger_vector)))
        if similarity <= 0.0:
            continue
        proximity = 1.0 / (1.0 + percept.distance)
        total_strength += similarity * proximity
        if nearest_distance is None or percept.distance < nearest_distance:
            nearest_distance = percept.distance
            flee_dx, flee_dy = percept.dx, percept.dy
    return flee_dx, flee_dy, total_strength


def neuron_indices_of_type(
    neuron_ids: list[int], type_of: pd.Series, index_of: dict[int, int], target_type: str, required: bool = False
) -> list[int]:
    matching = [body_id for body_id in neuron_ids if type_of[body_id] == target_type]
    if required and not matching:
        raise ValueError(
            f"No '{target_type}' neuron in the built circuit -- increase hops/max_neurons "
            f"so the escape pathway actually reaches a motor neuron."
        )
    return [index_of[body_id] for body_id in matching]


@dataclass
class EscapeCircuitTemplate:
    blueprint: CircuitBlueprint
    neurotransmitters: pd.Series
    seed_idx: list[int]
    motor_idx: list[int]
    danger_vector: np.ndarray  # see sense_danger() above and decisions.md #24


def build_escape_template(
    encoder: ItemEncoder,
    seed_type: str = "DNp01",
    hops: int = 2,
    max_neurons: int = 60,
    edges_per_hop: int = 300,
    danger_description: str = DANGER_DESCRIPTION,
) -> EscapeCircuitTemplate:
    connectome = load_connectome_data()
    blueprint = build_circuit(
        connectome.weights, connectome.annotations, seed_type,
        hops=hops, max_neurons=max_neurons, edges_per_hop=edges_per_hop,
    )
    type_of = connectome.annotations.set_index("bodyId")["type"].reindex(blueprint.neuron_ids)
    index_of = {body_id: i for i, body_id in enumerate(blueprint.neuron_ids)}

    seed_idx = neuron_indices_of_type(blueprint.neuron_ids, type_of, index_of, seed_type)
    motor_idx = neuron_indices_of_type(blueprint.neuron_ids, type_of, index_of, MOTOR_TYPE, required=True)
    return EscapeCircuitTemplate(
        blueprint=blueprint, neurotransmitters=connectome.neurotransmitters,
        seed_idx=seed_idx, motor_idx=motor_idx, danger_vector=encoder.encode(danger_description),
    )


class EscapeAgent:
    def __init__(self, template: EscapeCircuitTemplate, stim_gain: float = 1.5) -> None:
        self.circuit = Circuit(template.blueprint, template.neurotransmitters)
        self.stim_gain = stim_gain
        self.seed_idx = template.seed_idx
        self.motor_idx = template.motor_idx
        self.danger_vector = template.danger_vector

    def reset(self) -> None:
        self.circuit.reset()

    def decide(self, obs: Observation) -> Action | None:
        """None means "no escape override this tick" (TTMn didn't spike)
        -- distinct from Action.STAY, which is a real decision. Lets a
        combined brain (training/colony.py, decisions.md #26) fall
        through to the plasticity circuit's own decision instead of
        forcing STAY whenever the escape reflex has nothing to say.
        """
        flee_dx, flee_dy, danger_strength = sense_danger(obs.nearby, self.danger_vector)

        current = np.zeros(self.circuit.n)
        current[self.seed_idx] += danger_strength * self.stim_gain
        spikes = self.circuit.step(current)

        if not spikes[self.motor_idx].any():
            return None
        return flee_direction(flee_dx, flee_dy)

    def act(self, obs: Observation) -> Action:
        decision = self.decide(obs)
        return decision if decision is not None else Action.STAY

    def get_params(self) -> np.ndarray:
        return self.circuit.get_params()

    def set_params(self, theta: np.ndarray) -> None:
        self.circuit.set_params(theta)


def load_starting_gains(
    template: EscapeCircuitTemplate, checkpoint_path: pathlib.Path | None = None
) -> np.ndarray:
    """A trained checkpoint gives a strong starting genome (see
    wiki/decisions.md #13 for why ES-then-reproduction, not one or the
    other, and #4 for why curriculum training chains stage N from stage
    N-1's checkpoint rather than starting fresh each time); falls back
    to untrained (real biology, gain=1.0) if none is given or found.

    Lives here, not in training/ or game/, so both can call it without
    either importing from the other -- training/run.py uses it to chain
    curriculum stages, game/colony.py uses it to bootstrap a live colony
    from a finished stage's weights.
    """
    if checkpoint_path is not None and checkpoint_path.exists():
        logger.info("Loaded starting genome from %s", checkpoint_path)
        return np.load(checkpoint_path)
    logger.info("No checkpoint given/found -- starting from untrained (real biology, gain=1.0)")
    return np.ones(len(template.blueprint.edges))
