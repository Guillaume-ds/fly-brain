"""EscapeAgent: wraps a Circuit as a policy for the survival task -- turns
a world Observation into input current, steps the circuit, and turns its
output spikes into one of the world's 5 actions.

Only wires up the escape pathway (seeded on DNp01) for now; food/hunger
are part of Observation but unused here (the foraging pathway is a
separate later addition, see wiki/roadmap.md). Whether to flee is the
trained part (TTMn spiking); which direction is plain geometry, not the
circuit's decision -- see wiki/decisions.md for why.

Circuit-building (loading the connectome, expanding the circuit, finding
seed/motor neurons) is expensive and identical for every fly of the same
seed_type -- separated into EscapeCircuitTemplate so a colony of many
flies builds it once and each fly gets a cheap EscapeAgent instance with
its own Circuit state (genome via set_params(), membrane potential).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from world.env import Action, Observation

from .circuit import Circuit, CircuitBlueprint, build_circuit
from .data import load_connectome_data

MOTOR_TYPE = "TTMn"


def flee_direction(threat_dx: float, threat_dy: float) -> Action:
    flee_dx, flee_dy = -threat_dx, -threat_dy
    if abs(flee_dx) >= abs(flee_dy):
        return Action.RIGHT if flee_dx > 0 else Action.LEFT
    return Action.DOWN if flee_dy > 0 else Action.UP


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


def build_escape_template(
    seed_type: str = "DNp01",
    hops: int = 2,
    max_neurons: int = 60,
    edges_per_hop: int = 300,
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
        seed_idx=seed_idx, motor_idx=motor_idx,
    )


class EscapeAgent:
    def __init__(self, template: EscapeCircuitTemplate, stim_gain: float = 1.5) -> None:
        self.circuit = Circuit(template.blueprint, template.neurotransmitters)
        self.stim_gain = stim_gain
        self.seed_idx = template.seed_idx
        self.motor_idx = template.motor_idx

    def reset(self) -> None:
        self.circuit.reset()

    def act(self, obs: Observation) -> Action:
        current = np.zeros(self.circuit.n)
        current[self.seed_idx] += obs.threat_signal * self.stim_gain
        spikes = self.circuit.step(current)

        if not spikes[self.motor_idx].any():
            return Action.STAY
        return flee_direction(obs.threat_dx, obs.threat_dy)

    def get_params(self) -> np.ndarray:
        return self.circuit.get_params()

    def set_params(self, theta: np.ndarray) -> None:
        self.circuit.set_params(theta)
