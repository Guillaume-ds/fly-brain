"""EscapeAgent: wraps a Circuit as a policy for the survival task -- turns
a world Observation into input current, steps the circuit, and turns its
output spikes into one of the world's 5 actions.

Only wires up the escape pathway (seeded on DNp01) for now; food/hunger
are part of Observation but unused here (the foraging pathway is a
separate later addition, see wiki/roadmap.md). Whether to flee is the
trained part (TTMn spiking); which direction is plain geometry, not the
circuit's decision -- see wiki/decisions.md for why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from world.env import Action, Observation

from .circuit import Circuit, build_circuit
from .data import load_connectome_data

MOTOR_TYPE = "TTMn"


def flee_direction(threat_dx: float, threat_dy: float) -> Action:
    flee_dx, flee_dy = -threat_dx, -threat_dy
    if abs(flee_dx) >= abs(flee_dy):
        return Action.RIGHT if flee_dx > 0 else Action.LEFT
    return Action.DOWN if flee_dy > 0 else Action.UP


class EscapeAgent:
    def __init__(
        self,
        seed_type: str = "DNp01",
        hops: int = 2,
        max_neurons: int = 60,
        edges_per_hop: int = 300,
        stim_gain: float = 1.5,
    ) -> None:
        connectome = load_connectome_data()
        blueprint = build_circuit(
            connectome.weights, connectome.annotations, seed_type,
            hops=hops, max_neurons=max_neurons, edges_per_hop=edges_per_hop,
        )
        self.circuit = Circuit(blueprint, connectome.neurotransmitters)
        self.stim_gain = stim_gain

        type_of = connectome.annotations.set_index("bodyId")["type"].reindex(blueprint.neuron_ids)
        self.seed_idx = self.neuron_indices_of_type(blueprint.neuron_ids, type_of, seed_type)
        self.motor_idx = self.neuron_indices_of_type(
            blueprint.neuron_ids, type_of, MOTOR_TYPE, required=True
        )

    def neuron_indices_of_type(
        self, neuron_ids: list[int], type_of: pd.Series, target_type: str, required: bool = False
    ) -> list[int]:
        matching = [body_id for body_id in neuron_ids if type_of[body_id] == target_type]
        if required and not matching:
            raise ValueError(
                f"No '{target_type}' neuron in the built circuit -- increase hops/max_neurons "
                f"so the escape pathway actually reaches a motor neuron."
            )
        return [self.circuit.index_of[body_id] for body_id in matching]

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
