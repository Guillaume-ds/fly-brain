"""Wraps a Circuit as a policy for the survival task: turns a world
Observation into input current, steps the circuit, and turns its output
spikes into one of the world's 5 actions.

Stage-1 scope only: this wires up the escape pathway (seeded on DNp01, the
Giant Fiber) reacting to threat_signal/threat_dx/threat_dy. food_signal,
food_dx, food_dy, and hunger are part of Observation but intentionally
unused here -- the foraging pathway is a separate, later addition (stage 3,
added alongside this one via freeze+override, not by changing this class).

Escape decision vs. escape direction are deliberately split:
  - WHETHER to flee is the trained part -- it comes from whether the
    circuit's motor neuron (TTMn, the real jump-muscle motor neuron
    downstream of DNp01) spikes. This is what ES will train.
  - WHICH direction to flee is plain geometry (directly away from the
    threat's position), not something the circuit decides. Real Giant
    Fiber circuitry triggers a stereotyped takeoff, not a steered one, so
    asking the trained circuit to also pick a direction would be
    over-claiming what this pathway biologically does.
"""

from __future__ import annotations

import numpy as np

from . import data
from .circuit import Circuit, build_circuit

MOTOR_TYPE = "TTMn"


class EscapeAgent:
    def __init__(
        self,
        seed_type: str = "DNp01",
        hops: int = 2,
        max_neurons: int = 60,
        edges_per_hop: int = 300,
        stim_gain: float = 1.5,
    ) -> None:
        ann = data.load_annotations()
        nt = data.load_neurotransmitters().set_index("body")["predicted_nt"]
        weights = data.load_weights()

        neurons, edges = build_circuit(
            weights, ann, seed_type,
            hops=hops, max_neurons=max_neurons, edges_per_hop=edges_per_hop,
        )
        self.circuit = Circuit(neurons, edges, nt)
        self.stim_gain = stim_gain

        type_of = ann.set_index("bodyId")["type"].reindex(neurons)
        self.seed_idx = [self.circuit.index_of[b] for b in neurons if type_of[b] == seed_type]
        motor_ids = [b for b in neurons if type_of[b] == MOTOR_TYPE]
        if not motor_ids:
            raise ValueError(
                f"No '{MOTOR_TYPE}' neuron in the built circuit -- increase hops/max_neurons "
                f"so the escape pathway actually reaches a motor neuron."
            )
        self.motor_idx = [self.circuit.index_of[b] for b in motor_ids]

    def reset(self) -> None:
        self.circuit.reset()

    def act(self, obs) -> "Action":
        from world.env import Action  # local import: brain stays independent of world at module load

        current = np.zeros(self.circuit.n)
        current[self.seed_idx] += obs.threat_signal * self.stim_gain
        spikes = self.circuit.step(current)

        if not spikes[self.motor_idx].any():
            return Action.STAY
        return _flee_direction(obs.threat_dx, obs.threat_dy)

    def get_params(self) -> np.ndarray:
        return self.circuit.get_params()

    def set_params(self, theta: np.ndarray) -> None:
        self.circuit.set_params(theta)


def _flee_direction(threat_dx: float, threat_dy: float) -> "Action":
    from world.env import Action

    flee_dx, flee_dy = -threat_dx, -threat_dy
    if abs(flee_dx) >= abs(flee_dy):
        return Action.RIGHT if flee_dx > 0 else Action.LEFT
    return Action.DOWN if flee_dy > 0 else Action.UP
