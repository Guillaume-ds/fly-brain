"""Build a small circuit from the real connectome and step it forward with a
toy leaky integrate-and-fire (LIF) model. NOT a validated biophysical model
-- see wiki/stack.md and wiki/decisions.md for the caveats and for what
training is/isn't allowed to touch (only synaptic_gain; topology and sign
stay fixed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INHIBITORY_NT = {"gaba"}


@dataclass(frozen=True)
class SynapseEdge:
    pre_body_id: int
    post_body_id: int
    weight: int


@dataclass(frozen=True)
class CircuitBlueprint:
    neuron_ids: list[int]
    edges: list[SynapseEdge]


def build_circuit(
    weights: pd.DataFrame,
    annotations: pd.DataFrame,
    seed_type: str,
    hops: int = 2,
    max_neurons: int = 80,
    edges_per_hop: int = 300,
) -> CircuitBlueprint:
    """Expand outward from all neurons of `seed_type`, following the
    strongest outgoing connections, up to `hops` hops and `max_neurons`
    neurons total.
    """
    seeds = annotations.loc[annotations["type"] == seed_type, "bodyId"].tolist()
    if not seeds:
        matches = annotations.loc[
            annotations["type"].str.contains(seed_type, case=False, na=False),
            "type",
        ].unique()
        hint = f" Did you mean one of: {', '.join(matches[:10])}?" if len(matches) else ""
        raise ValueError(f"No neurons found with type == '{seed_type}'.{hint}")

    visited = set(seeds)
    frontier = set(seeds)
    edges: list[SynapseEdge] = []

    for _ in range(hops):
        if len(visited) >= max_neurons:
            break
        hop = weights[weights["body_pre"].isin(frontier)]
        hop = hop.sort_values("weight", ascending=False).head(edges_per_hop)
        next_frontier = set()
        for row in hop.itertuples(index=False):
            if len(visited) + len(next_frontier) >= max_neurons and row.body_post not in visited:
                continue
            edges.append(SynapseEdge(row.body_pre, row.body_post, row.weight))
            if row.body_post not in visited:
                next_frontier.add(row.body_post)
        visited |= next_frontier
        frontier = next_frontier

    edges = [e for e in edges if e.pre_body_id in visited and e.post_body_id in visited]
    return CircuitBlueprint(neuron_ids=sorted(visited), edges=edges)


class Circuit:
    """A small LIF spiking network over real connectome neurons/edges."""

    def __init__(
        self,
        blueprint: CircuitBlueprint,
        neurotransmitters: pd.Series,
        threshold: float = 1.0,
        leak: float = 0.85,
        propagation_gain: float = 1.5,
    ) -> None:
        self.neurons = blueprint.neuron_ids
        self.index_of = {body_id: i for i, body_id in enumerate(self.neurons)}
        self.n = len(self.neurons)
        self.threshold = threshold
        self.leak = leak
        self.propagation_gain = propagation_gain

        max_weight = max((edge.weight for edge in blueprint.edges), default=1)
        pre_indices, post_indices, base_weights = [], [], []
        for edge in blueprint.edges:
            sign = -1.0 if neurotransmitters.get(edge.pre_body_id) in INHIBITORY_NT else 1.0
            pre_indices.append(self.index_of[edge.pre_body_id])
            post_indices.append(self.index_of[edge.post_body_id])
            base_weights.append(sign * (edge.weight / max_weight))

        # Derived, tightly-coupled internal wiring -- kept private on purpose:
        # nothing outside Circuit should touch these directly, only
        # synaptic_gain is meant to be read/written (via get_params/set_params).
        self._edge_pre = np.array(pre_indices, dtype=int)
        self._edge_post = np.array(post_indices, dtype=int)
        self._edge_base_weight = np.array(base_weights, dtype=float)

        self.synaptic_gain = np.ones(len(blueprint.edges))  # trainable, one per synapse
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(self.n)
        self.spikes_prev = np.zeros(self.n, dtype=bool)

    def step(self, external_current: np.ndarray) -> np.ndarray:
        """Advance one tick given external input current per neuron.
        Returns the boolean spike vector for this tick.
        """
        current = external_current.copy()
        active = self.spikes_prev[self._edge_pre]
        if active.any():
            contributions = (
                self._edge_base_weight[active]
                * self.synaptic_gain[active]
                * self.propagation_gain
            )
            np.add.at(current, self._edge_post[active], contributions)

        self.v = self.leak * self.v + current
        spikes = self.v >= self.threshold
        self.v[spikes] = 0.0
        self.spikes_prev = spikes
        return spikes

    def get_params(self) -> np.ndarray:
        return self.synaptic_gain.copy()

    def set_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.synaptic_gain.shape:
            raise ValueError(f"expected {self.synaptic_gain.shape} params, got {theta.shape}")
        self.synaptic_gain = theta

    @property
    def num_params(self) -> int:
        return len(self.synaptic_gain)
