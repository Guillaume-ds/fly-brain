"""Build a small circuit from the real connectome and step it forward with a
toy leaky integrate-and-fire (LIF) model.

This is the piece originally written for the `simulate` terminal demo,
pulled out here so agent.py can reuse the exact same mechanics instead of
duplicating them. Same caveat as before: this is NOT a validated
biophysical model -- synaptic sign is guessed from predicted
neurotransmitter, weights are rescaled synapse counts, no real time
constants. Good enough to carry real wiring into a trainable circuit; not a
research-grade simulation.

Trainable part: each synapse gets a `synaptic_gain` multiplier, initialized
to 1.0 (so a fresh Circuit behaves exactly like the original untrained
demo). Training (ES, later REINFORCE) only ever adjusts these gains --
topology and sign stay fixed, exactly as decided: real wiring is kept,
only synaptic strength is learned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INHIBITORY_NT = {"gaba"}


def build_circuit(
    weights: pd.DataFrame,
    annotations: pd.DataFrame,
    seed_type: str,
    hops: int = 2,
    max_neurons: int = 80,
    edges_per_hop: int = 300,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Expand outward from all neurons of `seed_type`, following the
    strongest outgoing connections, up to `hops` hops and `max_neurons`
    neurons total. Returns (neuron body ids, (pre, post, weight) edges).
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
    edges: list[tuple[int, int, int]] = []

    for _ in range(hops):
        if len(visited) >= max_neurons:
            break
        hop = weights[weights["body_pre"].isin(frontier)]
        hop = hop.sort_values("weight", ascending=False).head(edges_per_hop)
        next_frontier = set()
        for row in hop.itertuples(index=False):
            if len(visited) + len(next_frontier) >= max_neurons and row.body_post not in visited:
                continue
            edges.append((row.body_pre, row.body_post, row.weight))
            if row.body_post not in visited:
                next_frontier.add(row.body_post)
        visited |= next_frontier
        frontier = next_frontier

    edges = [e for e in edges if e[0] in visited and e[1] in visited]
    return sorted(visited), edges


class Circuit:
    """A small LIF spiking network over real connectome neurons/edges."""

    def __init__(
        self,
        neurons: list[int],
        edges: list[tuple[int, int, int]],
        neurotransmitters: pd.Series,
        threshold: float = 1.0,
        leak: float = 0.85,
        propagation_gain: float = 1.5,
    ) -> None:
        self.neurons = neurons
        self.index_of = {b: i for i, b in enumerate(neurons)}
        self.n = len(neurons)
        self.threshold = threshold
        self.leak = leak
        self.propagation_gain = propagation_gain

        max_w = max((e[2] for e in edges), default=1)
        pre, post, base_weight = [], [], []
        for body_pre, body_post, weight in edges:
            sign = -1.0 if neurotransmitters.get(body_pre) in INHIBITORY_NT else 1.0
            pre.append(self.index_of[body_pre])
            post.append(self.index_of[body_post])
            base_weight.append(sign * (weight / max_w))
        self._edge_pre = np.array(pre, dtype=int)
        self._edge_post = np.array(post, dtype=int)
        self._edge_base_weight = np.array(base_weight, dtype=float)

        self.synaptic_gain = np.ones(len(edges))  # trainable, one per synapse
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
            raise ValueError(
                f"expected {self.synaptic_gain.shape} params, got {theta.shape}"
            )
        self.synaptic_gain = theta

    @property
    def num_params(self) -> int:
        return len(self.synaptic_gain)
