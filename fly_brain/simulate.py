"""A small, fun "fun-project" demo: pick a real neuron type from the MaleCNS
connectome, pull its real synaptic circuit (real neurons, real connection
weights), and watch a signal propagate through it with a toy leaky
integrate-and-fire (LIF) model.

This is NOT a validated biophysical simulation -- synaptic sign (excitatory
vs. inhibitory) is guessed from the predicted neurotransmitter, weights are
just rescaled synapse counts, and there's no realistic time constant. It's a
lightweight way to *feel* the real wiring diagram in your terminal. For a
research-grade embodied simulation see projects like flybody/flyvis linked
in the README.
"""

from __future__ import annotations

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"

INHIBITORY_NT = {"gaba"}


def build_circuit(
    weights: pd.DataFrame,
    annotations: pd.DataFrame,
    seed_type: str,
    hops: int = 2,
    max_neurons: int = 80,
    edges_per_hop: int = 300,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    seeds = annotations.loc[annotations["type"] == seed_type, "bodyId"].tolist()
    if not seeds:
        matches = annotations.loc[
            annotations["type"].str.contains(seed_type, case=False, na=False),
            "type",
        ].unique()
        hint = f" Did you mean one of: {', '.join(matches[:10])}?" if len(matches) else ""
        raise SystemExit(f"No neurons found with type == '{seed_type}'.{hint}")

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


def run(
    seed_type: str = "DNp01",
    hops: int = 2,
    max_neurons: int = 60,
    steps: int = 40,
    stim_steps: int = 4,
    animate: bool = True,
) -> None:
    print(f"Loading connectome data (weights file is ~500MB, cached after first run)...")
    ann = data.load_annotations()
    nt = data.load_neurotransmitters().set_index("body")["predicted_nt"]
    w = data.load_weights()

    print(f"Building circuit around neuron type '{seed_type}' ({hops} hops)...")
    neurons, edges = build_circuit(w, ann, seed_type, hops=hops, max_neurons=max_neurons)
    idx = {b: i for i, b in enumerate(neurons)}
    n = len(neurons)
    type_of = ann.set_index("bodyId")["type"].reindex(neurons).fillna("?")
    print(f"Circuit has {n} neurons and {len(edges)} connections.\n")

    if n == 0:
        raise SystemExit("Empty circuit -- try a different --seed-type.")

    max_w = max((e[2] for e in edges), default=1)
    signed_edges = []
    for pre, post, weight in edges:
        sign = -1.0 if nt.get(pre) in INHIBITORY_NT else 1.0
        signed_edges.append((idx[pre], idx[post], sign * (weight / max_w)))

    seed_idx = [idx[b] for b in neurons if type_of[b] == seed_type]

    threshold, leak, gain, stim_strength = 1.0, 0.85, 1.5, 1.2
    v = np.zeros(n)
    spikes_prev = np.zeros(n, dtype=bool)
    raster = np.zeros((steps, n), dtype=bool)

    labels = [f"{type_of[b]:<12}" for b in neurons]
    label_width = max(len(lbl) for lbl in labels)

    for t in range(steps):
        current = np.zeros(n)
        if t < stim_steps:
            current[seed_idx] += stim_strength
        for i, j, w_signed in signed_edges:
            if spikes_prev[i]:
                current[j] += w_signed * gain
        v = leak * v + current
        spikes = v >= threshold
        v[spikes] = 0.0
        raster[t] = spikes
        spikes_prev = spikes

        if animate:
            row = "".join("*" if s else "." for s in spikes)
            print(f"t={t:02d} [{row}]  ({spikes.sum()} spiking)")
            sys.stdout.flush()
            time.sleep(0.06)

    print(f"\nNeurons that spiked at least once: {int(raster.any(axis=0).sum())}/{n}")
    total_spikes = raster.sum(axis=0)
    order = np.argsort(-total_spikes)
    print("Most active neurons:")
    for i in order[:10]:
        if total_spikes[i] == 0:
            break
        print(f"  {neurons[i]:>8}  {type_of[neurons[i]]:<15} spiked {total_spikes[i]} / {steps} steps")

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.12)))
    ys, xs = np.where(raster.T)
    ax.scatter(xs, ys, s=8, marker="|")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("timestep")
    ax.set_title(f"Spike raster: '{seed_type}' circuit ({n} neurons, {len(edges)} synapses)")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"raster_{seed_type}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nSaved raster plot to {path}")
