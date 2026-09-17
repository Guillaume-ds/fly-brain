"""A small, fun terminal demo: pick a real neuron type from the MaleCNS
connectome, pull its real synaptic circuit, and watch a signal propagate
through it via circuit.Circuit's toy LIF model.
"""

from __future__ import annotations

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data
from .circuit import Circuit, build_circuit

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"


def run(
    seed_type: str = "DNp01",
    hops: int = 2,
    max_neurons: int = 60,
    steps: int = 40,
    stim_steps: int = 4,
    animate: bool = True,
) -> None:
    print("Loading connectome data (weights file is ~500MB, cached after first run)...")
    ann = data.load_annotations()
    nt = data.load_neurotransmitters().set_index("body")["predicted_nt"]
    w = data.load_weights()

    print(f"Building circuit around neuron type '{seed_type}' ({hops} hops)...")
    neurons, edges = build_circuit(w, ann, seed_type, hops=hops, max_neurons=max_neurons)
    if not neurons:
        raise SystemExit("Empty circuit -- try a different --seed-type.")

    type_of = ann.set_index("bodyId")["type"].reindex(neurons).fillna("?")
    print(f"Circuit has {len(neurons)} neurons and {len(edges)} connections.\n")

    circuit = Circuit(neurons, edges, nt)
    seed_idx = [circuit.index_of[b] for b in neurons if type_of[b] == seed_type]
    stim_strength = 1.2

    raster = np.zeros((steps, circuit.n), dtype=bool)
    labels = [f"{type_of[b]:<12}" for b in neurons]

    for t in range(steps):
        current = np.zeros(circuit.n)
        if t < stim_steps:
            current[seed_idx] += stim_strength
        spikes = circuit.step(current)
        raster[t] = spikes

        if animate:
            row = "".join("*" if s else "." for s in spikes)
            print(f"t={t:02d} [{row}]  ({spikes.sum()} spiking)")
            sys.stdout.flush()
            time.sleep(0.06)

    print(f"\nNeurons that spiked at least once: {int(raster.any(axis=0).sum())}/{circuit.n}")
    total_spikes = raster.sum(axis=0)
    order = np.argsort(-total_spikes)
    print("Most active neurons:")
    for i in order[:10]:
        if total_spikes[i] == 0:
            break
        print(f"  {neurons[i]:>8}  {type_of[neurons[i]]:<15} spiked {total_spikes[i]} / {steps} steps")

    fig, ax = plt.subplots(figsize=(10, max(4, circuit.n * 0.12)))
    ys, xs = np.where(raster.T)
    ax.scatter(xs, ys, s=8, marker="|")
    ax.set_yticks(range(circuit.n))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("timestep")
    ax.set_title(f"Spike raster: '{seed_type}' circuit ({circuit.n} neurons, {len(edges)} synapses)")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"raster_{seed_type}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nSaved raster plot to {path}")
