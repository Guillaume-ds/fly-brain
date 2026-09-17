"""A small, fun terminal demo: pick a real neuron type from the MaleCNS
connectome, pull its real synaptic circuit, and watch a signal propagate
through it via circuit.Circuit's toy LIF model.
"""

from __future__ import annotations

import logging
import pathlib
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .circuit import Circuit, build_circuit
from .data import load_connectome_data

logger = logging.getLogger(__name__)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
STIM_STRENGTH = 1.2


@dataclass
class RasterResult:
    spikes: np.ndarray  # (steps, n) bool
    labels: list[str]
    neuron_ids: list[int]


def run(
    seed_type: str = "DNp01",
    hops: int = 2,
    max_neurons: int = 60,
    steps: int = 40,
    stim_steps: int = 4,
    animate: bool = True,
) -> None:
    logger.info("Loading connectome data (weights file is ~500MB, cached after first run)...")
    connectome = load_connectome_data()

    logger.info("Building circuit around neuron type '%s' (%d hops)...", seed_type, hops)
    blueprint = build_circuit(connectome.weights, connectome.annotations, seed_type, hops=hops, max_neurons=max_neurons)
    if not blueprint.neuron_ids:
        raise SystemExit("Empty circuit -- try a different --seed-type.")
    logger.info("Circuit has %d neurons and %d connections.", len(blueprint.neuron_ids), len(blueprint.edges))

    circuit = Circuit(blueprint, connectome.neurotransmitters)
    type_of = connectome.annotations.set_index("bodyId")["type"].reindex(blueprint.neuron_ids).fillna("?")
    seed_idx = [circuit.index_of[b] for b in blueprint.neuron_ids if type_of[b] == seed_type]

    result = run_simulation(circuit, seed_idx, type_of, blueprint.neuron_ids, steps, stim_steps, animate)
    report_most_active_neurons(result, steps)
    save_raster_plot(result, seed_type, len(blueprint.edges))


def run_simulation(
    circuit: Circuit,
    seed_idx: list[int],
    type_of,
    neuron_ids: list[int],
    steps: int,
    stim_steps: int,
    animate: bool,
) -> RasterResult:
    raster = np.zeros((steps, circuit.n), dtype=bool)
    labels = [f"{type_of[b]:<12}" for b in neuron_ids]

    for t in range(steps):
        current = np.zeros(circuit.n)
        if t < stim_steps:
            current[seed_idx] += STIM_STRENGTH
        spikes = circuit.step(current)
        raster[t] = spikes

        if animate:
            row = "".join("*" if s else "." for s in spikes)
            logger.info("t=%02d [%s]  (%d spiking)", t, row, spikes.sum())
            time.sleep(0.06)

    return RasterResult(spikes=raster, labels=labels, neuron_ids=neuron_ids)


def report_most_active_neurons(result: RasterResult, steps: int) -> None:
    total_spikes = result.spikes.sum(axis=0)
    ever_spiked = int(result.spikes.any(axis=0).sum())
    logger.info("Neurons that spiked at least once: %d/%d", ever_spiked, len(result.neuron_ids))

    logger.info("Most active neurons:")
    for i in np.argsort(-total_spikes)[:10]:
        if total_spikes[i] == 0:
            break
        logger.info("  %8d  %-12s spiked %d / %d steps", result.neuron_ids[i], result.labels[i].strip(), total_spikes[i], steps)


def save_raster_plot(result: RasterResult, seed_type: str, num_edges: int) -> pathlib.Path:
    n = len(result.neuron_ids)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.12)))
    ys, xs = np.where(result.spikes.T)
    ax.scatter(xs, ys, s=8, marker="|")
    ax.set_yticks(range(n))
    ax.set_yticklabels(result.labels, fontsize=6)
    ax.set_xlabel("timestep")
    ax.set_title(f"Spike raster: '{seed_type}' circuit ({n} neurons, {num_edges} synapses)")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"raster_{seed_type}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved raster plot to %s", path)
    return path
