"""Summary analysis of the MaleCNS connectome: composition, neurotransmitters,
and connectivity/degree stats.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"


def _bar(series, title, filename, top=15):
    counts = series.value_counts().head(top)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.iloc[::-1].plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("neuron count")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run(with_connectivity: bool = False) -> None:
    ann = data.load_annotations()
    nt = data.load_neurotransmitters()

    print(f"Neurons (bodies) annotated: {len(ann):,}")
    print(f"Neurons with a neurotransmitter prediction: {len(nt):,}")

    print("\nTop superclasses:")
    print(ann["superclass"].value_counts().head(10).to_string())

    print("\nReconstruction status:")
    print(ann["status"].value_counts().to_string())

    merged_nt = nt["consensus_nt"].fillna(nt["predicted_nt"])
    print("\nNeurotransmitter breakdown:")
    print(merged_nt.value_counts().to_string())

    p1 = _bar(ann["superclass"], "Neurons by superclass", "superclass_counts.png")
    p2 = _bar(merged_nt, "Neurons by neurotransmitter", "neurotransmitter_counts.png")
    print(f"\nSaved plots to {p1} and {p2}")

    if not with_connectivity:
        print("\n(run with --connectivity for degree stats; downloads the ~500MB "
              "weights file)")
        return

    w = data.load_weights()
    print(f"\nSynaptic connections (body pairs): {len(w):,}")
    print(f"Total synapses summed across connections: {w['weight'].sum():,}")

    out_deg = w.groupby("body_pre")["weight"].sum()
    in_deg = w.groupby("body_post")["weight"].sum()

    type_of = ann.set_index("bodyId")["type"]

    def _top_neurons(series, label):
        top = series.sort_values(ascending=False).head(10)
        print(f"\nTop 10 neurons by {label} synapse weight:")
        for body_id, val in top.items():
            print(f"  {body_id:>8}  {type_of.get(body_id, '?'):<15} {val:>10,}")

    _top_neurons(out_deg, "outgoing")
    _top_neurons(in_deg, "incoming")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(out_deg.clip(upper=out_deg.quantile(0.99)), bins=60)
    ax.set_title("Distribution of total outgoing synapse weight per neuron\n"
                  "(clipped at 99th percentile)")
    ax.set_xlabel("summed weight")
    ax.set_ylabel("neuron count")
    fig.tight_layout()
    path = OUTPUT_DIR / "out_degree_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nSaved plot to {path}")
