"""Summary analysis of the MaleCNS connectome: composition, neurotransmitters,
and connectivity/degree stats.
"""

from __future__ import annotations

import logging
import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data

logger = logging.getLogger(__name__)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"


def run(with_connectivity: bool = False) -> None:
    annotations = data.load_annotations()
    neurotransmitters = data.load_neurotransmitters()

    report_composition(annotations)
    merged_nt = report_neurotransmitters(neurotransmitters)
    save_composition_plots(annotations, merged_nt)

    if not with_connectivity:
        logger.info("(run with --connectivity for degree stats; downloads the ~500MB weights file)")
        return

    weights = data.load_weights()
    report_connectivity(weights, annotations)


def report_composition(annotations: pd.DataFrame) -> None:
    logger.info("Neurons (bodies) annotated: %s", f"{len(annotations):,}")
    logger.info("Top superclasses:\n%s", annotations["superclass"].value_counts().head(10).to_string())
    logger.info("Reconstruction status:\n%s", annotations["status"].value_counts().to_string())


def report_neurotransmitters(neurotransmitters: pd.DataFrame) -> pd.Series:
    logger.info("Neurons with a neurotransmitter prediction: %s", f"{len(neurotransmitters):,}")
    merged_nt = neurotransmitters["consensus_nt"].fillna(neurotransmitters["predicted_nt"])
    logger.info("Neurotransmitter breakdown:\n%s", merged_nt.value_counts().to_string())
    return merged_nt


def save_composition_plots(annotations: pd.DataFrame, merged_nt: pd.Series) -> None:
    superclass_path = bar_chart(annotations["superclass"], "Neurons by superclass", "superclass_counts.png")
    nt_path = bar_chart(merged_nt, "Neurons by neurotransmitter", "neurotransmitter_counts.png")
    logger.info("Saved plots to %s and %s", superclass_path, nt_path)


def report_connectivity(weights: pd.DataFrame, annotations: pd.DataFrame) -> None:
    logger.info("Synaptic connections (body pairs): %s", f"{len(weights):,}")
    logger.info("Total synapses summed across connections: %s", f"{weights['weight'].sum():,}")

    type_of = annotations.set_index("bodyId")["type"]
    out_degree = weights.groupby("body_pre")["weight"].sum()
    in_degree = weights.groupby("body_post")["weight"].sum()
    report_top_neurons(out_degree, type_of, "outgoing")
    report_top_neurons(in_degree, type_of, "incoming")

    save_degree_distribution_plot(out_degree)


def report_top_neurons(degree: pd.Series, type_of: pd.Series, label: str) -> None:
    top = degree.sort_values(ascending=False).head(10)
    lines = [f"  {body_id:>8}  {type_of.get(body_id, '?'):<15} {value:>10,}" for body_id, value in top.items()]
    logger.info("Top 10 neurons by %s synapse weight:\n%s", label, "\n".join(lines))


def save_degree_distribution_plot(out_degree: pd.Series) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(out_degree.clip(upper=out_degree.quantile(0.99)), bins=60)
    ax.set_title("Distribution of total outgoing synapse weight per neuron\n(clipped at 99th percentile)")
    ax.set_xlabel("summed weight")
    ax.set_ylabel("neuron count")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "out_degree_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved plot to %s", path)
    return path


def bar_chart(series: pd.Series, title: str, filename: str, top: int = 15) -> pathlib.Path:
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
