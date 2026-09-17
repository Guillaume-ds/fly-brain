"""Download and load MaleCNS connectome files from Google's public bucket.

Dataset: HHMI Janelia FlyEM + Google Research + Cambridge + MRC-LMB, "MaleCNS"
v1.0 (released Sept 2026) -- the full wiring diagram of an adult male fruit
fly's central nervous system. Files are public, CC-BY, no account needed.
Browse: https://male-cns.janelia.org/download/
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pandas as pd
import requests
from tqdm import tqdm

BUCKET = "flyem-male-cns"
PREFIX = "v1.0/connectome-data/flat-connectome"
BASE_URL = f"https://storage.googleapis.com/{BUCKET}/{PREFIX}"

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# name -> (filename, approx size in MB) - only the files this project uses.
# Larger sibling files (syn-partners, syn-points, tbar-neurotransmitters) hold
# raw per-synapse points and run into the multi-GB range; skipped here since
# the body-level weights already give per-neuron-pair synapse counts.
FILES = {
    "annotations": ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14),
    "neurotransmitters": ("body-neurotransmitters-male-cns-v1.0.feather", 43),
    "weights": (
        "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        500,
    ),
}


def _download(filename: str) -> pathlib.Path:
    dest = DATA_DIR / filename
    if dest.exists():
        return dest

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    tmp.rename(dest)
    return dest


def fetch(name: str) -> pathlib.Path:
    """Download (if needed) and return the local path for a known dataset file."""
    filename, _ = FILES[name]
    return _download(filename)


def load_annotations() -> pd.DataFrame:
    """Per-neuron metadata: body id, type, class, side, superclass, etc."""
    return pd.read_feather(fetch("annotations"))


def load_neurotransmitters() -> pd.DataFrame:
    """Per-neuron predicted neurotransmitter."""
    return pd.read_feather(fetch("neurotransmitters"))


def load_weights() -> pd.DataFrame:
    """Body-to-body synaptic connection weights (pre id, post id, weight).

    This is the big one (~500MB download, ~1-2GB in memory) -- only fetched
    when actually needed (degree stats, building a circuit for simulation).
    """
    return pd.read_feather(fetch("weights"))


@dataclass
class ConnectomeData:
    annotations: pd.DataFrame
    neurotransmitters: pd.Series  # indexed by body id -> predicted neurotransmitter
    weights: pd.DataFrame


def load_connectome_data() -> ConnectomeData:
    """Load the three connectome files circuit-building needs together."""
    return ConnectomeData(
        annotations=load_annotations(),
        neurotransmitters=load_neurotransmitters().set_index("body")["predicted_nt"],
        weights=load_weights(),
    )
