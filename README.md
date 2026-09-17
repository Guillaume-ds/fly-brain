# fly-brain

> This project is growing beyond the exploration tools below into training
> the real connectome (via RL) to survive in a small game world. See
> **[wiki/](wiki/README.md)** for the architecture, the stack, and a running
> log of the decisions behind it.

Small project for exploring and playing with the **MaleCNS connectome** — the
complete wiring diagram of an adult male fruit fly's central nervous system,
released in September 2026 by HHMI Janelia (FlyEM), Google Research, the
University of Cambridge, and the MRC Laboratory of Molecular Biology.

The dataset maps **176,422 neurons and ~125 million synapses** across the
central brain, optic lobes, and ventral nerve cord. It's public (CC-BY), no
account or API token needed — this project pulls it straight from Google's
public storage bucket (`gs://flyem-male-cns/v1.0/...`).

Two things live here:

- **`analyze`** — connectome-wide stats: neuron composition, neurotransmitter
  breakdown, and (optionally) connectivity/degree stats and plots.
- **`simulate`** — a fun toy demo: pick a real neuron type (default: `DNp01`,
  the fly's **Giant Fiber** escape neuron), pull its real downstream circuit
  from the connectome, and watch a signal cascade through it with a simple
  leaky integrate-and-fire model, animated live in your terminal. On the
  default seed it visibly reaches `TTMn` — the jump-muscle motor neuron the
  Giant Fiber actually drives in the real fly's escape reflex.

The simulation is a simplified toy (synaptic sign is guessed from predicted
neurotransmitter, weights are just rescaled synapse counts) — it's meant to
make the real wiring diagram tangible, not to be a validated biophysical
model. For that, see the embodied projects linked below.

## Setup

```bash
git clone https://github.com/Guillaume-ds/fly-brain
cd fly-brain
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# connectome-wide composition + neurotransmitter stats (downloads ~58MB)
python -m fly_brain analyze

# also compute degree/connectivity stats (downloads the ~500MB weights file, cached after)
python -m fly_brain analyze --connectivity

# the fun part: simulate the Giant Fiber escape circuit
python -m fly_brain simulate

# try a different circuit -- any neuron `type` from the dataset works
python -m fly_brain simulate --seed-type LC10 --hops 2 --steps 50
```

Downloaded files are cached in `data/` (gitignored) so you only pay the
download cost once. Plots are written to `output/`.

### Example output

`analyze --connectivity` on the full dataset:

```
Neurons (bodies) annotated: 211,577
Synaptic connections (body pairs): 25,568,639
Total synapses summed across connections: 124,039,080

Top superclasses:
cb_intrinsic         32164
vnc_intrinsic        13161
visual_projection     9201
...
```

See `output/*.png` for the generated charts, including a spike raster from
the Giant Fiber demo.

## Data

| file | size | contents |
|---|---|---|
| `body-annotations-*.feather` | ~14MB | per-neuron type/class/side/status metadata |
| `body-neurotransmitters-*.feather` | ~43MB | per-neuron predicted neurotransmitter |
| `connectome-weights-*-significant-only.feather` | ~500MB | body-to-body synapse weights (25.5M connections) |

Full file listing: https://male-cns.janelia.org/download/
Interactive 3D viewer: https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso

## Related projects

- [natverse/malecns](https://github.com/natverse/malecns) — R access to the same dataset
- [Lulzx/fly-brain](https://github.com/Lulzx/fly-brain) — embodied whole-CNS simulation in the browser (flybody + flyvis)
- [freewangfei/digitalfly](https://github.com/freewangfei/digitalfly) — full connectome driving a physics fly body
- [cobanov/awesome-fly](https://github.com/cobanov/awesome-fly) — curated list of connectome projects
