# Stack

Kept deliberately minimal — a dependency only gets added when something
concrete needs it.

| Tool | Used for | Why this one |
|---|---|---|
| **pandas / pyarrow** | loading the connectome's `.feather` files | the dataset ships in Arrow/feather format; pyarrow reads it, pandas is the natural interface for the tabular filtering/joins `circuit.py` and `analyze.py` do |
| **numpy** | the LIF circuit simulation, vectorized spike propagation | no need for anything heavier — the circuit is a few hundred neurons at most, plain array math is fast enough and keeps the mechanics fully transparent (no framework abstracting away what's happening) |
| **matplotlib** | plots (`analyze`, `simulate` output) | standard, no reason to reach for anything else for static charts |
| **requests / tqdm** | downloading connectome files with a progress bar | small, no-auth HTTP GETs against a public bucket — nothing fancier needed |

## Deliberately not used (yet)

- **No deep learning framework (PyTorch/JAX) yet.** Evolution Strategies
  (the first training algorithm — see `decisions.md`) doesn't need
  autodiff; it only needs to run episodes and perturb a parameter vector.
  This will change when REINFORCE is implemented, since that needs a
  differentiable path from parameters to action log-probability — PyTorch
  is the planned choice at that point, not decided in detail yet.
- **No RL library** (Stable-Baselines3, RLlib, etc.). The whole point of
  this project is understanding the mechanics directly — ES and REINFORCE
  are both simple enough to write by hand, and doing so is the actual
  learning goal, not just the fastest path to a working policy.
- **No game engine / pygame yet.** The environment is a plain grid with no
  rendering needs so far; visualization (if added) will be a separate,
  optional module (`world/render.py`), not a dependency of the core sim.
- **No neuprint-python / API token.** Everything used so far comes from
  the public, no-auth flat files in the GCS bucket — see `decisions.md`
  for why that path was chosen over the authenticated API.
