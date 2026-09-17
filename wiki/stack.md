# Stack

Kept deliberately minimal — a dependency only gets added when something
concrete needs it.

| Tool | Used for | Why this one |
|---|---|---|
| **pandas / pyarrow** | loading the connectome's `.feather` files | the dataset ships in Arrow/feather format; pyarrow reads it, pandas is the natural interface for the tabular filtering/joins `circuit.py` and `analyze.py` do |
| **numpy** | the LIF circuit simulation, vectorized spike propagation | no need for anything heavier — the circuit is a few hundred neurons at most, plain array math is fast enough and keeps the mechanics fully transparent (no framework abstracting away what's happening) |
| **matplotlib** | plots (`analyze`, `simulate` output) | standard, no reason to reach for anything else for static charts |
| **requests / tqdm** | downloading connectome files with a progress bar | small, no-auth HTTP GETs against a public bucket — nothing fancier needed |

## Frontend/serving — decided, not yet built (see `decisions.md` #19)

Sequenced deliberately after the core Python game loop (reproduction
mechanic, live director loop) is working — see #19 for why.

| Tool | Role |
|---|---|
| **FastAPI + WebSocket** | Python backend: streams `Environment` state out as JSON each tick, routes player requests into `director/`. Needs a persistent process (not serverless) — WebSocket requires a long-lived connection. |
| **Next.js + TypeScript** | Frontend app shell — chosen because it's what's already known well, lowering friction. Deployable to Vercel; the Python backend is hosted separately. |
| **Phaser 3** | The actual canvas/tile rendering, mounted client-only inside one Next.js component (`"use client"` + `next/dynamic({ssr: false})` — Phaser touches `window`/canvas, which don't exist during Next's server render). Chosen over hand-rolling Canvas draws or Pygame: built specifically for 2D tile games (tilemaps, sprite animation, camera, particles included), and browser deployment rules out Pygame regardless (Python doesn't run natively in a browser; Pyodide was considered and rejected — heavy given pandas/numpy/pyarrow and the connectome data, and ES training wants many fast server-side rollouts, not client-side ones). |

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
- **No neuprint-python / API token.** Everything used so far comes from
  the public, no-auth flat files in the GCS bucket — see `decisions.md`
  for why that path was chosen over the authenticated API.
