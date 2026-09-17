# Roadmap

## Status

| Piece | Status |
|---|---|
| Connectome data access (`fly_brain/data.py`) | done |
| Connectome exploration (`analyze`, `simulate` commands) | done |
| `world/env.py` — grid, hunger, food/threats, curriculum toggles | done, tested |
| `fly_brain/circuit.py` — trainable LIF circuit over real connectome | done, tested |
| `fly_brain/agent.py` — `EscapeAgent` wiring circuit into the world | done, integration-tested (untrained weights) |
| `training/curriculum.py` — stage configs | **not started** |
| `training/trainer.py` — ES training loop | **not started** |
| Stage 1 training run (clean escape) | not started |
| Stage 2 training run (noisy escape) | not started |
| Stage 3: freeze escape + train foraging pathway (transfer) | not started |
| REINFORCE implementation (comparison to ES) | not started |
| Open-world / random generation / distinct trap types | not started (explicitly deferred, see `decisions.md` #9) |

## Immediate next step

`training/trainer.py`: an Evolution Strategies loop that
1. perturbs `EscapeAgent.get_params()` (the `synaptic_gain` vector),
2. rolls out an episode per perturbation via `Environment` + `agent.act()`,
3. uses total survival ticks (or a shaped variant) as fitness,
4. updates the parameter vector toward better-scoring perturbations.

Alongside it, `training/curriculum.py` for the stage-1 config: threats
only, no food (`food_enabled=False`), so the escape circuit trains in
isolation before stage 2 adds distractor noise.

## After that

- Stage 1 training run, checked against the untrained baseline already
  captured (see `decisions.md` #7 — untrained real weights already avoid
  threats reasonably; the open question training answers is how much
  better real, tuned weights can do).
- Stage 2: same environment, harder/noisier threat signal.
- Stage 3: freeze stage-2 weights, add and train a new foraging pathway,
  combine via the override rule from `decisions.md` #5.
- REINFORCE implementation, compared against ES on the same stage-1 task.
- Only after all of the above: revisit the open-world / random-generation
  vision from the original project pitch.
