# Roadmap

## Status

| Piece | Status |
|---|---|
| Connectome data access (`fly_brain/data.py`) | done |
| Connectome exploration (`analyze`, `simulate` commands) | done |
| `world/env.py` — grid, hunger, food/threats, curriculum toggles | done, tested |
| `fly_brain/circuit.py` — trainable LIF circuit over real connectome | done, tested |
| `fly_brain/agent.py` — `EscapeAgent` wiring circuit into the world | done, integration-tested (untrained weights) |
| `training/curriculum.py` — stage configs | done (stage 1 only) |
| `training/trainer.py` / `training/run.py` — ES training loop | done, mechanically tested (correctly implements the OpenAI-ES update; see `decisions.md` #12 for why a *real* training run is still blocked) |
| Stage 1 training run (clean escape) | **blocked** — see `decisions.md` #12 |
| Stage 2 training run (noisy escape) | not started |
| Stage 3: freeze escape + train foraging pathway (transfer) | not started |
| REINFORCE implementation (comparison to ES) | not started |
| Open-world / random generation / distinct trap types | not started (explicitly deferred, see `decisions.md` #9) |

## Immediate next step

Resolve the open question in `decisions.md` #12 (threats likely need to
move) and re-tune the stage-1 config, then run stage 1 for real.

## After that

- Stage 1 training run once unblocked.
- Stage 2: same environment, harder/noisier threat signal.
- Stage 3: freeze stage-2 weights, add and train a new foraging pathway,
  combine via the override rule from `decisions.md` #5.
- REINFORCE implementation, compared against ES on the same stage-1 task.
- **Open-ended world** — replace the fixed-size grid with the original
  vision: open-ended, randomly-generated, with traps (e.g. spiderweb-style,
  distinct from plain threats) alongside threats and fruit. Comes only
  after the stages above, once training on the simple world is proven out
  (see `decisions.md` #9).
