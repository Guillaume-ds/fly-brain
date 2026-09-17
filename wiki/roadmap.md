# Roadmap

## Current vision (see `decisions.md` #13)

A fly colony survives (and reproduces) against spider/food pressure that a
human, via an LLM translator, can dial up or down. Not a building game —
see #13 for why that was dropped.

## Status

| Piece | Status |
|---|---|
| Connectome data access (`fly_brain/data.py`) | done |
| Connectome exploration (`analyze`, `simulate` commands) | done |
| `world/env.py` — grid, hunger, food/threats | done, tested — **about to change**: fixed-count placement → continuous spawn-rate parameters (`decisions.md` #14) |
| `fly_brain/circuit.py` — trainable LIF circuit over real connectome | done, tested |
| `fly_brain/agent.py` — `EscapeAgent` wiring circuit into the world | done, integration-tested (untrained weights) |
| `training/curriculum.py` — stage configs | done (stage 1 only, will need updating for continuous spawning) |
| `training/trainer.py` / `training/run.py` — ES training loop | done, mechanically tested |
| Stage 1 training run (clean escape) | was blocked on `decisions.md` #12 — expected to unblock once continuous spawning lands, not yet re-verified |
| `director/` — swappable LLM world-controller layer | **designing now** (see `decisions.md` for the entry once settled) |
| Reproduction mechanic (offspring = parent gains + ES perturbation, survival = selection) | designed in #13, not implemented |
| Stage 2 / stage 3 (noisy escape, forage transfer) | not started |
| REINFORCE implementation (comparison to ES) | not started |
| Open-world / random generation / distinct trap types, spiderweb v2 mechanic | not started, explicitly deferred (`decisions.md` #9) |

## Immediate next steps, in order

1. `world/env.py`: replace fixed-count-at-reset placement with the two
   bounded, independent continuous spawn-rate parameters
   (`spider_spawn_rate`, `food_spawn_rate`) from `decisions.md` #14, plus
   plain methods to adjust each.
2. Re-run stage-1 ES training and confirm it now has real fitness
   variance (the actual test of whether #14 fixes #12).
3. `director/`: the swappable LLM world-controller layer sitting on top
   of those methods — architecture being discussed now.
4. Reproduction mechanic from #13.

## After that

- Stage 2: same environment, harder/noisier threat signal.
- Stage 3: freeze stage-2 weights, add and train a new foraging pathway,
  combine via the override rule from `decisions.md` #5.
- REINFORCE implementation, compared against ES on the same stage-1 task.
- **Open-ended world** — replace the fixed-size grid with true open-ended,
  randomly-generated terrain, plus the spiderweb v2 mechanic (spiders drop
  webs on death; web = distinct immobilize effect, not instant death).
  Comes only after the above is proven out (`decisions.md` #9).
