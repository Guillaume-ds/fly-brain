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
| `world/env.py` — multi-fly, continuous spawn-rate params, threat movement | done, tested (`decisions.md` #14, #15, #20) |
| `fly_brain/circuit.py` — trainable LIF circuit over real connectome | done, tested |
| `fly_brain/agent.py` — `EscapeAgent` wiring circuit into the world | done, integration-tested (untrained weights) |
| `training/curriculum.py` — stage configs | done (stage 1 only) |
| `training/trainer.py` / `training/run.py` — ES training loop | done, mechanically tested |
| Stage 1 training run (clean escape) | **unblocked and verified** — real `population_reward_std` every iteration (`decisions.md` #15); 15-iteration smoke test only, a longer real run is still future work |
| `director/` — swappable LLM world-controller layer | done; registry → rule-based controller → `Environment` verified end to end; `ClaudeController` built to spec but **not tested live** (no API credentials in this environment) — see `decisions.md` #16 |
| Reproduction mechanic — multi-fly `Environment`, stochastic trigger, parent cost | **done, tested** (`decisions.md` #20) |
| `training/colony.py` — per-fly circuits/genomes, offspring genome creation, headless colony runner | **done, tested** (`decisions.md` #21) — real natural birth observed in a full run, colony driven by an actual trained circuit end to end |
| No foraging behavior — flies only eat food they wander into by luck | known limitation, not a bug (`decisions.md` #21); stage 3's job |
| Live game loop (`training/live_run.py`: `director/` commands affecting a running `Colony` continuously) | **done, tested** (`decisions.md` #23) |
| Stage 2 / stage 3 (noisy escape, forage transfer) | not started |
| REINFORCE implementation (comparison to ES) | not started |
| Open-world / random generation / distinct trap types, spiderweb v2 mechanic | not started, explicitly deferred (`decisions.md` #9) |
| Frontend: FastAPI+WebSocket backend, Next.js/TypeScript + Phaser 3 rendering | **decided** (`decisions.md` #19), not started — deliberately deferred until the core Python game loop works end to end |

## Immediate next steps, in order

1. ~~`world/env.py`: continuous spawn-rate parameters~~ done.
2. ~~Re-run stage-1 ES training, confirm real fitness variance~~ done —
   see `decisions.md` #15.
3. ~~`director/`: the swappable LLM world-controller layer~~ done —
   registry + `RuleBasedController` verified end to end. **Still needed:
   a live test of `ClaudeController` with a real API key** (not possible
   in this environment).
4. ~~Reproduction mechanic~~ done — see `decisions.md` #20.
5. ~~`training/colony.py`: wire real per-fly circuits/genomes~~ done —
   see `decisions.md` #21. Verified with a headless CLI run, not yet
   connected to `director/` or anything live/interactive.
6. ~~Live game loop~~ done — see `decisions.md` #23.
   `training/live_run.py` (`python -m training.live_run`) connects
   `director/`'s `WorldController` to a continuously-ticking `Colony`: a
   background thread queues player requests, the main loop drains and
   applies them each tick without pausing the world. This is what "the
   core Python game loop" in `decisions.md` #19's sequencing condition
   referred to — the frontend is now unblocked.

All six immediate next steps above are done. What's next is a choice
between two independent tracks, not a fixed order: implementing the
sensing/learning redesign (`decisions.md` #22, anonymous percepts + the
real KC/MBON/DAN plasticity circuit + encoder-based item authoring —
design-complete, nothing built yet), or starting the frontend now that
its blocker (this section) is cleared.

## After that

- Stage 2: same environment, harder/noisier threat signal.
- Stage 3: freeze stage-2 weights, add and train a new foraging pathway,
  combine via the override rule from `decisions.md` #5.
- REINFORCE implementation, compared against ES on the same stage-1 task.
- **Open-ended world** — replace the fixed-size grid with true open-ended,
  randomly-generated terrain, plus the spiderweb v2 mechanic (spiders drop
  webs on death; web = distinct immobilize effect, not instant death).
  Comes only after the above is proven out (`decisions.md` #9).
- **Frontend** — FastAPI+WebSocket backend, Next.js/TypeScript + Phaser 3
  rendering (`decisions.md` #19). The core Python game loop it was
  deferred until (reproduction mechanic + a live director loop) is now
  done (`decisions.md` #23) — unblocked, not yet started.
