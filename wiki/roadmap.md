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
| `world/env.py` — multi-fly, continuous spawn-rate params, threat movement, anonymous `Percept` observation | done, tested (`decisions.md` #14, #15, #20, #25) |
| `world/items.py`, `world/results.py` — item-encoding pipeline + Result registry | done, tested (`decisions.md` #24, #25); real `NomicItemEncoder` untested live (`huggingface.co` blocked here) |
| Player-driven item creation — generic `Item`, `create_item` director action | **done, tested** (`decisions.md` #27) |
| `fly_brain/circuit.py` — trainable LIF circuit over real connectome | done, tested |
| `fly_brain/agent.py` — `EscapeAgent` wiring circuit into the world | done, integration-tested (untrained weights) |
| `training/curriculum.py` — stage configs | done (stage 1 only) |
| `training/trainer.py` / `training/run.py` — ES training loop | done, mechanically tested |
| Stage 1 training run (clean escape) | **unblocked and verified** — real `population_reward_std` every iteration (`decisions.md` #15); 15-iteration smoke test only, a longer real run is still future work |
| `director/` — swappable LLM world-controller layer | done; registry → rule-based controller → `Environment` verified end to end; `ClaudeController` built to spec but **not tested live** (no API credentials in this environment) — see `decisions.md` #16 |
| Reproduction mechanic — multi-fly `Environment`, stochastic trigger, parent cost | **done, tested** (`decisions.md` #20) |
| `game/colony.py` — per-fly circuits/genomes (escape + plasticity), offspring genome creation, headless colony runner | **done, tested** (`decisions.md` #21, #26) — real natural birth observed in a full run, colony driven by actual trained/learning circuits end to end |
| `fly_brain/plasticity.py` — KC/MBON/DAN circuit, dopamine-gated lifetime plasticity, audit hooks | **done, tested** (`decisions.md` #26) — real learning curves verified (reward, punishment, generalization), wired into `Colony`, prior-vs-live gain split verified on reproduction |
| No foraging behavior — flies only eat food they wander into by luck | known limitation (`decisions.md` #21); **partly a reward-design gap, not just a curriculum one** — starvation produces no learning signal at all (`decisions.md` #28) |
| `tests/` — contract tests (item contract, mutation-verified) | **done** (`decisions.md` #28) |
| Live game loop (`game/live_run.py`: `director/` commands affecting a running `Colony` continuously) | **done, tested** (`decisions.md` #23) |
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
5. ~~`game/colony.py`: wire real per-fly circuits/genomes~~ done —
   see `decisions.md` #21. Verified with a headless CLI run, not yet
   connected to `director/` or anything live/interactive.
6. ~~Live game loop~~ done — see `decisions.md` #23.
   `game/live_run.py` (`python -m game.live_run`) connects
   `director/`'s `WorldController` to a continuously-ticking `Colony`: a
   background thread queues player requests, the main loop drains and
   applies them each tick without pausing the world. This is what "the
   core Python game loop" in `decisions.md` #19's sequencing condition
   referred to — the frontend is now unblocked.

All six immediate next steps above are done. Now implementing the
sensing/learning redesign (`decisions.md` #22), in checkpointed phases:

7. ~~Phase 1: `world/items.py`, the item-encoding pipeline~~ done, see
   `decisions.md` #24. `ItemEncoder` swap point, `NomicItemEncoder`
   (real backend, untested live — `huggingface.co` blocked in this dev
   environment) and `HashingItemEncoder` (tested stub).
8. ~~Phase 2: anonymous `Percept`/`Observation`, `Fly` health/
   `stuck_ticks`, the Result registry~~ done, see `decisions.md` #25.
   The open design point from #24 (how the escape circuit gets its
   stimulus without `threat_signal`) is resolved: `sense_danger()`
   sums clipped-cosine-similarity to a `danger_vector` across every
   nearby percept. Retrained stage 1 from scratch against the new
   stimulus (baseline 65.3 → 80.0 fitness) and verified the full
   pipeline live in `colony_run.py` (real food pickups changing
   hunger/stuck_ticks were traced during an actual run).
9. ~~Phase 3: the KC/MBON/DAN circuit + dopamine-gated plasticity rule~~
   done, see `decisions.md` #26. Real, bounded 481-neuron subgraph (100
   KCs by real KC→MBON weight, 50 MBONs, 300 PAM + 31 PPL/PPM DANs);
   two real bugs found and fixed empirically along the way (a uniform
   DAN current saturates instead of grading with magnitude; the
   textbook depression-only rule doesn't move a direct-spike-readout
   MBON). Auditability hooks (`probe()`, `gain_drift()`,
   `plasticity_summary()`) built in from the start per an explicit ask;
   real learning curves verified for both reward and punishment.
10. ~~Phase 4: `game/colony.py` integration~~ done, see
    `decisions.md` #26. Both circuits run per fly; escape overrides
    plasticity only when `TTMn` actually spikes; offspring inherit the
    plasticity prior, never the parent's own lifetime-drifted gains
    (verified directly).

#22 is now fully implemented (phases 1–4).

11. ~~Player-driven item creation~~ done, see `decisions.md` #27. A
    generic, single-use, stationary `Item` (no special-cased behavior —
    everything comes from its attribute vector via the existing
    Percept/Result-registry machinery), a `create_item(description)`
    director action (`director/`'s v1 contract's first parameterized
    action, extended rather than replaced), `RuleBasedController`
    handling it via a crude trigger phrase and `ClaudeController` via a
    real `input_schema`. Verified end to end including through a live
    `Colony`: a freshly player-created item sensed, picked up, and
    correctly triggering `reinforce()`.

12. ~~Quality review before the frontend~~ done, see `decisions.md` #28.
    Collapsed `Food` into one `Item` concept (three near-duplicates →
    one), made `Channel` an enforced enum, split the game out of
    `training/` into `game/`, wrote the item contract test and verified
    it fails when the contract breaks, and documented the starvation
    learning gap honestly rather than leaving it framed as purely a
    curriculum issue.

Remaining, not yet started: an evolutionary process for the plasticity
circuit's own prior/hyperparameters (currently untrained, gain=1.0); a
dedicated audit CLI/visualization on top of the hooks already in place;
finer per-item behavior (movement, spawn-rate weighting) beyond the one
shared default `create_item` currently gives every item — deliberately
left simple, per #27; and the open question of whether hunger loss
should carry a punishment signal (#28).

The frontend is a separate, independent track, also unblocked and not
yet started (see "After that" below).

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
- **Structure/effect split for items, threats, and tiles** — separating
  *what moves/is consumed/occupies an area* (structure, discrete,
  tool-selected) from *what it does to a fly* (effect, continuous,
  description-driven), plus a new permanent "trait" effect tier
  alongside today's transient `Channel` one. Brainstormed, not
  designed, not started (`decisions.md` #29).
- **Two-player "god vs devil" mode** — human vs computer or human vs a
  friend, one growing the colony, one destroying it, same
  instructions→world→learning loop multiplexed across two sources.
  Noted for the future; multiplayer explicitly out of scope for now
  (`decisions.md` #29).
