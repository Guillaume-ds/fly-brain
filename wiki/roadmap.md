# Roadmap

## Current vision (see `decisions.md` #13, `wiki/state.md`)

Two players (human or AI) send instructions; an LLM translates each into
one call from a fixed action registry; those calls shape the world
(items, mobs, environment); a fly colony survives, reproduces and learns
against whatever the world becomes. Not a building game — see #13 for
why that was dropped. Today only one instruction source and one
creatable element kind (items) actually exist — see `wiki/world.md`.

## Status

| Piece | Status |
|---|---|
| Connectome data access (`fly_brain/data.py`) | done |
| Connectome exploration (`analyze`, `simulate` commands) | done |
| `world/env.py` — multi-fly, continuous spawn-rate params, mob movement, anonymous `Percept` observation | done, tested (`decisions.md` #14, #15, #20, #25) |
| `world/items.py`, `world/results.py` — item-encoding pipeline + Result registry | done, tested (`decisions.md` #24, #25); real `NomicItemEncoder` untested live (`huggingface.co` blocked here) |
| Player-driven item creation — generic `Item`, `create_item` director action | **done, tested** (`decisions.md` #27, #32) |
| Player-driven *tile* creation — `Tile`, `create_tile` director action | **done, tested** (`decisions.md` #31, #32) — reuses the item mechanism, never consumed, a fraction re-applied every tick |
| Player-driven *mob* creation — `Threat` renamed `Mob`, `create_mob` director action | **done, tested** (`decisions.md` #30–#32) — authored `effect`/`strength`, never derived from the embedding; the built-in spider is unaffected (still unconditional insta-kill) |
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

13. ~~Player-driven mob/tile creation~~ done, see `decisions.md` #30–#32.
    `create_tile` (a new `Tile` entity — an `Item` in every way except
    never consumed, its effect a fraction re-applied every tick) and
    `create_mob` (`Threat` renamed `Mob`; an authored `effect`/`strength`
    pair drives its contact effect directly, never the embedding, which
    now drives perception only) both land alongside `create_item`, all
    three sharing one `strength` magnitude field. `director/`'s
    `ActionSpec` gained a real per-action `argument_schema` (multiple
    typed fields, one of them a real enum) in place of the old single
    free-text argument — the schema itself is what makes an
    unsupported request ("spits fire") have nowhere to land. The
    contract test now covers item/tile/mob together: item and tile
    satisfy the full (a)/(b)/(c) item contract identically (tile scaled
    per tick), mob is pinned as the documented carve-out on (c), and
    every `Effect` member is checked to move only its own channel in
    the right direction. Verified live: graded mob damage (a
    strength-1 mob survivable over several ticks, unlike the built-in
    spider's unconditional insta-kill, which is unaffected), a
    positive-strength `free` mob rescuing an already-stuck fly, and the
    mandatory embedding clause firing only for `damage`/`starve`.

Remaining, not yet started: an evolutionary process for the plasticity
circuit's own prior/hyperparameters (currently untrained, gain=1.0); a
dedicated audit CLI/visualization on top of the hooks already in place;
the open question of whether hunger loss should carry a punishment
signal (#28); a live test of `create_mob`'s mandatory clause against
the real encoder (`world/measure_encoder.py --encoder nomic`, needs a
machine that can reach huggingface.co); and the follow-on ideas raised
alongside this work but deliberately not folded in — multi-colony
ownership per player, a resource-cost system for creation actions, and
a preview/confirm UX pattern (generate a template from a request, let
the player edit numbers or text before committing) that's really a
frontend-and-resource-cost-sequenced feature, not a backend one.

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
- **Multi-colony ownership** — N players, each with their own colony in
  one shared world, helping their own and degrading everyone else's; a
  real step up from "god vs devil"'s one shared colony. **Designed**
  (`decisions.md` #34, perception revised by #35 — `Fly.owner`,
  per-owner extinction, a `target` field on creation, player identity
  threaded through `director/`, and fly-vs-fly combat: every fly
  perceives every other fly, but through a **fixed, exactly orthogonal
  vector per owner** (a standard basis vector, not jittered) rather
  than one shared generic one — the pillar anonymity contract stays
  exactly as strict as it is today (still just `attributes, dx, dy,
  distance`, no owner field), while a colony can now actually learn
  "this rival is weak, that one is strong" per opponent. Contact
  between different-owner flies deals symmetric authored `HEALTH`
  damage; a fly that dies in combat transfers a fraction of its own
  hunger to the rival(s) that killed it, same tick — grounding "killing
  is good" in a real, existing reward channel (`reward = max(0,
  ΔHUNGER)`) instead of an injected bonus, and self-limiting against
  gratuitous kills since a starving rival has nothing to transfer),
  not implemented. Still open: home-region geometry, whether
  `max_population` stays shared or goes per-player, the combat damage
  constant, and the kill-transfer fraction (real risk to tune around:
  could make combat more lucrative than foraging, on top of #28's
  already-known foraging weakness).
- **Resource-cost system** — bound creation (item/mob/tile all share one
  `strength` field now, `decisions.md` #32, which is the hook a uniform
  cost formula needs) behind a **per-player** energy pool instead of a
  flat type-count cap (which stays as a much-higher backstop, not
  removed), so spamming creation is a trade-off, not a wall. **Per
  player, deliberately not per colony** — energy is the capacity to
  issue instructions, an attribute of whoever's sending them, not of
  the fly population they land on; per-colony has nowhere to put two
  pools in "god vs devil" mode's one shared colony. **Designed**
  (`decisions.md` #33 — cost = `(kind, strength)` only, fixed per-tick
  regen not tied to colony state, gates creation only, the three
  `add_*_type` methods gain a `bool` return for denial logging), not
  implemented; today's single instruction source means one global pool
  for now, structured to key by player once a second source exists.
- **Preview/confirm creation UX** — before committing, show the player
  what a request actually produced (the Result-registry blend weights,
  `strength`, an eventual resource cost) and let them either refine the
  text or edit the numbers directly, the latter decoupling the
  description from what it derives. Needs the resource-cost system (for
  the cost figure) and the frontend (for the modal/sliders) to exist
  first — a UX pattern layered on top of `create_item`/`create_tile`/
  `create_mob`, not a change to their mechanics. Raised, not designed.
