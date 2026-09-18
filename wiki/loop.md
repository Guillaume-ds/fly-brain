# The loop that connects them

Part of the standing checkpoint doc set — see [state.md](state.md) for
the catchphrase and how these three files fit together.

**Pillar contract:** the world and the colony must never need to know
about each other's internals to interact — everything that crosses
between "the user's edits" and "what a fly experiences" goes through
`Environment`'s public surface, in real time, without either side
blocking the other.

---

### How does time actually pass?

**What it is:** continuously, in real time — a background thread takes
the user's edits while the world keeps ticking; nothing pauses to wait
for input.

**Where we stand:** built and tested (`training/live_run.py`,
`decisions.md` #23) — a request queue drained each tick, ticking paced
by `--ticks-per-second`, verified to keep advancing in real time between
requests rather than only on input.

**What's left:**
- done recently: the live loop itself — this was the last blocker noted
  for the frontend (#19)
- next: nothing planned for the loop mechanics themselves; the frontend
  would consume this same loop, not replace it

**Logic for testing:** contract = the colony keeps ticking at
approximately the configured rate regardless of whether requests are
arriving. Test by running the loop for a fixed wall-clock duration with
no input at all and asserting the tick count lands within a tolerance
of `duration * ticks_per_second` — already done once with staggered
input (#23); worth keeping as a standing check since a future change
that accidentally makes ticking wait on I/O would be easy to miss
otherwise.

---

### How does the user see what's happening?

**What it is:** text logs only, right now — population, births, deaths,
and periodic learning-audit lines. No visual frontend yet.

**Where we stand:** logging is the only observability surface that
exists; it's functional but not designed to be read by a player, only
by whoever's developing this.

**What's left:**
- done recently: periodic plasticity-audit log lines during a run, not
  just at the end (#26) — the last observability gap before a real
  frontend
- next: the frontend itself (FastAPI+WebSocket backend, Next.js/Phaser
  rendering) — decided (`decisions.md` #19), not started, and the
  explicitly agreed next big piece of work after this wiki pass

**Logic for testing:** no contract yet worth writing — there's nothing
here to hold invariant beyond "logging doesn't crash the loop," which
the loop-level test above already covers implicitly. A real contract
for this bullet should get written once the frontend exists (e.g. "the
UI's view of population/state never diverges from `Environment`'s own,"
which is really a WebSocket-sync contract, not a world/colony one).

---

### What does "winning" actually mean today?

**What it is:** colony extinction is detected (`colony_extinct`), but
there's no scoring, no win/lose screen, no explicit end-of-game moment
beyond that flag.

**Where we stand:** the flag itself is solid — it's exactly the
condition `run_colony()`/`run_live()` already stop on — but nothing
downstream of it means anything to a player yet.

**What's left:**
- next: decide what "the game ending" should actually look like for a
  player (a score based on how long survival took? how much the LLM had
  to invent? something else?) — open design question, not started, and
  probably worth deciding *before* the frontend renders anything final,
  since the UI would need to show it

**Logic for testing:** contract = `colony_extinct` is `True` if and only
if `Environment.flies` is empty, and it's `True` on the tick population
reaches zero, never later. Test by driving a colony to extinction under
a few different causes-of-death mixes and asserting the flag flips on
the exact tick the last fly dies — cheap, already implicitly exercised
by every extinction seen in this project's testing so far, worth making
explicit once a score/win-screen concept exists to hang off of it.

---

### Does anything carry over between sessions?

**What it is:** no — every run starts fresh; player-created item types
and the colony's whole learning history don't persist.

**Where we stand:** no persistence layer exists at all. Every
`live_run.py`/`colony_run.py` invocation calls `Environment.reset()`
(or constructs fresh) with nothing loaded from a prior session, except
the ES-trained escape checkpoint, which is a training artifact, not a
live-session save.

**What's left:**
- next: decide whether persistence matters before or after the
  frontend — a browser session naturally raises the question of "can I
  close the tab and come back," which the current CLI mostly sidesteps

**Logic for testing:** no contract yet — nothing to hold invariant until
a persistence mechanism exists. Once one does, the natural contract is
"a saved-then-reloaded colony produces bit-identical behavior to one
that was never interrupted, given the same subsequent inputs" — a
determinism check, not a serialization-format check.
