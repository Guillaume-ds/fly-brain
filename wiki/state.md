# Project state

The standing checkpoint doc: what this game *is*, right now, in plain
terms. Re-read and update this whenever the core loop changes —
`roadmap.md` tracks piece-by-piece status, `decisions.md` explains why
things are the way they are; this page is for "what is the game,
concretely, today."

## The idea

**A game where the user shapes an open-ended world, through an LLM, to
try to wipe out a fly colony that adapts to everything it encounters.**

The colony adapts at two speeds — a fly learns within its own life, and
the population evolves across generations — which is what keeps it from
being a static target.

## 1. The world the user shapes

- **How is it built?** A grid holding flies, food, threats, and generic
  items, ticking forward in discrete steps — spawn/movement/death rules
  live entirely in `world/`, with no knowledge of any fly's brain.
- **How does the user shape it, and how directly?** Never directly —
  only through an LLM translating free text into one of a fixed set of
  actions (`director/`). Two kinds exist: nudging existing spawn
  pressure, and inventing a brand-new item type from a description
  (`decisions.md` #27).
- **What is an item?** A point entity with a text-derived attribute
  vector, consumed on contact, whose effect on a fly (heal/damage/
  immobilize, blendable) falls entirely out of that vector — never
  hand-assigned per item (`decisions.md` #22 part 5, #25).
- **What's still missing from "open"?** Anything area-based (a lava
  tile, a slowing zone) — everything today is a discrete thing you
  touch, not a place you're in. And `Threat` (spiders) sits outside this
  system entirely: it moves and kills on contact through its own
  hardcoded path, not through items at all.

## 2. The colony that lives in it

- **How does one fly survive, tick to tick?** Hunger, health, and
  stuck-ticks, each decaying or shifting from real contact with things
  in the world; death comes from starvation, a threat, or damage
  (`decisions.md` #25).
- **How does it perceive?** Anonymously — an attribute vector and a
  relative position per nearby thing, nothing that says what it is
  (`decisions.md` #22 part 1).
- **How does it act on what it perceives?** Two circuits, not one: an
  innate reflex fixed for life that always wins when it fires, and a
  learned valence circuit underneath it that decides approach-or-avoid
  the rest of the time (`decisions.md` #5, #26).
- **How does it learn within its own life?** Real consequences (hunger
  up, health down) drive real reward/punishment signaling that locally
  rewires the valence circuit — never told what's good or bad, only what
  happened (`decisions.md` #22 part 2, #26).
- **How does it evolve across generations?** A separate, slower
  mechanism: reproduction is stochastic and costs the parent, and
  offspring inherit a mutated *starting point*, never what a parent
  personally learned. Selection only acts here (`decisions.md` #20,
  #21, #22 part 3).
- **How can you check any of this is real?** A read-only probe, a
  learning-so-far scalar, and a colony-wide summary — built so learning
  is checkable, not just asserted (`decisions.md` #26).

## 3. The loop that connects them

- **How does time actually pass?** Continuously, in real time — a
  background thread takes the user's edits while the world keeps
  ticking; nothing pauses to wait for input (`decisions.md` #23).
- **How does the user see what's happening?** Text logs only, right
  now — population, births, deaths, and periodic learning-audit lines.
  No visual frontend yet (`decisions.md` #19).
- **What does "winning" actually mean today?** Colony extinction is
  detected, but there's no scoring, no win/lose screen, no explicit
  end-of-game moment beyond that flag.
- **Does anything carry over between sessions?** No — every run starts
  fresh; player-created item types and the colony's whole learning
  history don't persist.

## Known simplifications vs. the real target

Worth re-checking at every checkpoint, not just noting once:

- The item encoder is a crude orthographic stub
  (`HashingItemEncoder`) — the real semantic encoder
  (`NomicItemEncoder`) is built but untested live, `huggingface.co` is
  policy-blocked in this dev environment (`decisions.md` #24).
- The plasticity circuit is a bounded ~500-neuron real subgraph, not
  the full ~4,500-neuron real population — computationally impractical
  per-tick, per-fly at full scale (`decisions.md` #26).
- The MBON approach/avoid split is a structural simplification (sorted
  body-id parity), not a biological claim — there's no real
  downstream-of-MBON connectivity in the bounded subgraph to derive it
  from (`decisions.md` #26).
