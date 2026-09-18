# The world the players shape

Part of the standing checkpoint doc set — see [state.md](state.md) for
the catchphrase and how these three files fit together. This file is
the first link in the chain: **instructions → world** → flies →
learning.

**Pillar contract:** every instruction, whichever player it came from
and whether that player is a human or an AI, must land on the world
through the action registry — nudging an existing rate, or creating a
new element from a description — and must never touch a fly's internal
state directly. An instruction that asks for something the engine
doesn't implement produces the closest thing the engine *does*
implement, or nothing; it never changes what the engine is capable of.
Anything that counts as an item, however it was created, must resolve
to a plain attribute vector before it can affect a fly; nothing about
*what it is* should ever reach a fly as a name or type. `Threat` is the
single documented exception, and it's pinned by a test so it can't
widen (see "What is an item?" below).

---

### How is the world built?

**What it is:** a grid holding flies, food, threats, and generic items,
advancing in fixed discrete ticks. Spawning, movement, and death rules
live entirely in `world/env.py`; nothing in that file knows a fly's
brain exists.

**Where we stand:** built and tested — multi-fly, continuous spawning,
health/damage/stuck-ticks state, a generic item registry
(`decisions.md` #14, #15, #20, #25, #27).

**What's left:**
- done recently: anonymous `Percept` sensing (#22/#25), the generic
  `Item` entity and player-created item types (#27)
- next: nothing specifically planned for the grid/tick mechanics
  themselves — the open item is terrain (see below)

**Logic for testing:** contract = `Environment.step()` returns a
structurally valid `ColonyStepResult` for *any* mix of registered
item/food/threat types, and never raises. Test by building an
`Environment` with a randomized mix of item types (including freshly
created ones), running N ticks, and asserting no exception plus valid
observation/death/birth shapes every tick — one generic invariant test,
not per-function coverage.

---

### How do instructions reach the world?

**What it is:** never directly — a player writes free text, an LLM
translates it into at most one call from a fixed action registry
(`director/`), and that call is the only thing the world ever sees.
The registry is the engine's whole vocabulary: if an instruction asks
for something no action covers, the untranslatable part is dropped, not
built. The engine's capabilities only ever change by someone editing
this project's source, never by a player asking well enough.

Two kinds of action exist today: nudging an existing spawn rate, and
creating a new *item* type from a description.

**Where we stand:** `RuleBasedController` + `Environment` wiring fully
tested end to end, including the parameterized `create_item` action.
`ClaudeController` is built to the identical contract but has **never
been exercised live** — no API credentials in this sandbox
(`decisions.md` #16, #27).

The gap against the chain in [state.md](state.md): of the three element
kinds an instruction is supposed to be able to create — **item, mob,
environment tile** — only *item* has a creation function.
`env.add_item_type(description)` exists and takes free text only.
There is no `create_mob`/`add_threat_type`: `Threat` spawns from one
built-in type fixed at `Environment.__init__`, so players can nudge how
often spiders appear but cannot invent a second kind of spider. There
is no tile concept at all. And there is exactly one instruction source
wired in, not two.

**What's left:**
- done recently: extending the action registry to support a real
  argument (`create_item`, #27) — the first crack in the old
  zero-argument-only contract
- next: **per-kind typed creation functions** — one function per element
  kind (`create_item`, `create_mob`, `create_tile`) instead of a single
  free-text one, each with a real parameter schema the translating LLM
  fills in, and each composing its own normalized description string to
  hand to the encoder. Two properties this is meant to buy: *which*
  function gets called is a discrete, engine-defined choice (the
  structure/effect split in `decisions.md` #29), and an unsupported
  parameter in a request simply has nowhere to go in the schema, so it
  is dropped silently rather than tempting anything to widen the
  engine. **Designed, not implemented** (`decisions.md` #30) — no code
  written. `create_item` and `create_mob` are deliberately asymmetric
  there: an item's embedding drives both perception and effect, a mob's
  drives perception only while `strength`/`type` drive its damage
  directly. Gated on one measurement first: a creatable mob's *name*
  has to reach the encoder (otherwise every mob sharing a
  type/strength shares one vector and flies can't tell a spider from a
  wasp), but a food-sounding name on a lethal mob then perceives as
  food — and the escape reflex is frozen for life, so unlike the
  plasticity circuit it can never learn around a convincing mimic. The
  proposed answer is to compose the embedded string from the name plus
  a mandatory mechanical clause phrased in the Result registry's own
  vocabulary, so the mimicry is imperfect. Whether that clause survives
  being swamped by the name is empirical: `python -m
  world.measure_encoder --encoder nomic` measures it, and needs a
  machine that can reach huggingface.co
- also next: a live test of `ClaudeController` with a real key
- later: a second instruction source (the two-player mode, #29) —
  nothing in this surface is single-player-shaped, but nothing
  multiplexes it either

**Logic for testing:** contract = any `WorldController`, given the same
registry, either returns a call against a real `ActionSpec` in that
registry, or returns nothing — never a name absent from it, never an
argument-taking action with missing arguments, and never an argument
the action's own schema doesn't declare. Test by running a fixed
battery of sample requests through *both* controllers — deliberately
including requests for things the engine can't do ("make the spiders
breathe fire") — and asserting every returned call validates against
the registry that was passed in, with the impossible parts absent
rather than smuggled through. This test is controller-agnostic on
purpose — it'll validate a Claude-backed run with zero changes the
moment credentials exist — and the impossible-request half is what pins
the "instructions never widen the engine" half of the pillar contract.

---

### What is an item?

**What it is:** a point entity with a text-derived attribute vector,
consumed on contact. Its effect on a fly (heal/damage/immobilize,
blendable) comes entirely from that vector via the Result registry —
never assigned per item, never hardcoded by name.

There is exactly **one** item entity in the code (`decisions.md` #28).
Food is not a separate class — it's a registered `ItemType` like any
player-created one, spawning through the same path into the same list.
If you can't tell food and a player-invented berry apart by reading the
code, that's the point.

**Where we stand:** built and tested. The contract below is enforced by
a real test (`tests/test_item_contract.py`), parametrised over every
item-producing pathway, and verified to actually fail when the contract
is broken (`decisions.md` #24, #25, #27, #28).

**What's left:**
- done recently: the Result registry (#25), the `create_item` action
  (#27), and collapsing `Food` into `Item` so there's one concept
  rather than three near-duplicates (#28)
- next: swap in the real semantic encoder (`NomicItemEncoder`) once a
  reachable environment exists — everything today runs on the crude
  orthographic hashing stub, which is functionally correct but not
  semantically meaningful the way the real encoder would be

**Logic for testing:** **Contract:** anything that counts as an item must
(a) carry a unit-norm attribute vector, (b) be perceivable by a fly
*only* as an anonymous `Percept` — no name, type, or id ever reaches
`Observation` — and (c) produce its effect on a fly purely through
Result-registry similarity, never a hardcoded per-type constant.

**`Threat` satisfies (a) and (b) but deliberately not (c)** — it is
perceived exactly like an item, but it moves, is never consumed, and
kills through `determine_fly_death()` instead. That carve-out exists
because routing lethality through encoder similarity would make a
spider's deadliness depend on the encoder's judgment, silently changing
what the ES-trained escape circuit was trained against. The test pins
the carve-out precisely, so a threat can't quietly start behaving like
an item (or vice versa) without a test failing.

`tests/test_item_contract.py` implements exactly this: for every
pathway, spawn one, put a fly on it, and assert (a)/(b)/(c) on the
resulting `Percept` and the resulting state delta — checking the delta
against what `blend_deltas()` says it should be, which is what actually
rules out a hardcoded constant hiding somewhere. It doesn't care *how*
an item got made or what changed internally; only that "item-ness"
survives. Run it whenever item-related code changes, instead of
re-testing every function that touches an item.

---

### What's still missing from "open"?

**What it is:** two of the three element kinds an instruction is
supposed to be able to create don't exist as creatable things.

*Mobs:* there is one built-in `Threat` type, created in
`Environment.__init__` and tunable only by rate. A player can ask for
more spiders; a player cannot ask for a *different* spider. Making them
creatable runs straight into the carve-out below.

*Environment:* nothing area-based exists at all — no lava tile, no
slowing zone, no persistent hazard. Everything today is a discrete
thing you touch, not a place you're in.

**Where we stand:** acknowledged gaps, not designed.

**What's left:**
- next: per-kind typed creation functions (see "How do instructions
  reach the world?" above) — the shape that would give mobs and tiles
  the same player-creatable status items already have
- also next: decide whether a terrain effect is a variant of `Item`
  (e.g. one with a radius that lingers instead of being consumed) or a
  genuinely new subsystem — open design question, not started
- the blocker on creatable mobs is unchanged and is *not* structural:
  a spider's lethality goes through `determine_fly_death()`, not the
  Result registry, precisely so it doesn't depend on the encoder's
  judgment. Under today's stub encoder a spider reads as more food-like
  than damage-like (#28), so a described-into-existence mob can't be
  allowed to derive its own danger until the real semantic encoder is
  in place. One gate, not two (`decisions.md` #29)

**Logic for testing:** none yet — no contract exists until the design
does. Once one exists, the first thing to check against it should be
the item contract above: does a terrain tile still satisfy "anonymous
to the fly, effect derived from its vector, nothing hardcoded"? If not,
that's a sign the item contract itself needs to widen, not that terrain
needs a special case.
