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
*what it is* should ever reach a fly as a name or type. `Mob` is the
single documented exception, and it's pinned by a test so it can't
widen (see "What is an item?" below).

---

### How is the world built?

**What it is:** a grid holding flies, mobs, items, and tiles, advancing
in fixed discrete ticks. Spawning, movement, and death rules live
entirely in `world/env.py`; nothing in that file knows a fly's brain
exists.

**Where we stand:** built and tested — multi-fly, continuous spawning,
health/damage/stuck-ticks state, and generic item/tile/mob registries
(`decisions.md` #14, #15, #20, #25, #27, #30–#32).

**What's left:**
- done recently: anonymous `Percept` sensing (#22/#25); the generic
  `Item` entity and player-created item types (#27); `Tile` (an `Item`
  that's never consumed) and `Mob` (renamed from `Threat`, an
  authored `effect`/`strength` pair driving its contact effect) and
  their own player-created types (#30–#32)
- next: nothing specifically planned for the grid/tick mechanics
  themselves — the remaining open item is true area terrain, since
  today's `Tile` is a point-and-radius zone, not irregular shaped
  ground (see "What's still missing from open" below)

**Logic for testing:** contract = `Environment.step()` returns a
structurally valid `ColonyStepResult` for *any* mix of registered
item/tile/mob types, and never raises. Test by building an
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
creating a new element — an *item*, a *tile*, or a *mob* — from a
request.

**Where we stand:** `RuleBasedController` + `Environment` wiring fully
tested end to end, including all three parameterized create actions.
`ClaudeController` is built to the identical contract but has **never
been exercised live** — no API credentials in this sandbox
(`decisions.md` #16, #27).

All three element kinds an instruction is supposed to be able to create
— **item, mob, environment tile** — now have a creation function
(`decisions.md` #30–#32). Each `ActionSpec` in the registry carries a
real per-action `argument_schema` (multiple typed fields, `create_mob`'s
`effect` a real enum) instead of the old single free-text argument —
that schema is the actual mechanism behind "instructions never widen
the engine": a request for something outside it ("spits fire") has
nowhere to bind and is silently dropped, never routed around.

`create_item`/`create_tile` share one shape: `name`, a free-text
`description` whose embedding drives both perception and the *shape* of
the effect (via the Result registry), and `strength` scaling that
effect's magnitude only. `create_mob` is deliberately different:
`name`+embedding drive perception only, while an authored `effect`
(`heal`/`damage`/`feed`/`starve`/`trap`/`free`) and `strength` drive the
actual effect directly — the encoder is never trusted with anything
that could invalidate the frozen escape circuit's training. A mob whose
`effect` is lethal (`damage`/`starve`) gets a mandatory clause forced
into its embedded string, worded from the Result registry's own
vocabulary, so a mimic with a harmless-sounding name still registers
some danger to the frozen reflex — whether that clause survives being
swamped by the name on a *real* encoder is still unverified
(`python -m fly_brain.measure_encoder --encoder nomic` needs a machine
that can reach huggingface.co; everything above runs on the
orthographic stub today).

Creation also costs something now: `Environment.energy`, one pool per
owner, that `creation_cost(kind, strength)` draws down and a fixed
per-tick regen refills — never tied to colony state (`decisions.md`
#33, keyed by owner since #34). All three `add_*_type` methods return
whether they actually created something, so a type-cap rejection, an
unaffordable request, and success are three distinguishable outcomes
rather than one silent no-op; `game/live_run.py` logs an
understood-but-rejected request as `f"{name} (rejected)"`. The
translating LLM never sees the balance — the environment enforces it
the same way it already enforces `effect`/`strength` validity, silently.

Instructions also carry a real owner now (`decisions.md` #34): `owner`
is supplied by the caller (`game/live_run.py`'s request queue), never
by the translating controller, since it's session identity, not
request content — `WorldController.choose_action()` itself is
unchanged, owner-agnostic on purpose. `create_item`/`create_tile`/
`create_mob` gained an optional `target` field (`"own"` or a named
rival) so a spawned instance can land near a specific territory
instead of anywhere on the grid; omitted, spawning is exactly what it
always was. `target` needed a new `required_arguments` mechanism on
`ActionSpec` since it's the first optional field any schema here has
had — every other field stays mandatory.

**What's left:**
- done recently: `create_tile` and `create_mob` (`decisions.md`
  #30–#32) — `Threat` renamed `Mob` to match, since the class stopped
  meaning "always dangerous" the moment it could be beneficial; the
  action registry's argument mechanism generalized from one free-text
  field to a real per-action schema; the resource-cost system
  (`decisions.md` #33) gating all three; `owner`/`target` threading and
  per-owner energy (`decisions.md` #34, #35)
- next: a live test of `ClaudeController` with a real key, and the
  `measure_encoder` run above against the real encoder; the energy
  constants (`STARTING_ENERGY`/`MAX_ENERGY`/regen rate/per-kind cost)
  are placeholders, not tuned by actually playing yet
- later: a second *concurrent* instruction source — the plumbing for
  multiple owners exists and is tested (`Environment.spawn_colony()`/
  `Colony.add_colony()`, `decisions.md` #34), but `game/live_run.py`'s
  CLI still only reads from one local stdin stream, tagged to one
  default owner; an actual second input source (a second terminal, a
  socket, the eventual frontend) is still unbuilt. A preview/confirm
  step showing a request's derived numbers and cost before committing
  (raised, not designed, needs the frontend)

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
blendable) comes entirely from that vector via the Result registry,
scaled by `strength`'s overall magnitude only (`decisions.md` #32) —
never assigned per item, never hardcoded by name.

There is exactly **one** item entity in the code (`decisions.md` #28).
Food is not a separate class — it's a registered `ItemType` like any
player-created one, spawning through the same path into the same list.
If you can't tell food and a player-invented berry apart by reading the
code, that's the point. `Tile` (`decisions.md` #31) is exactly this
same mechanism, minus one thing: it's never consumed, and re-applies a
fraction of its effect every tick a fly stays within its radius instead
of once on contact — a place, not a thing you touch once.

**Where we stand:** built and tested. The contract below is enforced by
a real test (`tests/test_item_contract.py`), parametrised over every
item- and tile-producing pathway, and verified to actually fail when
the contract is broken (`decisions.md` #24, #25, #27, #28, #32).

**What's left:**
- done recently: the Result registry (#25), the `create_item` action
  (#27), collapsing `Food` into `Item` so there's one concept rather
  than three near-duplicates (#28), and `create_tile` plus a `strength`
  magnitude field shared by both (#31, #32)
- next: swap in the real semantic encoder (`NomicItemEncoder`) once a
  reachable environment exists — everything today runs on the crude
  orthographic hashing stub, which is functionally correct but not
  semantically meaningful the way the real encoder would be

**Logic for testing:** **Contract:** anything that counts as an item must
(a) carry a unit-norm attribute vector, (b) be perceivable by a fly
*only* as an anonymous `Percept` — no name, type, or id ever reaches
`Observation` — and (c) produce its effect on a fly purely through
Result-registry similarity, never a hardcoded per-type constant.

**`Mob` satisfies (a) and (b) but deliberately not (c)** — it is
perceived exactly like an item, but it moves, is never consumed, and
its effect (if any) is authored from `effect`/`strength` directly,
never derived from the Result registry (`decisions.md` #30). That
carve-out exists because routing lethality through encoder similarity
would make a mob's deadliness depend on the encoder's judgment, silently
changing what the ES-trained escape circuit was trained against — true
for the built-in spider (still an unconditional insta-kill on contact,
unaffected by any of this) and equally true for anything a player
creates. The test pins the carve-out precisely, so a mob can't quietly
start behaving like an item (or vice versa) without a test failing.

`tests/test_item_contract.py` implements exactly this: for every
pathway, spawn one, put a fly on it, and assert (a)/(b)/(c) on the
resulting `Percept` and the resulting state delta — checking the delta
against what `blend_deltas()` says it should be (scaled by `strength`),
which is what actually rules out a hardcoded constant hiding somewhere.
For `Mob`, it additionally verifies every `Effect` member moves only its
own channel, in the direction a positive `strength` implies. It doesn't
care *how* an item/tile/mob got made or what changed internally; only
that "item-ness" (or its documented absence) survives. Run it whenever
item/tile/mob-related code changes, instead of re-testing every
function that touches one.

---

### What's still missing from "open"?

**What it is:** area effects exist now (`Tile`, `decisions.md` #31,
#32), but only as a point-and-radius zone — a circle of fixed size
centered on one cell, not a general shape. There's still no way to
carve an L-shaped lava lake or a winding slow-zone; every tile today is
round.

**Where we stand:** the *mechanism* (perception, Result-registry effect,
persistence) is built and tested; the *geometry* is deliberately the
simplest thing that could reuse it, not a considered design for
irregular terrain.

**What's left:**
- done recently: `Tile` itself, and the mob-creation blocker this
  section used to describe (`decisions.md` #29's "one gate, not two")
  no longer applies to a *created* mob at all — `effect`/`strength` are
  authored, read directly, never through the Result registry, so a
  created mob's effect doesn't depend on the encoder's judgment. What
  the encoder still decides is perception quality: can a fly tell one
  described mob from another, and does a misleadingly gentle name blunt
  the frozen escape reflex's response to something actually lethal
  (`create_mob`'s mandatory clause, `decisions.md` #30, #31). That's
  unverified against the real encoder — `fly_brain/measure_encoder.py`
  needs a machine that can reach huggingface.co
- next: decide whether true irregular terrain is worth building at all
  before the frontend exists to make placing it comprehensible — open
  design question, not started

**Logic for testing:** none yet for irregular terrain specifically — no
contract exists until the design does. `Tile`'s own contract is already
covered above, by the same item contract every other pathway satisfies.
Whatever irregular terrain becomes, the first thing to check against it
should be that same contract: does it still satisfy "anonymous to the
fly, effect derived from its vector, nothing hardcoded"? If not, that's
a sign the item contract itself needs to widen, not that terrain needs
a special case.
