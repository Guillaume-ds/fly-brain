# The world the user shapes

Part of the standing checkpoint doc set — see [state.md](state.md) for
the catchphrase and how these three files fit together.

**Pillar contract:** anything a user does to the world must go through
one of two channels — nudging an existing rate, or describing a new
item — and never touch a fly's internal state directly. Anything that
counts as an item, however it was created, must resolve to a plain
attribute vector before it can affect a fly; nothing about *what it is*
should ever reach a fly as a name or type.

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

### How does the user shape it, and how directly?

**What it is:** never directly — only through an LLM translating free
text into one of a fixed action registry (`director/`). Two kinds of
action exist: nudging an existing spawn rate, and creating a new item
type from a description.

**Where we stand:** `RuleBasedController` + `Environment` wiring fully
tested end to end, including the parameterized `create_item` action.
`ClaudeController` is built to the identical contract but has **never
been exercised live** — no API credentials in this sandbox
(`decisions.md` #16, #27).

**What's left:**
- done recently: extending the action registry to support a real
  argument (`create_item`, #27) — the first crack in the old
  zero-argument-only contract
- next: a live test of `ClaudeController` with a real key; nothing else
  planned for this control surface at v1 scope (still just 5 actions
  total)

**Logic for testing:** contract = any `WorldController`, given the same
registry, either returns `(name, argument)` for a real `ActionSpec` in
that registry, or returns `None` — never a name absent from it, never a
`takes_argument` action with a missing argument. Test by running a
fixed battery of sample requests through *both* controllers and
asserting every returned name/argument pair is valid against the
registry passed in. This test is controller-agnostic on purpose — it'll
validate a Claude-backed run with zero changes the moment credentials
exist.

---

### What is an item?

**What it is:** a point entity with a text-derived attribute vector,
consumed on contact. Its effect on a fly (heal/damage/immobilize,
blendable) comes entirely from that vector via the Result registry —
never assigned per item, never hardcoded by name.

**Where we stand:** `Food`, `Threat` (partially — see below), and
generic player-created `Item` all produce real, verified Result-registry
effects; the encode → jitter → spawn pipeline and the `MAX_ITEM_TYPES`
cap are both tested (`decisions.md` #24, #25, #27).

**What's left:**
- done recently: the Result registry itself (#25), the generic `Item`
  entity and `create_item` action (#27)
- next: swap in the real semantic encoder (`NomicItemEncoder`) once a
  reachable environment exists — everything today runs on the crude
  orthographic hashing stub, which is functionally correct but not
  semantically meaningful the way the real encoder would be

**Logic for testing:** this is the one you named explicitly, so it's
worth being precise about. **Contract:** anything that counts as an item
must (a) carry a unit-norm attribute vector, (b) be perceivable by a fly
*only* as an anonymous `Percept` — no name, type, or id ever reaches
`Observation`, and (c) produce its effect on a fly purely through
Result-registry similarity, never a hardcoded per-type constant.
**Test:** for every current item-producing pathway (`Food`, `Threat`,
a freshly created `Item`), spawn one, place a fly adjacent to it, and
assert all three properties hold on the resulting `Percept` and the
resulting state delta. This test doesn't care *how* an item got made or
what changes internally about it — only that "item-ness" survives.
It's the one test to re-run any time item-related code changes, rather
than re-testing every function that touches an item.

---

### What's still missing from "open"?

**What it is:** anything area-based — a lava tile, a slowing zone, a
persistent hazard. Everything today is a discrete thing you touch, not
a place you're in. And `Threat` (spiders) sits outside the item system
entirely: it moves and kills on contact through its own hardcoded path,
never through the Result registry.

**Where we stand:** acknowledged gap, not designed.

**What's left:**
- next: decide whether a terrain effect is a variant of `Item` (e.g. one
  with a radius that lingers instead of being consumed) or a genuinely
  new subsystem — open design question, not started

**Logic for testing:** none yet — no contract exists until the design
does. Once one exists, the first thing to check against it should be
the item contract above: does a terrain tile still satisfy "anonymous
to the fly, effect derived from its vector, nothing hardcoded"? If not,
that's a sign the item contract itself needs to widen, not that terrain
needs a special case.
