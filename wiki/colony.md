# The colony that lives in it

Part of the standing checkpoint doc set — see [state.md](state.md) for
the catchphrase and how these three files fit together. This file is
the second link in the chain: instructions → world → **flies →
learning**.

**Pillar contract:** a fly's behavior must always be explainable as one
of three things — a fixed reflex, a learned response to real lived
experience, or inherited/evolved structure — never a hardcoded rule
tied to a specific item's identity. If a behavior can't be traced to one
of those three, something has broken the design.

---

### How does one fly survive, tick to tick?

**What it is:** hunger, health, and stuck-ticks, each decaying or
shifting from real contact with things in the world. Death comes from
starvation, a threat, or damage.

**Where we stand:** built and tested, including the newer `health`/
`stuck_ticks` channels and the `"damage"` death cause (`decisions.md`
#25).

**What's left:**
- done recently: replacing "eating food always fully restores hunger"
  with a real Result-registry blend across all three channels (#25)
- next: no foraging behavior — a fly only finds food/items by luck while
  fleeing threats, never by seeking them out. Known, long-standing
  limitation, not a bug (`decisions.md` #21); caps how long any colony
  can realistically survive regardless of how well it's learned
- **and a second, independent cause of the same thing, found during the
  #28 review: starvation produces no learning signal at all.** Reward is
  `max(0, Δhunger)`, so hunger *loss* is never punished — only gaining
  hunger rewards. A fly can starve to death having learned nothing,
  and starvation is the most common death in every run so far. That
  means the learning rule can only reinforce behavior *after* a lucky
  success; it cannot bootstrap search. Fixing foraging is therefore not
  purely "stage 3's job" — it's partly a reward-design question that
  hasn't been decided yet

**Logic for testing:** contract = a fly's three state channels only ever
move through `apply_result()`'s Result-registry blend or the fixed
per-tick hunger decay — never a direct, ad hoc assignment anywhere
else in the codebase. Test by asserting no other write to
`fly.hunger`/`fly.health`/`fly.stuck_ticks` exists outside those two
paths (a grep-able invariant, cheap to keep re-checking as the codebase
grows) plus one integration check: any registered item, regardless of
its description, produces a state delta that's fully explained by
`blend_deltas()` on its own attribute vector.

---

### How does it perceive?

**What it is:** anonymously — an attribute vector and a relative
position per nearby thing, nothing that says what it is.

**Where we stand:** built and tested (`decisions.md` #22 part 1, #25,
#34, #35, #39). Multi-colony ownership extended *who* gets perceived —
every fly now perceives every other fly, own or rival, through
`observe()` — without touching *what* a Percept can contain: each
owner has a fixed, exactly orthogonal vector, distinguishable across
owners so a colony can learn per-rival valence, but the Percept itself
is still exactly `attributes, dx, dy, distance` — no owner field, ever.
As of #39, *how far* a fly can sense an item/tile/mob is no longer the
same number as how close it has to get to actually interact with one —
`WORLD_SENSING_RADIUS` governs `observe()`, each entity's own (much
smaller) `radius` still gates the actual pickup/tile-effect/mob-contact,
independently. A fly can now smell food long before it's close enough
to eat it.

**What's left:**
- done recently: replacing the old fixed `food_signal`/`threat_signal`
  fields with an open-ended `Percept` list; extending perception to
  other flies for multi-colony ownership (`decisions.md` #34, #35);
  splitting sensing distance from interaction distance so a fly can
  perceive something well before it's reachable (#39)
- next: nothing planned for `Percept`'s own shape — this is considered
  settled and foundational. `WORLD_SENSING_RADIUS = 8.0` is a
  placeholder, not tuned by playing yet

**Logic for testing:** contract = `Observation.nearby` never contains
anything but `Percept(attributes, dx, dy, distance)` — no field, no
attached metadata, that could reveal an entity's world-internal type.
Test by introspecting every `Percept` a colony run produces (across all
current item-producing pathways) and asserting its fields are exactly
that tuple — a schema check, not a per-scenario one, so it stays valid
no matter how many new item types get created.

---

### How does it act on what it perceives?

**What it is:** two circuits, not one — an innate reflex fixed for life
that always wins when it fires, and a learned valence circuit
underneath it that decides approach-or-avoid the rest of the time.

**Where we stand:** both built, both tested, combined via the
freeze+override rule (`decisions.md` #5, #22, #26).

**What's left:**
- done recently: `EscapeAgent.decide()` returning `None` instead of
  forcing `STAY`, so `Colony` can fall through to the plasticity
  circuit's own decision (#26)
- next: nothing specifically planned — stage 2/3 curriculum (noisier
  escape training, a trained foraging pathway) would extend the reflex
  side, not this split itself

**Logic for testing:** contract = whenever the escape circuit's `TTMn`
neuron spikes, the fly's action is always the escape decision, never the
plasticity circuit's; whenever it doesn't, the fly's action is always
the plasticity circuit's. Test by feeding a `Colony` a scenario with a
clearly dangerous percept (forces a spike) and a clearly neutral one
(no spike) and asserting the resulting action matches the expected
source in each case — checks the override *rule*, not either circuit's
internal correctness.

---

### How does it learn within its own life?

**What it is:** real consequences (hunger up, health down) drive real
reward/punishment signaling that locally rewires the valence circuit —
never told what's good or bad, only what happened. That signal is now
specifically the *effect* of what touched the fly — an item, tile, mob,
or another fly — never a whole-tick net state diff, which used to
silently net a real effect against that same tick's constant hunger
decay (`decisions.md` #37).

**Where we stand:** built and tested, including real learning curves
for both reward and punishment, and generalization to similar-but-
different items (`decisions.md` #22 part 2, #26). Effect-attribution
fixed a real signal-quality bug (#37): a weak but genuine food pickup
could previously net to a negative number against decay and produce
*zero* reward — `ColonyStepResult.effects` isolates it now. Hunger
*loss* itself is punished as of #38, symmetric with health loss — in
practice this fires exactly and only from a `starve`-effect mob, since
decay never reaches `effects` and items/tiles can't produce a negative
hunger delta under the current Result vocabulary. As of #39, credit for
a multi-tick approach — sensing food from a distance and walking toward
it over several ticks before contact — was confirmed to already work
through the existing KC eligibility trace (`kc_trace`, #22/#26): it's
nonzero, reflecting real prior sensing, *before* the tick contact
happens on, not reset to a blank slate each tick. Found honestly, not
assumed either way: a longer sensed approach doesn't automatically
out-learn an instant close-range contact, since per-tick stimulus
strength falls off with distance and the circuit's spiking is threshold-
gated, not a smooth ramp — what's guaranteed is that the approach is
credited at all, not that it's credited *more*.

**What's left:**
- done recently: two real bugs found and fixed while verifying this —
  a uniform DAN current that saturated instead of grading with
  magnitude, and a depression-only Hebbian rule that never moved a
  direct-spike-readout MBON (#26); effect-attributed reward, replacing
  a whole-tick state diff that diluted weak effects with decay (#37);
  hunger-loss punishment, closing #28's last open clause — a fly that
  gets killed by a `starve` mob now actually learns to avoid it (#38);
  confirmed the eligibility trace already credits a sensed multi-tick
  approach, not just the final contact tick (#39)
- next: decay itself remains deliberately unpunished (#38) — a colony
  that quietly starves without ever touching a `starve` mob still gets
  no punishment signal from that decline. #39 fixed the *sensing* half
  of exploration-bootstrapping (a fly can now smell food from a real
  distance) but not the *movement* half — a fly with nothing sensed at
  all still does nothing but `STAY` (`PlasticityAgent.
  _valence_to_action()`), so it still can't get within sensing range of
  food on its own. An evolved wander behavior — agreed in design
  discussion, not yet implemented — is the next piece. A real semantic
  encoder (see world.md) would make *what* gets learned more meaningful
  without changing *how* learning works

**Logic for testing:** contract = repeatedly reinforcing the same
percept with a consistent-sign outcome must move `probe()`'s valence
for that percept in the matching direction, monotonically enough to be
detectable, and must leave `gain_drift()` > 0. Test by running the
existing reward/punishment exposure sequences (already used to verify
this, #26) as a standing regression check rather than a one-off
verification — cheap to re-run, and it would have caught both bugs
above immediately if it had existed first.

---

### How does it evolve across generations?

**What it is:** a separate, slower mechanism from lifetime learning.
Reproduction is stochastic and costs the parent; offspring inherit a
mutated *starting point* — for the reflex circuit that's evolved gains,
for the plasticity circuit specifically the inherited prior, never what
a parent personally learned. Selection only acts here.

**Where we stand:** reproduction mechanic and the prior-vs-live gain
split both built and verified directly (`decisions.md` #20, #21, #22
part 3, #26). ES trains the reflex circuit before a session starts.

**What's left:**
- done recently: verifying offspring gains match a parent's *prior*,
  not its lifetime-drifted live gains (#26)
- next: no evolutionary process exists yet for the plasticity circuit's
  *own* prior/hyperparameters — it always starts untrained (gain=1.0).
  Evolution currently only toughens the reflex, never the capacity to
  learn

**Logic for testing:** contract = an offspring's plasticity genome
always equals `parent.prior_gains + mutation`, never
`parent.get_params() + mutation` (the live, possibly-learned-from
value). Test by artificially drifting a parent's live gains away from
its prior mid-life, forcing a birth, and asserting the offspring matches
the prior and differs from the drifted live gains — already done once
as ad hoc verification (#26); worth keeping as a standing regression
check since it's exactly the kind of thing a careless refactor could
silently break.

---

### How can you check any of this is real?

**What it is:** a read-only probe, a learning-so-far scalar, and a
colony-wide summary — built so learning is checkable, not just
asserted.

**Where we stand:** `probe()`, `gain_drift()`, and
`Colony.plasticity_summary()` all built and exercised; `run_colony()`/
`run_live()` log the summary periodically, not just at the end
(`decisions.md` #26).

**What's left:**
- done recently: switching both the behavioral readout and the audit
  probe from spike count to MBON membrane potential, after spike count
  proved too coarse to track gradual synaptic change (#26)
- next: a dedicated audit CLI/visualization on top of these hooks —
  nothing renders them today beyond log lines

**Logic for testing:** contract = `probe()` must never mutate the live
agent it's called on (zero side effects), and must produce a
deterministic result for a fixed set of gains and a fixed attribute
vector. Test by calling `probe()` twice in a row with nothing in
between and asserting identical output, then asserting the agent's own
`get_params()`/`kc_trace` are byte-identical before and after — a purity
check, which matters a lot for something whose entire purpose is being
trustworthy to read without disturbing what it's reading.
