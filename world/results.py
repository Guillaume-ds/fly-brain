"""The Result registry: a small, fixed set of real mechanical outcomes an
item can have on a fly, each encoded the same way items are (see
wiki/decisions.md #22 part 5). An item's actual effect is a
clipped-cosine-similarity blend across every registered Result, applied
simultaneously -- so an item close to both `food` and `damage` heals and
hurts at once (Minecraft-rotten-flesh-style), with no per-item authored
number needed.

This stays entirely on the world/mechanics side: a Result computes real
deltas to a Fly's physiological channels (hunger/health/stuck_ticks).
Reward/dopamine is derived from those real deltas by fly_brain/'s
plasticity circuit -- never from this similarity score directly, which
is the rule that keeps this consistent with #22 part 2.

Also home to `Effect` (decisions.md #30-#32): a mob's effect is authored
from `effect`/`strength` directly, never derived from its attribute
vector the way an item's/tile's is -- the encoder can't be trusted with
anything that could invalidate the frozen escape circuit's training.
`Effect` lives here rather than in items.py because it's fundamentally
about which `Channel` moves and in which direction, the same vocabulary
`ResultConcept` already uses.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .items import ItemEncoder


class Channel(enum.StrEnum):
    """The physiological channels a Result can move. Each member's value
    is exactly the `Fly` attribute it writes to -- that link is what
    lets `Environment.apply_result()` stay generic instead of hardcoding
    one branch per channel (decisions.md #28). Adding a channel means
    adding a member, a matching `Fly` field, and a bound in
    `Environment.channel_limits`; anything missed raises rather than
    being silently dropped.
    """

    HUNGER = "hunger"
    HEALTH = "health"
    STUCK_TICKS = "stuck_ticks"


@dataclass(frozen=True)
class ResultConcept:
    name: str
    reference_vector: np.ndarray
    channel: Channel
    sign: float  # +1.0 or -1.0 -- the direction of this Result's effect on its channel
    scale: float  # channel units at similarity == 1.0


def blend_deltas(attributes: np.ndarray, concepts: list[ResultConcept]) -> dict[Channel, float]:
    """The real mechanical effect of one item on one fly: for each
    registered Result, similarity to its reference vector (clipped at 0,
    so unrelated Results contribute nothing) scales that Result's effect
    on its channel. Multiple Results affecting the same channel add up.
    """
    deltas: dict[Channel, float] = {}
    for concept in concepts:
        weight = max(0.0, float(np.dot(attributes, concept.reference_vector)))
        deltas[concept.channel] = deltas.get(concept.channel, 0.0) + concept.sign * concept.scale * weight
    return deltas


def build_default_results(
    encoder: ItemEncoder,
    max_hunger: int,
    max_health: int,
    max_stuck_ticks: int,
) -> list[ResultConcept]:
    """The starting Result vocabulary: food (heals), damage (hurts),
    immobilize (sticks). Adding a Result that reuses an existing
    `Channel` needs nothing else changed; adding one on a *new* channel
    also needs a `Channel` member, a `Fly` field, and a bound in
    `Environment.channel_limits` (see `Channel`'s docstring). This
    registry is the one place in the system that's deliberately discrete
    rather than open-ended -- see decisions.md #22 part 5 for why.
    """
    return [
        ResultConcept(
            name="food",
            reference_vector=encoder.encode("nourishing, edible food"),
            channel=Channel.HUNGER,
            sign=1.0,
            scale=float(max_hunger),
        ),
        ResultConcept(
            name="damage",
            reference_vector=encoder.encode("toxic, dangerous, harmful poison"),
            channel=Channel.HEALTH,
            sign=-1.0,
            scale=float(max_health),
        ),
        ResultConcept(
            name="immobilize",
            reference_vector=encoder.encode("sticky, tangling, trapping web"),
            channel=Channel.STUCK_TICKS,
            sign=1.0,
            scale=float(max_stuck_ticks),
        ),
    ]


class Effect(enum.StrEnum):
    """An authored mob effect (decisions.md #30-#32): what contact with a
    mob actually does to a fly, read directly rather than derived from
    its embedding. Each member names an outcome a player/translator can
    ask for directly -- `strength`'s sign never has to be inferred, the
    name already says which way it goes. `EFFECT_CHANNELS` says which
    `Channel` each one moves and in which direction relative to a
    positive `strength` (positive strength always helps the fly,
    whichever channel it targets -- see `mob_effect_delta`).
    """

    HEAL = "heal"
    DAMAGE = "damage"
    FEED = "feed"
    STARVE = "starve"
    TRAP = "trap"
    FREE = "free"


EFFECT_CHANNELS: dict[Effect, tuple[Channel, float]] = {
    Effect.HEAL: (Channel.HEALTH, 1.0),
    Effect.DAMAGE: (Channel.HEALTH, -1.0),
    Effect.FEED: (Channel.HUNGER, 1.0),
    Effect.STARVE: (Channel.HUNGER, -1.0),
    Effect.TRAP: (Channel.STUCK_TICKS, 1.0),
    Effect.FREE: (Channel.STUCK_TICKS, -1.0),
}

# Effects capable of killing a fly outright (HEALTH/HUNGER can reach zero).
# TRAP is harmful but never lethal on its own -- being stuck is unpleasant,
# not deadly. This is exactly what decides whether create_mob's mandatory
# embedding clause gets forced (decisions.md #30, #31): the frozen escape
# reflex's one real exposure to a mimic is a LETHAL_EFFECTS mob with a
# harmless-sounding name.
LETHAL_EFFECTS = {Effect.DAMAGE, Effect.STARVE}

STRENGTH_BOUNDS = (1, 5)
MID_STRENGTH = 3  # the strength at which item/tile magnitude scaling is exactly 1.0x -- unscaled, today's behavior
MOB_EFFECT_FRACTION = 0.1  # fraction of a channel's max moved per strength point per tick of mob contact -- graded, not instant (decisions.md #31)
TILE_EFFECT_FRACTION = 0.1  # fraction of a one-shot item pickup's effect a tile applies per tick a fly stays within it (decisions.md #31)

STRENGTH_WORDS = {1: "weak", 2: "minor", 3: "moderate", 4: "strong", 5: "lethal"}


def clamp_strength(strength: int) -> int:
    """Defensive, not advisory: `strength`'s [1, 5] bound is stated in the
    director schema for a well-behaved translator, but nothing stops a
    real one from sending something outside it -- this is what actually
    enforces the bound.
    """
    return min(max(int(strength), STRENGTH_BOUNDS[0]), STRENGTH_BOUNDS[1])


def strength_magnitude(strength: int) -> float:
    """Item/tile magnitude scaling (decisions.md #32): scales the
    description-derived blend's overall potency, never its shape or
    sign. `strength` never picks a channel or a direction here -- only
    the description does that -- so the pillar item contract (effect
    comes purely from the Result registry) is untouched by this.
    """
    return clamp_strength(strength) / MID_STRENGTH


def mob_effect_delta(effect: Effect, strength: int, channel_limits: dict[Channel, int]) -> tuple[Channel, float]:
    """The real mechanical effect of a mob on a fly, for one tick of
    contact -- authored from `effect`/`strength`, never `blend_deltas()`
    (decisions.md #30). Graded rather than instant (decisions.md #31): a
    per-tick fraction of the target channel's own max, scaled by
    strength, so a weak mob is survivable and a strong one still is
    dangerous without being an unconditional insta-kill.
    """
    channel, sign = EFFECT_CHANNELS[effect]
    delta = sign * clamp_strength(strength) * channel_limits[channel] * MOB_EFFECT_FRACTION
    return channel, delta


def compose_mob_description(name: str, effect: Effect, strength: int) -> str:
    """The string handed to the encoder for a created mob (decisions.md
    #30-#32). The mandatory clause is forced only when `effect` is in
    `LETHAL_EFFECTS` -- that's the frozen escape reflex's one real
    exposure to a mimic, since `heal`/`feed`/`free`/`trap` mobs can
    cost a fly a missed benefit or an unpleasant delay, never its life.
    Worded from this module's own `damage` Result vocabulary, since
    similarity to exactly that vector is what `sense_danger()` measures
    (decisions.md #24, #30).
    """
    if effect not in LETHAL_EFFECTS:
        return name
    channel, _ = EFFECT_CHANNELS[effect]
    domain = "toxic, harmful poison" if channel == Channel.HEALTH else "draining, harmful"
    word = STRENGTH_WORDS[clamp_strength(strength)]
    return f"{name}, a dangerous, fast predator dealing {word} {domain} damage"
