"""The Result registry: a small, fixed set of real mechanical outcomes an
item can have on a fly, each encoded the same way items are (see
wiki/decisions.md #22 part 5). An item's actual effect is a
clipped-cosine-similarity blend across every registered Result, applied
simultaneously -- so an item close to both `food` and `damage` heals and
hurts at once (Minecraft-rotten-flesh-style), with no per-item authored
number needed.

This stays entirely on the world/mechanics side: a Result computes real
deltas to a Fly's physiological channels (hunger/health/stuck_ticks).
Reward/dopamine is derived from those real deltas elsewhere (fly_brain/,
not built yet) -- never from this similarity score directly, which is
the rule that keeps this consistent with #22 part 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .items import ItemEncoder

Channel = str  # "hunger" | "health" | "stuck_ticks"


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
    max_stuck_ticks: int = 10,
) -> list[ResultConcept]:
    """The starting Result vocabulary: food (heals), damage (hurts),
    immobilize (sticks). Extensible later (e.g. a `stick`/web-specific
    Result down the roadmap) without touching anything else -- this is
    the one place in the system that's deliberately discrete rather than
    open-ended, see decisions.md #22 part 5 for why.
    """
    return [
        ResultConcept(
            name="food",
            reference_vector=encoder.encode("nourishing, edible food"),
            channel="hunger",
            sign=1.0,
            scale=float(max_hunger),
        ),
        ResultConcept(
            name="damage",
            reference_vector=encoder.encode("toxic, dangerous, harmful poison"),
            channel="health",
            sign=-1.0,
            scale=float(max_health),
        ),
        ResultConcept(
            name="immobilize",
            reference_vector=encoder.encode("sticky, tangling, trapping web"),
            channel="stuck_ticks",
            sign=1.0,
            scale=float(max_stuck_ticks),
        ),
    ]
