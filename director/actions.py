"""The action registry: everything a world-controller (any backend) is
allowed to do to the world. Provider-agnostic on purpose -- this file has
no knowledge that an LLM exists, only the world/env.py methods it wraps.

v1 contract (wiki/decisions.md #14): four zero-argument actions, two
independent bounded rates, nothing else exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from world.env import Environment


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    fn: Callable[[], None]


def build_registry(env: Environment) -> list[ActionSpec]:
    return [
        ActionSpec(
            "increase_spider_rate",
            "Make spiders spawn more often, increasing danger to the fly colony.",
            env.increase_spider_rate,
        ),
        ActionSpec(
            "decrease_spider_rate",
            "Make spiders spawn less often, decreasing danger to the fly colony.",
            env.decrease_spider_rate,
        ),
        ActionSpec(
            "increase_food_rate",
            "Make food spawn more often, helping the fly colony survive.",
            env.increase_food_rate,
        ),
        ActionSpec(
            "decrease_food_rate",
            "Make food spawn less often, making survival harder for the fly colony.",
            env.decrease_food_rate,
        ),
    ]
