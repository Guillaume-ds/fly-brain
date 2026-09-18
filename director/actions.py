"""The action registry: everything a world-controller (any backend) is
allowed to do to the world. Provider-agnostic on purpose -- this file has
no knowledge that an LLM exists, only the world/env.py methods it wraps.

v1 contract (wiki/decisions.md #14) was four zero-argument actions, two
independent bounded rates, nothing else exposed. `create_item`
(decisions.md #27) was the first action with a real argument.
`create_tile`/`create_mob` (decisions.md #30-#32) extend the same
argument_schema mechanism with more than one field each, and give
`create_mob` a real enum instead of free text -- the schema itself is
what makes an unsupported request ("spits fire") have nowhere to land,
rather than the engine growing to accommodate it. `target`
(decisions.md #34) is the first OPTIONAL field any schema here has --
`required_arguments` exists specifically to let it stay that way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from world.env import Environment

# Shared by create_item and create_tile (decisions.md #32): same shape,
# same reasoning -- an embedding-driven effect whose overall magnitude,
# not shape or sign, strength authors.
_NAME_FIELD = {"type": "string", "description": "A short handle for this, e.g. 'sweet berry'."}
_STRENGTH_FIELD = {
    "type": "integer",
    "minimum": 1,
    "maximum": 5,
    "description": "Overall potency, 1 (mild) to 5 (intense). Scales magnitude only.",
}
# Shared by all three create_* actions (decisions.md #34) -- optional on
# purpose: a request that never mentions a rival shouldn't be forced to
# invent a value, and env.add_*_type()'s own `target: str | None = None`
# default means omitting it is exactly today's single-player behavior.
_TARGET_FIELD = {
    "type": "string",
    "description": "Whose territory this should appear near: 'own' for the "
    "issuing player's own colony, or another player's name to target "
    "theirs. Omit if the request doesn't mention any particular location.",
}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    fn: Callable[..., None]  # zero-arg, or called with **kwargs matching argument_schema's keys.
    # The three create_* actions return bool (created or not, decisions.md #33); the rate nudges return None.
    argument_schema: dict[str, dict] | None = None  # {param_name: JSON-schema-property}; None means zero-argument
    required_arguments: tuple[str, ...] | None = None  # None means every argument_schema key is required (unchanged default); set explicitly to exempt an optional field like `target`


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
        ActionSpec(
            "create_item",
            "Create a new kind of item in the world (something a fly can pick up and consume). "
            "Its effect on a fly is determined entirely by the description, blended across every "
            "registered effect at once -- something that reads as both food and poison will heal "
            "and hurt together, never hardcoded.",
            env.add_item_type,
            argument_schema={
                "name": _NAME_FIELD,
                "description": {
                    "type": "string",
                    "description": "A short physical description. Drives both how flies perceive "
                    "it and what it does to them -- never invent an effect outside what the "
                    "description implies.",
                },
                "strength": _STRENGTH_FIELD,
                "target": _TARGET_FIELD,
            },
            required_arguments=("name", "description", "strength"),
        ),
        ActionSpec(
            "create_tile",
            "Create a new persistent area effect in the world (e.g. a healing spring or a lava "
            "pool), from a short physical description. Applies its effect to any fly that stays "
            "within it, every tick, for as long as the fly remains -- never consumed, unlike an item.",
            env.add_tile_type,
            argument_schema={
                "name": _NAME_FIELD,
                "description": {
                    "type": "string",
                    "description": "A short physical description. Drives both how flies perceive "
                    "it and what it does to them, exactly like create_item.",
                },
                "strength": _STRENGTH_FIELD,
                "target": _TARGET_FIELD,
            },
            required_arguments=("name", "description", "strength"),
        ),
        ActionSpec(
            "create_mob",
            "Create a new kind of moving creature in the world. Unlike create_item/create_tile, "
            "its effect on a fly is NOT derived from its name or flavor -- state the effect and "
            "strength explicitly to get it. A 'healing spirit' still needs effect='heal'; the name "
            "alone only shapes how flies perceive it, never what it actually does.",
            env.add_mob_type,
            argument_schema={
                "name": {
                    "type": "string",
                    "description": "A short flavor name, e.g. 'venomous spider'. Only shapes "
                    "perception -- never the actual effect.",
                },
                "effect": {
                    "type": "string",
                    "enum": ["heal", "damage", "feed", "starve", "trap", "free"],
                    "description": "What contact with this mob does to a fly. heal/damage move "
                    "health; feed/starve move hunger; trap/free move how stuck a fly is.",
                },
                "strength": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Magnitude of the effect, 1 (weak) to 5 (lethal, for damage/starve).",
                },
                "target": _TARGET_FIELD,
            },
            required_arguments=("name", "effect", "strength"),
        ),
    ]
