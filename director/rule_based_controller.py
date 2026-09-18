"""Zero-dependency world-controller: crude keyword match, no LLM involved.

Exists to test the registry/Environment wiring for free before any real
model is in the loop -- same discipline used everywhere else in this
project (EscapeAgent tested with untrained weights before ES, Environment
tested with random actions before any agent). Not meant to understand
nuanced requests; that's the real controller's job. `create_item`
(decisions.md #27, #32) is handled the same crude way: a fixed trigger
phrase, everything after it taken verbatim as both name and description,
strength fixed at the midpoint -- no attempt at real language
understanding, same spirit as everything else here.

`create_tile`/`create_mob` deliberately aren't covered by this stub --
picking a real `effect` (decisions.md #30) or deciding a tile's
description is a language-understanding task, not a keyword match. That's
exactly what ClaudeController exists for.
"""

from __future__ import annotations

from world.results import MID_STRENGTH

from .actions import ActionSpec
from .base import WorldController

UP_WORDS = ("more", "increase", "harder", "raise")
DOWN_WORDS = ("less", "fewer", "decrease", "easier", "lower")
ITEM_TRIGGERS = ("create item:", "add item:", "new item:")


class RuleBasedController(WorldController):
    def choose_action(self, request: str, actions: list[ActionSpec]) -> tuple[str, dict] | None:
        text = request.lower()

        for trigger in ITEM_TRIGGERS:
            if trigger in text:
                description = request[text.index(trigger) + len(trigger):].strip()
                if description and any(a.name == "create_item" for a in actions):
                    return "create_item", {"name": description, "description": description, "strength": MID_STRENGTH}
                return None

        subject = "food" if "food" in text else "spider"
        if any(w in text for w in UP_WORDS):
            direction = "increase"
        elif any(w in text for w in DOWN_WORDS):
            direction = "decrease"
        else:
            return None

        name = f"{direction}_{subject}_rate"
        return (name, {}) if any(a.name == name for a in actions) else None
