"""Zero-dependency world-controller: crude keyword match, no LLM involved.

Exists to test the registry/Environment wiring for free before any real
model is in the loop -- same discipline used everywhere else in this
project (EscapeAgent tested with untrained weights before ES, Environment
tested with random actions before any agent). Not meant to understand
nuanced requests; that's the real controller's job. `create_item`
(decisions.md #27) is handled the same crude way: a fixed trigger
phrase, everything after it taken verbatim as the description -- no
attempt at real language understanding, same spirit as everything else
here.
"""

from __future__ import annotations

from .actions import ActionSpec
from .base import WorldController

UP_WORDS = ("more", "increase", "harder", "raise")
DOWN_WORDS = ("less", "fewer", "decrease", "easier", "lower")
ITEM_TRIGGERS = ("create item:", "add item:", "new item:")


class RuleBasedController(WorldController):
    def choose_action(self, request: str, actions: list[ActionSpec]) -> tuple[str, str | None] | None:
        text = request.lower()

        for trigger in ITEM_TRIGGERS:
            if trigger in text:
                description = request[text.index(trigger) + len(trigger):].strip()
                if description and any(a.name == "create_item" for a in actions):
                    return "create_item", description
                return None

        subject = "food" if "food" in text else "spider"
        if any(w in text for w in UP_WORDS):
            direction = "increase"
        elif any(w in text for w in DOWN_WORDS):
            direction = "decrease"
        else:
            return None

        name = f"{direction}_{subject}_rate"
        return (name, None) if any(a.name == name for a in actions) else None
