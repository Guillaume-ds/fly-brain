"""Zero-dependency world-controller: crude keyword match, no LLM involved.

Exists to test the registry/Environment wiring for free before any real
model is in the loop -- same discipline used everywhere else in this
project (EscapeAgent tested with untrained weights before ES, Environment
tested with random actions before any agent). Not meant to understand
nuanced requests; that's the real controller's job.
"""

from __future__ import annotations

from .actions import ActionSpec
from .base import WorldController

_UP_WORDS = ("more", "increase", "harder", "raise")
_DOWN_WORDS = ("less", "fewer", "decrease", "easier", "lower")


class RuleBasedController(WorldController):
    def choose_action(self, request: str, actions: list[ActionSpec]) -> str | None:
        text = request.lower()
        subject = "food" if "food" in text else "spider"

        if any(w in text for w in _UP_WORDS):
            direction = "increase"
        elif any(w in text for w in _DOWN_WORDS):
            direction = "decrease"
        else:
            return None

        name = f"{direction}_{subject}_rate"
        return name if any(a.name == name for a in actions) else None
