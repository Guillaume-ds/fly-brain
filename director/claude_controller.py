"""Claude API implementation of WorldController -- the reference backend.

Uses real tool-use (the model picks one action from the registry, or
none) rather than parsing free-text output. Costs nothing extra for v1's
zero-argument actions and pays off the moment a future action needs a
real argument (e.g. a spawn location) -- see wiki/decisions.md.

Requires the `anthropic` package and API credentials (ANTHROPIC_API_KEY,
or any other source the SDK resolves automatically). Not exercised
against a live API in this environment -- no credentials were available
here; test with your own key before relying on it.
"""

from __future__ import annotations

import anthropic

from .actions import ActionSpec
from .base import WorldController

MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You translate a player's request into at most one world-editing action "
    "for a fly-colony survival game. If the request doesn't clearly call for "
    "one of the available actions, don't call any tool."
)


class ClaudeController(WorldController):
    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self.client = client or anthropic.Anthropic()

    def choose_action(self, request: str, actions: list[ActionSpec]) -> str | None:
        tools = [
            {
                "name": a.name,
                "description": a.description,
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
            for a in actions
        ]

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": request}],
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.name
        return None
