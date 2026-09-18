"""Claude API implementation of WorldController -- the reference backend.

Uses real tool-use (the model picks one action from the registry, or
none) rather than parsing free-text output. Costs nothing extra for v1's
zero-argument actions and pays off the moment an action needs a real
argument -- `create_item` (decisions.md #27) was the first;
`create_tile`/`create_mob` (decisions.md #30-#32) extend the same
mechanism to multi-field, enum-constrained schemas.

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


def build_tool_definitions(actions: list[ActionSpec]) -> list[dict]:
    return [
        {
            "name": action.name,
            "description": action.description,
            "input_schema": {
                "type": "object",
                "properties": action.argument_schema or {},
                "required": list((action.argument_schema or {}).keys()),
            },
        }
        for action in actions
    ]


class ClaudeController(WorldController):
    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self.client = client or anthropic.Anthropic()

    def choose_action(self, request: str, actions: list[ActionSpec]) -> tuple[str, dict] | None:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            tools=build_tool_definitions(actions),
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": request}],
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.name, dict(block.input or {})
        return None
