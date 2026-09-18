"""Claude API implementation of WorldController -- the reference backend.

Uses real tool-use (the model picks one action from the registry, or
none) rather than parsing free-text output. Costs nothing extra for v1's
zero-argument actions and pays off the moment a future action needs a
real argument -- that moment is `create_item` (decisions.md #27), the
first action with a real input_schema instead of an empty one.

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
            "input_schema": (
                {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                }
                if action.takes_argument
                else {"type": "object", "properties": {}, "required": []}
            ),
        }
        for action in actions
    ]


class ClaudeController(WorldController):
    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self.client = client or anthropic.Anthropic()

    def choose_action(self, request: str, actions: list[ActionSpec]) -> tuple[str, str | None] | None:
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
                argument = (block.input or {}).get("description")
                return block.name, argument
        return None
