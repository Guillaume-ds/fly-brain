"""The interface every world-controller backend must satisfy. This is the
entire swap point: implement this once and any backend -- Claude, a local
model, a keyword stub -- can drive the same action registry, and the rest
of the system never needs to know which one is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .actions import ActionSpec


class WorldController(ABC):
    @abstractmethod
    def choose_action(self, request: str, actions: list[ActionSpec]) -> tuple[str, dict] | None:
        """Return (action name, arguments) for one action in `actions`
        to run for this request, or None if no action applies.
        `arguments` is a dict of keyword arguments matching that
        action's argument_schema (decisions.md #27, #30-#32); pass {}
        when argument_schema is None."""
