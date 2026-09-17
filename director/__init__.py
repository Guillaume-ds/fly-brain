"""The swappable LLM world-controller layer.

A player's free-text request gets translated into one call against a
fixed registry of world-editing actions (director/actions.py). Which
backend does the translating -- Claude, a local model, a keyword stub --
is an implementation detail behind director/base.py's WorldController
interface; the registry and world/env.py never know which one is in use.
See wiki/decisions.md #14 for the v1 contract this implements.
"""
