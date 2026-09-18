# Project state

> **You shape the world. They learn to survive it.**

A game where the user, through an LLM, shapes an open-ended world to
try to wipe out a fly colony that adapts to everything it encounters —
within a life, and across generations. That adaptiveness is the whole
reason this is a game and not a shooting gallery.

This is the standing checkpoint doc set — what the game *is*, right
now, in plain terms. `roadmap.md` tracks piece-by-piece implementation
status, `decisions.md` explains why things are the way they are; these
pages are for "what is the game, concretely, today," and get re-read
and updated whenever the core loop changes.

The description splits into three problems, one file each:

- **[world.md](world.md)** — the world the user shapes: how it's built,
  how the user edits it, what an item is, what's still missing from
  "open."
- **[colony.md](colony.md)** — the colony that lives in it: how a fly
  survives, perceives, decides, learns within its life, and evolves
  across generations.
- **[loop.md](loop.md)** — the loop that connects them: how time
  passes, how the user sees what's happening, what "winning" means
  today, what persists between sessions.

Each bullet point in those three files follows the same shape: **what
it is** (a brief description), **where we stand**, **what's left**
(recent steps, then next steps), and **logic for testing** — a
contract for that piece of the game (an invariant that should hold
regardless of how the implementation changes underneath it) and how to
check it holds, in one integration-style test rather than exhaustive
per-function coverage. The `Item` contract in `world.md` is the
sharpest example: whatever "an item" is at any point in this project's
life, spawning one and putting a fly next to it should always produce
an anonymous `Percept` and a Result-registry-derived effect — nothing
more should ever need re-testing just because item-related code
changed.
