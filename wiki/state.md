# Project state

> **You shape the world. They learn to survive it.**

Two players — each either a human or an AI — send instructions to the
game engine. Those instructions shape an open-ended world: its items,
its mobs, its environment. The world acts on a fly colony living in it,
and the colony learns from every interaction it survives — within a
life, and across generations. One player is trying to grow the colony,
the other to wipe it out. Neither one ever touches a fly directly.

    instructions → world → flies → learning → back to the players

That single chain is the whole game, and every piece of this project
sits somewhere on it. The colony's adaptiveness is what makes it a game
rather than a shooting gallery: the same instruction stops working once
the flies have lived through it.

**Status of the framing itself:** the chain above is built and running
end to end for *one* instruction source and *one* element kind —
instructions become items (`director/`'s `create_item`), and items act
on flies. Two simultaneous players (`decisions.md` #29) and the other
two element kinds (mobs the players can create, environment tiles at
all) are the framing this is being built toward, not what exists today.
Each of the three files below says exactly where its own half of the
chain stands.

This is the standing checkpoint doc set — what the game *is*, right
now, in plain terms. `roadmap.md` tracks piece-by-piece implementation
status, `decisions.md` explains why things are the way they are; these
pages are for "what is the game, concretely, today," and get re-read
and updated whenever the core loop changes.

The description splits into three problems, one file each:

- **[world.md](world.md)** — the world the players shape: how it's
  built, how instructions reach it, what an item is, what's still
  missing from "open."
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
