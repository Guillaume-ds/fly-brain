# kill-the-flies wiki

> **You shape the world. They learn to survive it.**

A game where the user, through an LLM, shapes an open-ended world to
try to wipe out a fly colony that adapts to everything it encounters —
within a life, and across generations. The user never acts on the flies
directly — only on the world around them.

This is a working wiki, not a polished writeup — it's maintained turn by
turn as the project develops, so it stays an accurate source of truth for
"how does this work" and "why did we do it this way," not a snapshot from
day one. It's also part of the point of this project: showing how it was
actually built with AI assistance, decision by decision, not just the
end result.

## The game

Start here — what the game *is*, right now, split into the three
problems it's actually made of. Re-read and updated whenever the core
loop changes.

- **[state.md](state.md)** — the index: the catchphrase, how the three
  files below fit together, and the description/status/todo/testing-logic
  shape every bullet in them follows.
- **[world.md](world.md)** — the world the user shapes.
- **[colony.md](colony.md)** — the colony that lives in it.
- **[loop.md](loop.md)** — the loop that connects them.

## The build

How the game above actually got built, and why.

- **[architecture.md](architecture.md)** — how the project is structured,
  what each module does, and the interfaces between them.
- **[stack.md](stack.md)** — what libraries/tools are used and why.
- **[decisions.md](decisions.md)** — a log of the real design decisions
  made along the way, with the reasoning and the alternatives that were
  considered and rejected.
- **[roadmap.md](roadmap.md)** — current status and what's next.

If you're picking this project back up after a while, or an AI assistant
is resuming work on it, start with `state.md` for "what is this,"
`roadmap.md` for "where are we," then `decisions.md` for "why is it
built this way" before changing anything load-bearing.
