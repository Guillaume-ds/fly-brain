# kill-the-flies frontend

The client half of the WebSocket contract in `wiki/decisions.md` #44/#45 --
Next.js (App Router, TypeScript) + Phaser 3, talking to `server/app.py`.

## Running it

1. Start the backend from the repo root: `python -m server.app` (defaults to
   `ws://localhost:8000/ws`; see `wiki/architecture.md`'s `server/` section).
2. `npm install`
3. `npm run dev`, then open the printed local URL.

To point at a non-default backend, copy `.env.example` to `.env.local` and
set `NEXT_PUBLIC_WS_URL`.

## Layout

- `src/lib/protocol.ts` -- TypeScript types mirroring `server/serialize.py`'s
  wire format field-for-field. Kept in sync by hand; there's no shared
  schema source yet.
- `src/lib/useGameSocket.ts` -- owns the one WebSocket connection, parses
  every message type, exposes plain React state.
- `src/components/GameCanvas.tsx` -- the Phaser grid: flies (colored by
  owner) and items/tiles/mobs/corpses (colored by kind) as simple shapes,
  redrawn in full every tick (no diffing yet -- matches the wire
  contract's own "full state, not a diff" choice). Hover a fly for its
  hunger/health/owner.
- `src/components/Hud.tsx`, `RequestLog.tsx`, `RequestInput.tsx`,
  `JoinScreen.tsx` -- tick/population counters, your energy meter, a
  scrolling request/death/birth log, the join screen and instruction box.

## What this is (and isn't)

A first, spectator-plus-one-player iteration: simple colored shapes, not
sprite art; one interactive owner per client, not a multi-owner switcher.
See `wiki/decisions.md` #45 for what was decided and why.
