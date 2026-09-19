"""FastAPI + WebSocket layer (wiki/decisions.md #44) -- wraps the exact
same loop `game/live_run.py` already runs (a continuously-ticking
`Colony` driven by a swappable `director/` `WorldController`) behind a
WebSocket instead of stdin/print. Reuses `game.live_run.apply_requests()`
directly rather than reimplementing request handling -- it's already
pure, tested logic.

One shared `Colony` per server process, broadcast to every connected
client -- matches the "two players share one world" vision (`decisions.md`
#13), not one world per connection. A client tells the server who it is
via a `join` message; the server spawns a fresh colony for a name it
hasn't seen before (`Colony.add_colony()`, same mechanism `decisions.md`
#34/#35 already built for a second owner) and is a no-op for one it has.

No thread, unlike `live_run.py`'s stdin reader: everything here runs on
one asyncio event loop. `Colony.step()` and `apply_requests()` are both
synchronous with no `await` inside them, so once a task starts running
either, it runs to completion before anything else on the loop gets a
turn -- safe without any extra locking, and requests apply immediately
on receipt rather than needing a queue-and-drain pattern the way a real
OS thread would.

A client can also `watch` a single fly_id: after every tick, whichever
connections are watching a live fly each get their own `brain` message
carrying `Colony.brain_snapshot()` (decisions.md #46) for just that fly
-- a per-connection send, not a broadcast, since different clients can
watch different flies. Watching costs nothing for connections that
aren't doing it (`brain_snapshot()` is only ever called for fly ids
someone actually asked to watch).

Run as: python -m server.app  (or: uvicorn server.app:app)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from director.actions import ActionSpec, build_registry
from director.base import WorldController
from director.claude_controller import ClaudeController
from director.rule_based_controller import RuleBasedController
from fly_brain.agent import build_escape_template, load_starting_gains
from fly_brain.plasticity import build_plasticity_template
from game.colony import Colony
from game.live_run import UNBOUNDED_TICKS, apply_requests
from training.run import CHECKPOINT_DIR
from world.env import DEFAULT_OWNER, Environment
from world.items import HashingItemEncoder

from .serialize import serialize_tick, serialize_world_init

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "stage1_clean_escape.npy"
JOIN_POPULATION = 3  # flies a newly-joining owner's colony starts with


@dataclass
class AppState:
    colony: Colony
    controller: WorldController
    actions: list[ActionSpec]
    escape_gains: np.ndarray  # so a joining owner's colony starts from the same trained/loaded gains
    tick_interval: float
    connections: set[WebSocket] = field(default_factory=set)
    # decisions.md #46: which fly (if any) each connected client wants a
    # "brain" message for after every tick. Entry exists for every
    # connected websocket from the moment it connects (None = not
    # watching); populated/cleared by the "watch" message and cleared
    # back to None by send_brain_updates() once a watched fly dies.
    watching: dict[WebSocket, int | None] = field(default_factory=dict)


def build_controller(name: str) -> WorldController:
    if name == "claude":
        return ClaudeController()
    return RuleBasedController()


def build_state(
    checkpoint: pathlib.Path = DEFAULT_CHECKPOINT,
    initial_population: int = 3,
    max_population: int = 20,
    mutation_sigma: float = 0.1,
    ticks_per_second: float = 5.0,
    controller_name: str = "rule_based",
    seed: int = 0,
) -> AppState:
    """The real, expensive default: loads the actual connectome and
    escape checkpoint, exactly like `live_run.py`'s `main()` does. Tests
    build a much cheaper `AppState` directly instead of calling this --
    see `tests/test_server.py`.
    """
    encoder = HashingItemEncoder()
    escape_template = build_escape_template(encoder)
    plasticity_template = build_plasticity_template(encoder)
    gains = load_starting_gains(escape_template, checkpoint)

    env = Environment(
        initial_population=initial_population,
        max_population=max_population,
        max_ticks=UNBOUNDED_TICKS,
        encoder=encoder,
        seed=seed,
    )
    colony = Colony(env, escape_template, plasticity_template, gains, mutation_sigma=mutation_sigma, seed=seed)
    controller = build_controller(controller_name)
    actions = build_registry(env)

    return AppState(
        colony=colony,
        controller=controller,
        actions=actions,
        escape_gains=gains,
        tick_interval=1.0 / ticks_per_second,
    )


async def broadcast(connections: set[WebSocket], payload: dict) -> None:
    dead: set[WebSocket] = set()
    for websocket in connections:
        try:
            await websocket.send_json(payload)
        except Exception:
            dead.add(websocket)
    connections -= dead


async def send_brain_updates(state: AppState) -> None:
    """Per-connection, not a broadcast -- each client watches at most one
    fly, and different clients can watch different flies (decisions.md
    #46). `colony.brain_snapshot()` returning `None` means the watched
    fly died this tick: sent through as `data: null` so the frontend can
    show "this fly died" once, then the watch is cleared server-side so
    it isn't recomputed (as a `None` lookup) every tick after.
    """
    dead: set[WebSocket] = set()
    for websocket, fly_id in list(state.watching.items()):
        if fly_id is None:
            continue
        snapshot = state.colony.brain_snapshot(fly_id)
        if snapshot is None:
            state.watching[websocket] = None
        try:
            await websocket.send_json({"type": "brain", "data": snapshot})
        except Exception:
            dead.add(websocket)
    for websocket in dead:
        state.watching.pop(websocket, None)
        state.connections.discard(websocket)


async def tick_loop(state: AppState) -> None:
    """Runs forever -- a server outlives any one colony's extinction,
    unlike `live_run.py`'s CLI session, which ends when its one colony
    dies. A fully extinct world just keeps ticking with nothing in it
    until a new `join` spawns a fresh colony into it.

    Sleeps BEFORE stepping, not after: this task starts as soon as the
    server's lifespan begins, which can be before any client has
    connected at all. Stepping first would mean tick 1 (and anything
    that happens on it -- a death, a birth) gets broadcast to whoever
    happens to be connected at that instant, possibly nobody, and is
    then gone forever; sleeping first gives every freshly-started server
    one full `tick_interval` of grace before the world can change under
    a client that's still in the middle of connecting.
    """
    while True:
        await asyncio.sleep(state.tick_interval)
        result = state.colony.step()
        await broadcast(state.connections, {"type": "tick", "data": serialize_tick(state.colony.env, result)})
        await send_brain_updates(state)


async def handle_message(message: dict, websocket: WebSocket, state: AppState) -> None:
    message_type = message.get("type")
    data = message.get("data") or {}

    if message_type == "join":
        owner = data.get("owner")
        if not owner:
            await websocket.send_json({"type": "error", "data": {"message": "join requires 'owner'"}})
            return
        if owner not in state.colony.env.owners:
            state.colony.add_colony(owner, JOIN_POPULATION, state.escape_gains)
        await websocket.send_json({"type": "joined", "data": {"owner": owner}})

    elif message_type == "player_request":
        owner = data.get("owner", DEFAULT_OWNER)
        text = data.get("text", "")
        for request_text, status in apply_requests([(text, owner)], state.controller, state.actions):
            await websocket.send_json({"type": "request_result", "data": {"text": request_text, "status": status}})

    elif message_type == "watch":
        # decisions.md #46: fly_id=None explicitly unwatches (the "stop
        # watching" case, e.g. the frontend closing the brain panel), not
        # just "no argument given" -- same field either way, so there's
        # no separate "unwatch" message type to keep in sync.
        fly_id = data.get("fly_id")
        state.watching[websocket] = fly_id
        snapshot = state.colony.brain_snapshot(fly_id) if fly_id is not None else None
        await websocket.send_json({"type": "brain", "data": snapshot})

    else:
        await websocket.send_json({"type": "error", "data": {"message": f"unknown message type {message_type!r}"}})


def create_app(state: AppState | None = None) -> FastAPI:
    """`state=None` builds the real default at startup (the production
    path). Tests pass a cheap, pre-built `AppState` instead, so a test
    run never pays for loading the real connectome or checkpoint.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        game_state = state if state is not None else build_state()
        app.state.game = game_state
        task = asyncio.create_task(tick_loop(game_state))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    application = FastAPI(lifespan=lifespan)

    @application.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        game_state: AppState = websocket.app.state.game
        await websocket.accept()
        game_state.connections.add(websocket)
        game_state.watching[websocket] = None
        await websocket.send_json({"type": "world_init", "data": serialize_world_init(game_state.colony.env)})
        try:
            while True:
                message = await websocket.receive_json()
                await handle_message(message, websocket, game_state)
        except WebSocketDisconnect:
            pass
        finally:
            game_state.connections.discard(websocket)
            game_state.watching.pop(websocket, None)

    return application


app = create_app()  # module-level default, for `uvicorn server.app:app`


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    host = os.environ.get("FLY_BRAIN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLY_BRAIN_PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
