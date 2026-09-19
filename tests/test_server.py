"""The FastAPI+WebSocket layer (wiki/decisions.md #44).

Uses a cheap, directly-constructed `AppState` rather than
`server.app.build_state()` -- the real default loads the actual
connectome and ES checkpoint, which is what every other test file in
this project avoids too (see e.g. `tests/test_wander.py`'s own
module-scoped `templates` fixture, mirrored here).

The background tick loop runs concurrently with every test (FastAPI's
`TestClient` runs the whole ASGI app, lifespan included, in a real
thread). Most tests use a very slow tick interval so an automatic `tick`
broadcast can't interleave with the specific messages a test expects;
`test_tick_broadcasts_real_state` is the one test that turns the
interval down and actually waits for one.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from game.colony import Colony
from server.app import AppState, build_controller, create_app
from world.env import Action, DEFAULT_OWNER, Environment

OWNER = DEFAULT_OWNER
SLOW_TICK_INTERVAL = 60.0  # effectively "never" within one test's lifetime


@pytest.fixture(scope="module")
def templates():
    from fly_brain.agent import build_escape_template
    from fly_brain.plasticity import build_plasticity_template

    env = Environment(initial_population=1, seed=1)
    encoder = env.encoder
    return build_escape_template(encoder), build_plasticity_template(encoder)


def make_state(templates, tick_interval: float = SLOW_TICK_INTERVAL, initial_population: int = 1) -> AppState:
    from director.actions import build_registry

    escape_template, plasticity_template = templates
    env = Environment(
        initial_population=initial_population,
        food_spawn_rate=0.0,
        spider_spawn_rate=0.0,
        item_spawn_rate=0.0,
        mob_spawn_rate=0.0,
        tile_spawn_rate=0.0,
        starting_energy=1000.0,
        seed=3,
    )
    gains = np.ones(len(escape_template.blueprint.edges))
    colony = Colony(env, escape_template, plasticity_template, gains, seed=3)
    return AppState(
        colony=colony,
        controller=build_controller("rule_based"),
        actions=build_registry(env),
        escape_gains=gains,
        tick_interval=tick_interval,
    )


def receive_until(ws, message_type: str, max_messages: int = 20) -> dict:
    """Reads messages until one of `message_type` arrives, tolerating
    (and discarding) any interleaved `tick` broadcasts along the way.
    """
    for _ in range(max_messages):
        message = ws.receive_json()
        if message["type"] == message_type:
            return message
    raise AssertionError(f"no {message_type!r} message within {max_messages} messages")


# --- connecting ------------------------------------------------------------

def test_world_init_is_sent_on_connect(templates):
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
        assert message["type"] == "world_init"
        assert message["data"]["grid_size"] == 20
        assert any(t["name"] == "food" for t in message["data"]["item_types"])
        assert any(t["name"] == "spider" for t in message["data"]["mob_types"])


def test_brain_topology_is_sent_once_right_after_world_init(templates):
    """decisions.md #47: static, colony-wide -- sent once per connection,
    not repeated per tick or per watched fly the way `brain`/
    `neuron_state` are.
    """
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        second = ws.receive_json()
        assert first["type"] == "world_init"
        assert second["type"] == "brain_topology"
        assert set(second["data"]) == {"escape", "plasticity"}
        assert set(second["data"]["escape"]) == {"neuron_ids", "edges", "groups"}


# --- join --------------------------------------------------------------

def test_join_with_a_new_owner_spawns_a_colony(templates):
    state = make_state(templates)
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        assert "rival" not in state.colony.env.owners

        ws.send_json({"type": "join", "data": {"owner": "rival"}})
        ack = receive_until(ws, "joined")

        assert ack["data"]["owner"] == "rival"
        assert "rival" in state.colony.env.owners
        assert any(f.owner == "rival" for f in state.colony.env.flies)


def test_join_with_an_existing_owner_is_a_no_op(templates):
    state = make_state(templates)
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        population_before = state.colony.population

        ws.send_json({"type": "join", "data": {"owner": OWNER}})
        receive_until(ws, "joined")

        assert state.colony.population == population_before


def test_join_without_an_owner_is_an_error(templates):
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.send_json({"type": "join", "data": {}})
        error = receive_until(ws, "error")
        assert "owner" in error["data"]["message"]


# --- player_request -----------------------------------------------------

def test_player_request_applies_and_acks(templates):
    state = make_state(templates)
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        rate_before = state.colony.env.spider_type.spawn_rate

        ws.send_json({"type": "player_request", "data": {"owner": OWNER, "text": "more spiders"}})
        result = receive_until(ws, "request_result")

        assert result["data"]["text"] == "more spiders"
        assert result["data"]["status"] == "increase_spider_rate"
        assert state.colony.env.spider_type.spawn_rate > rate_before


def test_player_request_with_no_matching_action(templates):
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.send_json({"type": "player_request", "data": {"owner": OWNER, "text": "asdkjfh nonsense"}})
        result = receive_until(ws, "request_result")
        assert result["data"]["status"] is None


def test_unknown_message_type_is_an_error(templates):
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.send_json({"type": "not_a_real_type", "data": {}})
        error = receive_until(ws, "error")
        assert "not_a_real_type" in error["data"]["message"]


# --- the tick broadcast --------------------------------------------------

def test_tick_broadcasts_real_state(templates):
    state = make_state(templates, tick_interval=0.2)
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        tick = receive_until(ws, "tick", max_messages=50)

        assert tick["data"]["tick"] >= 1
        assert len(tick["data"]["flies"]) == 1
        fly_payload = tick["data"]["flies"][0]
        assert fly_payload["owner"] == OWNER
        assert set(fly_payload) == {"id", "x", "y", "hunger", "health", "owner", "stuck", "vulnerable"}


def test_a_death_shows_up_in_tick_events(templates):
    state = make_state(templates, tick_interval=0.2)
    state.colony.env.flies[0].health = 0  # dies on the first step
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        tick = receive_until(ws, "tick", max_messages=50)

        assert len(tick["data"]["events"]["deaths"]) == 1
        assert tick["data"]["events"]["deaths"][0]["cause"] == "damage"


# --- watch / brain (decisions.md #46) ---------------------------------------

def test_watch_replies_with_an_immediate_brain_snapshot(templates):
    state = make_state(templates)
    fly_id = next(iter(state.colony.observations))
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init

        ws.send_json({"type": "watch", "data": {"fly_id": fly_id}})
        brain = receive_until(ws, "brain")

        assert brain["data"]["fly_id"] == fly_id


def test_watch_with_fly_id_none_replies_with_a_null_snapshot(templates):
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init

        ws.send_json({"type": "watch", "data": {"fly_id": None}})
        brain = receive_until(ws, "brain")

        assert brain["data"] is None


def test_watched_fly_gets_a_brain_message_after_every_tick(templates):
    state = make_state(templates, tick_interval=0.2)
    fly_id = next(iter(state.colony.observations))
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.send_json({"type": "watch", "data": {"fly_id": fly_id}})
        receive_until(ws, "brain")  # the immediate reply to "watch" itself

        brain = receive_until(ws, "brain", max_messages=50)

        assert brain["data"]["fly_id"] == fly_id
        assert brain["data"]["action_source"] in {"escape", "wander", "plasticity"}


def test_a_second_connection_can_watch_a_different_fly(templates):
    state = make_state(templates, tick_interval=0.2, initial_population=2)
    fly_ids = list(state.colony.observations)
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        ws_a.receive_json()  # world_init
        ws_b.receive_json()  # world_init
        ws_a.send_json({"type": "watch", "data": {"fly_id": fly_ids[0]}})
        ws_b.send_json({"type": "watch", "data": {"fly_id": fly_ids[1]}})
        receive_until(ws_a, "brain")
        receive_until(ws_b, "brain")

        brain_a = receive_until(ws_a, "brain", max_messages=50)
        brain_b = receive_until(ws_b, "brain", max_messages=50)

        assert brain_a["data"]["fly_id"] == fly_ids[0]
        assert brain_b["data"]["fly_id"] == fly_ids[1]


def test_brain_goes_null_and_stops_once_the_watched_fly_dies(templates):
    state = make_state(templates, tick_interval=0.2)
    fly_id = next(iter(state.colony.observations))
    state.colony.env.flies[0].health = 0  # dies on the first step
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.send_json({"type": "watch", "data": {"fly_id": fly_id}})
        receive_until(ws, "brain")  # the immediate reply to "watch" itself

        brain = receive_until(ws, "brain", max_messages=50)
        assert brain["data"] is None

        # confirm the server actually stopped computing snapshots for the
        # dead fly rather than resending null forever -- wait through a
        # couple more tick intervals and make sure no further "brain"
        # message shows up.
        with pytest.raises(AssertionError):
            receive_until(ws, "brain", max_messages=10)


# --- neuron_state (decisions.md #47) ----------------------------------------

def test_watch_replies_with_an_immediate_neuron_state(templates):
    state = make_state(templates)
    fly_id = next(iter(state.colony.observations))
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.receive_json()  # brain_topology

        ws.send_json({"type": "watch", "data": {"fly_id": fly_id}})
        neurons = receive_until(ws, "neuron_state")

        assert neurons["data"]["fly_id"] == fly_id
        assert set(neurons["data"]["escape"]) == {"v", "spikes"}
        assert set(neurons["data"]["plasticity"]) == {"v", "spikes"}


def test_watch_with_fly_id_none_replies_with_a_null_neuron_state(templates):
    app = create_app(make_state(templates))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.receive_json()  # brain_topology

        ws.send_json({"type": "watch", "data": {"fly_id": None}})
        neurons = receive_until(ws, "neuron_state")

        assert neurons["data"] is None


def test_neuron_state_arrays_match_the_topology_neuron_counts(templates):
    state = make_state(templates, tick_interval=0.2)
    fly_id = next(iter(state.colony.observations))
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        topology = ws.receive_json()  # brain_topology
        ws.send_json({"type": "watch", "data": {"fly_id": fly_id}})
        receive_until(ws, "neuron_state")  # the immediate reply to "watch" itself

        neurons = receive_until(ws, "neuron_state", max_messages=50)

        assert len(neurons["data"]["escape"]["v"]) == len(topology["data"]["escape"]["neuron_ids"])
        assert len(neurons["data"]["plasticity"]["v"]) == len(topology["data"]["plasticity"]["neuron_ids"])


def test_neuron_state_goes_null_once_the_watched_fly_dies(templates):
    state = make_state(templates, tick_interval=0.2)
    fly_id = next(iter(state.colony.observations))
    state.colony.env.flies[0].health = 0  # dies on the first step
    app = create_app(state)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # world_init
        ws.receive_json()  # brain_topology
        ws.send_json({"type": "watch", "data": {"fly_id": fly_id}})
        receive_until(ws, "neuron_state")  # the immediate reply to "watch" itself

        neurons = receive_until(ws, "neuron_state", max_messages=50)
        assert neurons["data"] is None
