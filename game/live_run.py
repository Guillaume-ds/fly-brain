"""CLI entry point: python -m game.live_run

The live game loop: connects director/'s swappable WorldController to a
continuously-ticking Colony. A background thread reads player requests
from stdin into a queue; the main loop ticks the colony at a fixed
real-time rate and drains/applies any pending requests each tick, so
typing a request never pauses the world (see wiki/roadmap.md, "Live
game loop").
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import queue
import threading
import time

from director.actions import ActionSpec, build_registry
from director.base import WorldController
from director.claude_controller import ClaudeController
from director.rule_based_controller import RuleBasedController
from fly_brain.agent import build_escape_template
from fly_brain.plasticity import build_plasticity_template
from world.env import DEFAULT_OWNER, Environment
from world.items import HashingItemEncoder

from .colony import Colony, load_starting_gains
from training.run import CHECKPOINT_DIR

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "stage1_clean_escape.npy"
QUIT_WORDS = {"quit", "exit"}
UNBOUNDED_TICKS = 10**9  # a live session ends by extinction or user quit, not a tick cap


def read_requests(input_queue: "queue.Queue[tuple[str, str] | None]") -> None:
    """Runs in a background thread: blocks on stdin, pushes each non-empty
    line into the queue as (text, owner). There's only one local input
    stream today, so every line is tagged DEFAULT_OWNER (decisions.md
    #34) -- the plumbing supports a different owner per request, a
    second real input source just doesn't exist yet. Puts a `None`
    sentinel on EOF or a quit word, then returns. The only thing this
    thread touches is the queue -- colony/environment state is mutated
    exclusively by the main thread.
    """
    try:
        while True:
            line = input().strip()
            if not line:
                continue
            if line.lower() in QUIT_WORDS:
                input_queue.put(None)
                return
            input_queue.put((line, DEFAULT_OWNER))
    except EOFError:
        input_queue.put(None)


def drain(input_queue: "queue.Queue[tuple[str, str] | None]") -> tuple[list[tuple[str, str]], bool]:
    """Non-blocking: pulls everything currently queued. Returns (pending
    (request, owner) pairs in order, quit_requested).
    """
    pending: list[tuple[str, str]] = []
    quit_requested = False
    while True:
        try:
            item = input_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            quit_requested = True
            break
        pending.append(item)
    return pending, quit_requested


def apply_requests(
    requests: list[tuple[str, str]], controller: WorldController, actions: list[ActionSpec]
) -> list[tuple[str, str | None]]:
    """Pure logic, no I/O -- testable without a real stdin/thread. Each
    request is (text, owner) -- owner is who's paying and who "own"
    resolves to in a `target` field, supplied by the caller, never by
    the translating controller (decisions.md #34). Returns (request
    text, status) per request: the action name on success,
    f"{name} (rejected)" if a create_* action understood the request
    but a type cap or insufficient energy stopped it (decisions.md #33),
    or None if nothing in the registry matched at all. `arguments`
    (decisions.md #27, #30-#32) is passed through to the action's fn as
    **kwargs only when that action declares an argument_schema; `owner`
    is passed alongside them for those same actions, never for the
    zero-argument rate nudges, which stay owner-agnostic.
    """
    by_name = {action.name: action for action in actions}
    results: list[tuple[str, str | None]] = []
    for request_text, owner in requests:
        outcome = controller.choose_action(request_text, actions)
        action_name = outcome[0] if outcome is not None else None
        action = by_name.get(action_name) if action_name is not None else None
        if action is None:
            action_name = None
        elif action.argument_schema is not None:
            if action.fn(**outcome[1], owner=owner) is False:
                action_name = f"{action_name} (rejected)"
        else:
            action.fn()
        results.append((request_text, action_name))
    return results


def run_live(
    colony: Colony,
    controller: WorldController,
    input_queue: "queue.Queue[tuple[str, str] | None]",
    ticks_per_second: float,
    audit_every: int = 50,
) -> None:
    actions = build_registry(colony.env)
    tick_interval = 1.0 / ticks_per_second
    logger.info("Live loop running -- type a request and press Enter (type 'quit' to stop).")

    while True:
        pending, quit_requested = drain(input_queue)
        for request, action_name in apply_requests(pending, controller, actions):
            if action_name:
                logger.info("request=%r -> %s", request, action_name)
            else:
                logger.info("request=%r -> no matching action", request)

        result = colony.step()
        if result.deaths or result.births:
            logger.info(
                "tick %4d  population=%3d  deaths=%s  births=%d",
                colony.env.tick, colony.population, list(result.deaths.values()), len(result.births),
            )
        if colony.env.tick % audit_every == 0 and colony.population:
            summary = colony.plasticity_summary()
            logger.info(
                "tick %4d  plasticity audit: mean gain drift=%.3f, reinforcement events=%d",
                colony.env.tick, summary["mean_gain_drift"], summary["total_reinforcement_events"],
            )
        if all(result.colony_extinct.values()):
            logger.info("Colony extinct at tick %d", colony.env.tick)
            return
        if quit_requested:
            logger.info("Stopped by user at tick %d (population=%d)", colony.env.tick, colony.population)
            return

        time.sleep(tick_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--initial-population", type=int, default=3)
    parser.add_argument("--max-population", type=int, default=20)
    parser.add_argument("--mutation-sigma", type=float, default=0.1)
    parser.add_argument("--ticks-per-second", type=float, default=5.0)
    parser.add_argument("--controller", choices=["rule_based", "claude"], default="rule_based")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_controller(name: str) -> WorldController:
    if name == "claude":
        return ClaudeController()
    return RuleBasedController()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    encoder = HashingItemEncoder()
    escape_template = build_escape_template(encoder)
    plasticity_template = build_plasticity_template(encoder)
    gains = load_starting_gains(escape_template, args.checkpoint)

    env = Environment(
        initial_population=args.initial_population,
        max_population=args.max_population,
        max_ticks=UNBOUNDED_TICKS,
        encoder=encoder,
        seed=args.seed,
    )
    colony = Colony(
        env, escape_template, plasticity_template, gains, mutation_sigma=args.mutation_sigma, seed=args.seed,
    )
    controller = build_controller(args.controller)

    input_queue: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
    reader = threading.Thread(target=read_requests, args=(input_queue,), daemon=True)
    reader.start()

    logger.info("Starting colony: population=%d, max_population=%d", colony.population, args.max_population)
    run_live(colony, controller, input_queue, args.ticks_per_second)


if __name__ == "__main__":
    main()
