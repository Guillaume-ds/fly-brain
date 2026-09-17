"""CLI entry point: python -m training.colony_run"""

from __future__ import annotations

import argparse
import logging
import pathlib

from fly_brain.agent import build_escape_template
from world.env import Environment

from .colony import Colony, load_starting_gains, run_colony
from .run import CHECKPOINT_DIR

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "stage1_clean_escape.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--initial-population", type=int, default=3)
    parser.add_argument("--max-population", type=int, default=20)
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--mutation-sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    template = build_escape_template()
    gains = load_starting_gains(template, args.checkpoint)

    env = Environment(
        initial_population=args.initial_population,
        max_population=args.max_population,
        max_ticks=args.max_ticks,
        seed=args.seed,
    )
    colony = Colony(env, template, gains, mutation_sigma=args.mutation_sigma, seed=args.seed)

    logger.info("Starting colony: population=%d, max_population=%d", colony.population, args.max_population)
    history = run_colony(colony, args.max_ticks)
    logger.info("Final population: %d after %d ticks (peak: %d)", history[-1], len(history) - 1, max(history))


if __name__ == "__main__":
    main()
