"""CLI entry point: python -m training.run --stage 1"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

from fly_brain.agent import EscapeAgent, build_escape_template, load_starting_gains
from world.env import Environment
from world.items import HashingItemEncoder

from .curriculum import STAGES
from .trainer import ESConfig, rollout, train_es

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent.parent / "training" / "checkpoints"
BASELINE_EPISODES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--episodes-per-eval", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-checkpoint",
        type=pathlib.Path,
        default=None,
        help="Start from this checkpoint instead of the automatic default -- "
        "stage N chains from stage N-1's saved params if found (curriculum "
        "learning, decisions.md #4), or starts untrained for stage 1 / if "
        "nothing's found. Pass an empty path override to force untrained.",
    )
    return parser.parse_args()


def default_init_checkpoint(stage_number: int) -> pathlib.Path | None:
    """Chains curriculum stages: stage N starts from stage N-1's saved
    checkpoint if one exists, so "harder" (stage 2) or "different but
    related" (stage 3) training continues from what the easier stage
    already learned, rather than each stage training from scratch
    (decisions.md #4). None for stage 1, or if the prior stage was never
    trained -- load_starting_gains() already falls back to untrained
    gracefully in that case.
    """
    if stage_number - 1 not in STAGES:
        return None
    return CHECKPOINT_DIR / f"{STAGES[stage_number - 1].name}.npy"


def build_es_config(args: argparse.Namespace) -> ESConfig:
    return ESConfig(
        iterations=args.iterations,
        population=args.population,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        episodes_per_eval=args.episodes_per_eval,
        seed=args.seed,
    )


def save_checkpoint(stage_name: str, params: np.ndarray) -> pathlib.Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{stage_name}.npy"
    np.save(path, params)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    stage = STAGES[args.stage]
    logger.info("Stage: %s  (%s)", stage.name, stage.env_kwargs)

    encoder = HashingItemEncoder()
    env = Environment(seed=args.seed, encoder=encoder, **stage.env_kwargs)
    template = build_escape_template(encoder)
    agent = EscapeAgent(template)

    init_checkpoint = args.init_checkpoint or default_init_checkpoint(args.stage)
    agent.set_params(load_starting_gains(template, init_checkpoint))

    baseline = np.mean([rollout(agent, env) for _ in range(BASELINE_EPISODES)])
    logger.info("Starting-genome baseline fitness (%s): %.1f", init_checkpoint or "untrained", baseline)

    history = train_es(agent, env, build_es_config(args))

    checkpoint_path = save_checkpoint(stage.name, agent.get_params())
    logger.info("Saved trained params to %s", checkpoint_path)
    logger.info("Fitness: baseline=%.1f -> final=%.1f", baseline, history[-1])


if __name__ == "__main__":
    main()
