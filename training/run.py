"""CLI entry point: python -m training.run --stage 1"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

from fly_brain.agent import EscapeAgent, build_escape_template
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
    return parser.parse_args()


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
    agent = EscapeAgent(build_escape_template(encoder))

    baseline = np.mean([rollout(agent, env) for _ in range(BASELINE_EPISODES)])
    logger.info("Untrained (real biology, gain=1.0) baseline fitness: %.1f", baseline)

    history = train_es(agent, env, build_es_config(args))

    checkpoint_path = save_checkpoint(stage.name, agent.get_params())
    logger.info("Saved trained params to %s", checkpoint_path)
    logger.info("Fitness: baseline=%.1f -> final=%.1f", baseline, history[-1])


if __name__ == "__main__":
    main()
