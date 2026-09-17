"""CLI entry point: python -m training.run --stage 1"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from fly_brain.agent import EscapeAgent
from world.env import Environment

from .curriculum import STAGES
from .trainer import ESConfig, rollout, train_es

CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent.parent / "training" / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--episodes-per-eval", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stage = STAGES[args.stage]
    print(f"Stage: {stage.name}  ({stage.env_kwargs})")

    env = Environment(seed=args.seed, **stage.env_kwargs)
    agent = EscapeAgent()

    baseline = np.mean([rollout(agent, env) for _ in range(10)])
    print(f"Untrained (real biology, gain=1.0) baseline fitness: {baseline:.1f}")

    config = ESConfig(
        iterations=args.iterations,
        population=args.population,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        episodes_per_eval=args.episodes_per_eval,
        seed=args.seed,
    )
    history = train_es(agent, env, config)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CHECKPOINT_DIR / f"{stage.name}.npy", agent.get_params())
    print(f"Saved trained params to {CHECKPOINT_DIR / f'{stage.name}.npy'}")
    print(f"Fitness: baseline={baseline:.1f} -> final={history[-1]:.1f}")


if __name__ == "__main__":
    main()
