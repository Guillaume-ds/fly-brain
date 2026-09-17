"""Evolution Strategies training loop (see wiki/decisions.md #6 for why ES
before REINFORCE).

Implements the standard OpenAI-ES update (Salimans et al., 2017):

    theta <- theta + (lr / (population * sigma)) * sum_i(advantage_i * noise_i)

Rewards are standardized to zero mean / unit std across the population
each iteration before this update ("advantage" below) -- without that, the
update's scale depends entirely on the environment's raw reward magnitude
(here, up to `max_ticks`), which makes the learning rate impossible to
tune sensibly.

Only EscapeAgent.get_params()/set_params() (the per-synapse
synaptic_gain vector) is ever touched -- real topology and real synapse
sign are never modified, see wiki/decisions.md #7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_brain.agent import EscapeAgent
from world.env import Environment


def rollout(agent: EscapeAgent, env: Environment) -> float:
    obs = env.reset()
    agent.reset()
    total_reward = 0.0
    done = False
    while not done:
        result = env.step(agent.act(obs))
        obs = result.observation
        total_reward += result.reward
        done = result.done
    return total_reward


def evaluate(agent: EscapeAgent, env: Environment, episodes: int) -> float:
    return float(np.mean([rollout(agent, env) for _ in range(episodes)]))


@dataclass
class ESConfig:
    iterations: int = 50
    population: int = 32
    sigma: float = 0.1
    learning_rate: float = 0.05
    episodes_per_eval: int = 3
    seed: int | None = None


def train_es(agent: EscapeAgent, env: Environment, config: ESConfig) -> list[float]:
    """Mutates agent's params in place. Returns fitness (mean survival
    ticks) after each iteration, for plotting/inspection.
    """
    rng = np.random.default_rng(config.seed)
    theta = agent.get_params()
    history: list[float] = []

    for iteration in range(config.iterations):
        noise = rng.standard_normal((config.population, len(theta)))
        rewards = np.empty(config.population)
        for i in range(config.population):
            agent.set_params(theta + config.sigma * noise[i])
            rewards[i] = evaluate(agent, env, config.episodes_per_eval)

        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        theta = theta + (config.learning_rate / (config.population * config.sigma)) * (
            noise.T @ advantage
        )
        agent.set_params(theta)

        fitness = evaluate(agent, env, config.episodes_per_eval)
        history.append(fitness)
        print(f"iter {iteration:3d}  fitness={fitness:6.1f}  population_reward_std={rewards.std():5.2f}")

    return history
