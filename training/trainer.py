"""Evolution Strategies training loop (see wiki/decisions.md #6 for why ES
before REINFORCE).

Implements the standard OpenAI-ES update (Salimans et al., 2017):

    theta <- theta + (lr / (population * sigma)) * sum_i(advantage_i * noise_i)

Rewards are standardized to zero mean / unit std across the population
each iteration ("advantage" below) -- otherwise the update's scale depends
entirely on the environment's raw reward magnitude, making the learning
rate impossible to tune sensibly.

Only EscapeAgent.get_params()/set_params() (the per-synapse synaptic_gain
vector) is ever touched -- real topology and synapse sign are never
modified, see wiki/decisions.md #7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from fly_brain.agent import EscapeAgent
from world.env import Environment

logger = logging.getLogger(__name__)


def rollout(agent: EscapeAgent, env: Environment) -> float:
    """Curriculum training always runs a population of 1 (reproduction is
    structurally impossible there -- mate_availability is 0 at population
    1, see wiki/decisions.md #20), so there's always exactly one fly to
    drive each tick.
    """
    observations = env.reset()
    agent.reset()
    ticks_survived = 0.0
    while observations:
        actions = {fly_id: agent.act(obs) for fly_id, obs in observations.items()}
        result = env.step(actions)
        ticks_survived += 1.0
        observations = result.observations
        if all(result.colony_extinct.values()) or result.timed_out:
            break
    return ticks_survived


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


@dataclass
class ESIterationResult:
    theta: np.ndarray
    fitness: float
    population_reward_std: float


def run_es_iteration(
    agent: EscapeAgent, env: Environment, config: ESConfig, theta: np.ndarray, rng: np.random.Generator
) -> ESIterationResult:
    noise = rng.standard_normal((config.population, len(theta)))
    rewards = np.empty(config.population)
    for i in range(config.population):
        agent.set_params(theta + config.sigma * noise[i])
        rewards[i] = evaluate(agent, env, config.episodes_per_eval)

    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    updated_theta = theta + (config.learning_rate / (config.population * config.sigma)) * (noise.T @ advantage)
    agent.set_params(updated_theta)
    fitness = evaluate(agent, env, config.episodes_per_eval)

    return ESIterationResult(theta=updated_theta, fitness=fitness, population_reward_std=float(rewards.std()))


def train_es(agent: EscapeAgent, env: Environment, config: ESConfig) -> list[float]:
    """Mutates agent's params in place. Returns fitness (mean survival
    ticks) after each iteration, for plotting/inspection.
    """
    rng = np.random.default_rng(config.seed)
    theta = agent.get_params()
    history: list[float] = []

    for iteration in range(config.iterations):
        result = run_es_iteration(agent, env, config, theta, rng)
        theta = result.theta
        history.append(result.fitness)
        logger.info(
            "iter %3d  fitness=%6.1f  population_reward_std=%5.2f",
            iteration, result.fitness, result.population_reward_std,
        )

    return history
