"""Wires real per-fly circuits/genomes to the reproduction mechanic in
world.Environment: each living fly gets its own EscapeAgent; offspring
inherit their parent's synaptic_gain plus mutation -- the same
sigma-perturbation idea ES uses, just triggered live by a birth event
instead of a training-loop batch (see wiki/decisions.md #13, #20).
"""

from __future__ import annotations

import logging

import numpy as np

from fly_brain.agent import EscapeAgent, EscapeCircuitTemplate
from world.env import Action, ColonyStepResult, Environment

logger = logging.getLogger(__name__)


class Colony:
    def __init__(
        self,
        env: Environment,
        template: EscapeCircuitTemplate,
        initial_gains: np.ndarray,
        mutation_sigma: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.template = template
        self.mutation_sigma = mutation_sigma
        self.rng = np.random.default_rng(seed)

        self.observations = env.reset()
        self.agents: dict[int, EscapeAgent] = {
            fly_id: self.spawn_agent(initial_gains) for fly_id in self.observations
        }

    @property
    def population(self) -> int:
        return len(self.agents)

    def spawn_agent(self, gains: np.ndarray) -> EscapeAgent:
        agent = EscapeAgent(self.template)
        agent.set_params(gains.copy())
        agent.reset()
        return agent

    def step(self) -> ColonyStepResult:
        actions: dict[int, Action] = {
            fly_id: self.agents[fly_id].act(obs) for fly_id, obs in self.observations.items()
        }
        result = self.env.step(actions)

        for new_id, parent_id in result.births.items():
            parent_gains = self.agents[parent_id].get_params()
            mutation = self.rng.normal(0, self.mutation_sigma, size=parent_gains.shape)
            self.agents[new_id] = self.spawn_agent(parent_gains + mutation)

        for dead_id in result.deaths:
            del self.agents[dead_id]

        self.observations = result.observations
        return result


def load_starting_gains(template: EscapeCircuitTemplate, checkpoint_path=None) -> np.ndarray:
    """A trained checkpoint gives the colony a strong starting gene pool
    (see wiki/decisions.md #13 for why ES-then-reproduction, not one or
    the other); falls back to untrained (real biology, gain=1.0) if none
    is given or found.
    """
    if checkpoint_path is not None and checkpoint_path.exists():
        logger.info("Loaded starting genome from %s", checkpoint_path)
        return np.load(checkpoint_path)
    logger.info("No checkpoint given/found -- starting from untrained (real biology, gain=1.0)")
    return np.ones(len(template.blueprint.edges))


def run_colony(colony: Colony, max_ticks: int) -> list[int]:
    """Runs until extinction or max_ticks. Returns population size after
    every tick, for plotting/inspection.
    """
    population_history = [colony.population]
    for _ in range(max_ticks):
        result = colony.step()
        population_history.append(colony.population)
        if result.deaths or result.births:
            logger.info(
                "tick %4d  population=%3d  deaths=%s  births=%d",
                colony.env.tick, colony.population, list(result.deaths.values()), len(result.births),
            )
        if result.colony_extinct:
            logger.info("Colony extinct at tick %d", colony.env.tick)
            break
    return population_history
