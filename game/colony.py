"""Wires real per-fly circuits/genomes to the reproduction mechanic in
world.Environment: each living fly gets its own EscapeAgent (frozen,
ES-trained) and its own PlasticityAgent (live, dopamine-gated lifetime
learning -- see fly_brain/plasticity.py and decisions.md #22/#26).
Escape offspring inherit their parent's synaptic_gain plus mutation, the
same sigma-perturbation idea ES uses, just triggered live by a birth
event instead of a training-loop batch (see wiki/decisions.md #13, #20).
Plasticity offspring inherit the parent's INHERITED PRIOR plus mutation
-- never whatever gains the parent's own synapses drifted to during its
life, per decisions.md #22 part 3: learned associations aren't
inherited, only the capacity to learn them is.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

from fly_brain.agent import EscapeAgent, EscapeCircuitTemplate
from fly_brain.plasticity import PlasticityAgent, PlasticityCircuitTemplate
from world.entities import Fly
from world.env import Action, ColonyStepResult, Environment
from world.results import Channel

logger = logging.getLogger(__name__)


class Colony:
    def __init__(
        self,
        env: Environment,
        escape_template: EscapeCircuitTemplate,
        plasticity_template: PlasticityCircuitTemplate,
        initial_escape_gains: np.ndarray,
        mutation_sigma: float = 0.1,
        plasticity_mutation_sigma: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.escape_template = escape_template
        self.plasticity_template = plasticity_template
        self.mutation_sigma = mutation_sigma
        self.plasticity_mutation_sigma = plasticity_mutation_sigma
        self.rng = np.random.default_rng(seed)

        self.observations = env.reset()
        self.escape_agents: dict[int, EscapeAgent] = {}
        self.plasticity_agents: dict[int, PlasticityAgent] = {}
        for fly_id in self.observations:
            self.escape_agents[fly_id] = self.spawn_escape_agent(initial_escape_gains)
            self.plasticity_agents[fly_id] = self.spawn_plasticity_agent(None)

    @property
    def population(self) -> int:
        return len(self.escape_agents)

    def spawn_escape_agent(self, gains: np.ndarray) -> EscapeAgent:
        agent = EscapeAgent(self.escape_template)
        agent.set_params(gains.copy())
        agent.reset()
        return agent

    def spawn_plasticity_agent(self, prior_gains: np.ndarray | None) -> PlasticityAgent:
        return PlasticityAgent(self.plasticity_template, initial_gains=prior_gains)

    def state_of(self, fly: Fly) -> dict[Channel, int]:
        """A fly's current value on every channel a Result can move --
        read straight off the enum so this can't drift from the world's
        own definition (decisions.md #28).
        """
        return {channel: getattr(fly, channel.value) for channel in Channel}

    def snapshot_states(self) -> dict[int, dict[Channel, int]]:
        return {fly.id: self.state_of(fly) for fly in self.env.flies}

    def step(self) -> ColonyStepResult:
        before = self.snapshot_states()

        actions: dict[int, Action] = {}
        for fly_id, obs in self.observations.items():
            plasticity_action = self.plasticity_agents[fly_id].sense_and_decide(obs)
            escape_action = self.escape_agents[fly_id].decide(obs)
            actions[fly_id] = escape_action if escape_action is not None else plasticity_action

        result = self.env.step(actions)

        for fly in self.env.flies:
            if fly.id not in before:
                continue  # born this tick -- no prior state to diff against yet
            deltas = {channel: after - before[fly.id][channel] for channel, after in self.state_of(fly).items()}
            self.plasticity_agents[fly.id].reinforce(deltas)

        for new_id, parent_id in result.births.items():
            parent_escape_gains = self.escape_agents[parent_id].get_params()
            escape_mutation = self.rng.normal(0, self.mutation_sigma, size=parent_escape_gains.shape)
            self.escape_agents[new_id] = self.spawn_escape_agent(parent_escape_gains + escape_mutation)

            parent_prior = self.plasticity_agents[parent_id].prior_gains
            plasticity_mutation = self.rng.normal(0, self.plasticity_mutation_sigma, size=parent_prior.shape)
            self.plasticity_agents[new_id] = self.spawn_plasticity_agent(parent_prior + plasticity_mutation)

        for dead_id in result.deaths:
            del self.escape_agents[dead_id]
            del self.plasticity_agents[dead_id]

        self.observations = result.observations
        return result

    def plasticity_summary(self) -> dict[str, float]:
        """Colony-level audit snapshot (decisions.md #26): mean gain
        drift and total reinforcement events across every living fly --
        a cheap way to see whether the population is learning at all
        without inspecting individual agents.
        """
        if not self.plasticity_agents:
            return {"mean_gain_drift": 0.0, "total_reinforcement_events": 0}
        drifts = [agent.gain_drift() for agent in self.plasticity_agents.values()]
        events = sum(agent.reinforcement_events for agent in self.plasticity_agents.values())
        return {"mean_gain_drift": float(np.mean(drifts)), "total_reinforcement_events": events}


def load_starting_gains(
    template: EscapeCircuitTemplate, checkpoint_path: pathlib.Path | None = None
) -> np.ndarray:
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


def run_colony(colony: Colony, max_ticks: int, audit_every: int = 50) -> list[int]:
    """Runs until extinction or max_ticks. Returns population size after
    every tick, for plotting/inspection. Logs a plasticity audit snapshot
    (decisions.md #26) every `audit_every` ticks, alongside
    population/births/deaths -- a running view of whether the live
    population is actually learning, not just an end-of-run number that
    disappears the moment the colony goes extinct.
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
        if colony.env.tick % audit_every == 0 and colony.population:
            summary = colony.plasticity_summary()
            logger.info(
                "tick %4d  plasticity audit: mean gain drift=%.3f, reinforcement events=%d",
                colony.env.tick, summary["mean_gain_drift"], summary["total_reinforcement_events"],
            )
        if result.colony_extinct:
            logger.info("Colony extinct at tick %d", colony.env.tick)
            break
    return population_history
