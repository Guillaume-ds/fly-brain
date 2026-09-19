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

Each fly also gets a WanderAgent (fly_brain/wander.py, decisions.md
#39/#40): a fixed fallback movement reflex, fired only when a fly
perceives nothing at all, inherited and mutated at birth the same way
escape gains are -- fixed for a fly's whole lifetime, never touched by
reinforce().

`brain_snapshot()` (decisions.md #46) assembles a per-fly diagnostic
bundle -- which of the three tiers produced its last action, and the
real numbers behind that decision -- entirely from state these agents
already keep for auditing (decisions.md #26); nothing new is computed
except on-demand `probe()` calls for whichever fly is actually being
inspected. `brain_topology()`/`neuron_state()` (decisions.md #47) extend
this to the real neuron-graph itself -- the actual connectome subgraph
each circuit runs on, plus live per-neuron voltage/spike state for a
watched fly, straight from `Circuit`'s own public `v`/`spikes_prev`.
"""

from __future__ import annotations

import logging

import numpy as np

from fly_brain.agent import EscapeAgent, EscapeCircuitTemplate
from fly_brain.circuit import CircuitBlueprint
from fly_brain.plasticity import PlasticityAgent, PlasticityCircuitTemplate
from fly_brain.wander import DEFAULT_WANDER_PERSISTENCE, WanderAgent
from world.env import Action, ColonyStepResult, Environment

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
        wander_mutation_sigma: float = 1.0,
        initial_wander_persistence: float = DEFAULT_WANDER_PERSISTENCE,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.escape_template = escape_template
        self.plasticity_template = plasticity_template
        self.mutation_sigma = mutation_sigma
        self.plasticity_mutation_sigma = plasticity_mutation_sigma
        self.wander_mutation_sigma = wander_mutation_sigma
        self.rng = np.random.default_rng(seed)

        self.observations = env.reset()
        self.escape_agents: dict[int, EscapeAgent] = {}
        self.plasticity_agents: dict[int, PlasticityAgent] = {}
        self.wander_agents: dict[int, WanderAgent] = {}
        # decisions.md #46: which tier (escape/wander/plasticity) produced
        # each fly's most recent action -- set every tick in step(), read by
        # brain_snapshot(). A fly has no entry until its first real decision
        # tick (never true for initial population, briefly true for a
        # newborn between being added to env.flies and its first step()).
        self.last_action_source: dict[int, str] = {}
        for fly_id in self.observations:
            self.escape_agents[fly_id] = self.spawn_escape_agent(initial_escape_gains)
            self.plasticity_agents[fly_id] = self.spawn_plasticity_agent(None)
            self.wander_agents[fly_id] = self.spawn_wander_agent(initial_wander_persistence)

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

    def spawn_wander_agent(self, persistence: float) -> WanderAgent:
        return WanderAgent(persistence)

    def add_colony(
        self,
        owner: str,
        population: int,
        initial_escape_gains: np.ndarray,
        initial_wander_persistence: float = DEFAULT_WANDER_PERSISTENCE,
    ) -> list[int]:
        """Adds a second (or Nth) colony to an already-running Colony
        wrapper -- mirrors env.spawn_colony() (decisions.md #34) but
        also builds each new fly's brain agents, exactly as __init__
        does for the first one. Refreshes `observations` so the next
        step() call includes them. Returns the new fly ids.
        """
        new_flies = self.env.spawn_colony(owner, population)
        for fly in new_flies:
            self.escape_agents[fly.id] = self.spawn_escape_agent(initial_escape_gains)
            self.plasticity_agents[fly.id] = self.spawn_plasticity_agent(None)
            self.wander_agents[fly.id] = self.spawn_wander_agent(initial_wander_persistence)
        self.observations = {**self.observations, **{fly.id: self.env.observe(fly) for fly in new_flies}}
        return [fly.id for fly in new_flies]

    def step(self) -> ColonyStepResult:
        actions: dict[int, Action] = {}
        for fly_id, obs in self.observations.items():
            plasticity_action = self.plasticity_agents[fly_id].sense_and_decide(obs)
            escape_action = self.escape_agents[fly_id].decide(obs)
            # wander_action is None whenever obs.nearby is non-empty -- i.e.
            # exactly the ticks where plasticity_action would have been a
            # real reactive decision, never STAY-because-blind. Only fires
            # as a fallback under escape, same freeze+override precedence
            # as always (decisions.md #5/#22): the reflex still wins.
            wander_action = self.wander_agents[fly_id].decide(obs, self.rng)
            if escape_action is not None:
                actions[fly_id] = escape_action
                self.last_action_source[fly_id] = "escape"
            elif wander_action is not None:
                actions[fly_id] = wander_action
                self.last_action_source[fly_id] = "wander"
            else:
                actions[fly_id] = plasticity_action
                self.last_action_source[fly_id] = "plasticity"

        result = self.env.step(actions)

        for fly in self.env.flies:
            if fly.id not in self.plasticity_agents:
                continue  # born this tick -- its agents are built in the births loop below
            # result.effects (decisions.md #37) is EFFECT deltas only --
            # items/tiles/mobs/fly-combat/corpse-pickup -- never the
            # constant hunger decay, and never a before/after diff of net
            # state. A fly untouched by anything this tick simply has no
            # entry; reinforce()'s own early-return handles that the same
            # way it always has.
            self.plasticity_agents[fly.id].reinforce(result.effects.get(fly.id, {}))

        for new_id, parent_id in result.births.items():
            parent_escape_gains = self.escape_agents[parent_id].get_params()
            escape_mutation = self.rng.normal(0, self.mutation_sigma, size=parent_escape_gains.shape)
            self.escape_agents[new_id] = self.spawn_escape_agent(parent_escape_gains + escape_mutation)

            parent_prior = self.plasticity_agents[parent_id].prior_gains
            plasticity_mutation = self.rng.normal(0, self.plasticity_mutation_sigma, size=parent_prior.shape)
            self.plasticity_agents[new_id] = self.spawn_plasticity_agent(parent_prior + plasticity_mutation)

            parent_persistence = self.wander_agents[parent_id].persistence
            wander_mutation = self.rng.normal(0, self.wander_mutation_sigma)
            self.wander_agents[new_id] = self.spawn_wander_agent(parent_persistence + wander_mutation)

        for dead_id in result.deaths:
            del self.escape_agents[dead_id]
            del self.plasticity_agents[dead_id]
            del self.wander_agents[dead_id]
            self.last_action_source.pop(dead_id, None)

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

    def brain_snapshot(self, fly_id: int) -> dict | None:
        """Per-fly diagnostic bundle for a "brain inspector" (decisions.md
        #46): which tier produced its last action and the real numbers
        behind that decision, assembled entirely from state these agents
        already keep for auditing (decisions.md #26) -- `last_danger_
        strength`/`last_valence` are cached from the tick that just ran,
        `gain_drift()`/`reinforcement_events`/`cumulative_dopamine` are
        already-existing hooks. The only genuinely new computation is the
        two `probe()` calls, deliberately run only for whichever single
        fly is actually being inspected, not for the whole population
        every tick -- `probe()` steps a disposable circuit copy twice, real
        but modest cost that isn't worth paying for flies nobody is
        looking at.

        `None` if `fly_id` doesn't exist (dead, or never did) -- the
        caller's cue to stop watching it.
        """
        if fly_id not in self.plasticity_agents:
            return None
        escape_agent = self.escape_agents[fly_id]
        plasticity_agent = self.plasticity_agents[fly_id]
        wander_agent = self.wander_agents[fly_id]
        return {
            "fly_id": fly_id,
            "action_source": self.last_action_source.get(fly_id),
            "escape": {
                "danger_strength": escape_agent.last_danger_strength,
            },
            "plasticity": {
                "valence": plasticity_agent.last_valence,
                "gain_drift": plasticity_agent.gain_drift(),
                "reinforcement_events": plasticity_agent.reinforcement_events,
                "cumulative_dopamine": plasticity_agent.cumulative_dopamine,
                "probe_food": float(plasticity_agent.probe(self.env.food_type.attributes)),
                "probe_danger": float(plasticity_agent.probe(self.escape_template.danger_vector)),
            },
            "wander": {
                "persistence": wander_agent.persistence,
                "current_direction": (
                    wander_agent.current_direction.name if wander_agent.current_direction is not None else None
                ),
            },
        }

    def brain_topology(self) -> dict:
        """Static connectome topology for the brain inspector's Tier 3
        neuron-graph view (decisions.md #47): real MaleCNS body ids and
        synapse weights straight from each template's own public
        `blueprint` (`decisions.md` #46 already established these as
        public, non-circuit-internal data). Identical for every fly of a
        given kind -- only `synaptic_gain` differs per fly, never
        topology -- so this is colony-wide and worth computing once per
        connection, not once per tick per watched fly.
        """
        return {
            "escape": _serialize_topology(
                self.escape_template.blueprint,
                {"seed": self.escape_template.seed_idx, "motor": self.escape_template.motor_idx},
            ),
            "plasticity": _serialize_topology(
                self.plasticity_template.blueprint,
                {
                    "kc": self.plasticity_template.kc_idx,
                    "mbon_approach": self.plasticity_template.mbon_approach_idx,
                    "mbon_avoid": self.plasticity_template.mbon_avoid_idx,
                    "pam": self.plasticity_template.pam_idx,
                    "ppl": self.plasticity_template.ppl_idx,
                },
            ),
        }

    def neuron_state(self, fly_id: int) -> dict | None:
        """Live per-neuron membrane potential + spike state for a
        watched fly's two circuits (decisions.md #47) -- `Circuit.v`/
        `spikes_prev` are already stepped every tick regardless of
        whether anyone's watching; this just reads them, for whichever
        one fly is actually being inspected. Index-aligned with
        `brain_topology()`'s `neuron_ids` ordering for the same circuit.
        Same `None`-on-dead-or-nonexistent contract as `brain_snapshot()`.
        """
        if fly_id not in self.plasticity_agents:
            return None
        escape_circuit = self.escape_agents[fly_id].circuit
        plasticity_circuit = self.plasticity_agents[fly_id].circuit
        return {
            "fly_id": fly_id,
            "escape": {
                "v": escape_circuit.v.tolist(),
                "spikes": escape_circuit.spikes_prev.tolist(),
            },
            "plasticity": {
                "v": plasticity_circuit.v.tolist(),
                "spikes": plasticity_circuit.spikes_prev.tolist(),
            },
        }


def _serialize_topology(blueprint: CircuitBlueprint, groups: dict[str, list[int]]) -> dict:
    index_of = {body_id: i for i, body_id in enumerate(blueprint.neuron_ids)}
    return {
        "neuron_ids": blueprint.neuron_ids,
        "edges": [
            {"pre": index_of[edge.pre_body_id], "post": index_of[edge.post_body_id], "weight": edge.weight}
            for edge in blueprint.edges
        ],
        "groups": groups,
    }


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
        if all(result.colony_extinct.values()):
            logger.info("Colony extinct at tick %d", colony.env.tick)
            break
    return population_history
