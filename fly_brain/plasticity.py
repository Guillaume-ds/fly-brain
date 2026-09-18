"""PlasticityAgent: the second, real circuit from decisions.md #22 --
Kenyon Cells (KC) -> Mushroom Body Output Neurons (MBON), gated by
Dopaminergic Neurons (DAN), implementing lifetime, dopamine-gated
Hebbian plasticity. Kept fully separate from the frozen, ES-trained
escape circuit (agent.py) -- combined via the same freeze+override
pattern as decisions.md #5: the escape circuit's flee decision, when it
has one, always wins (see training/colony.py).

Real neuron population sizes, verified against the actual MaleCNS data
(decisions.md #22): 4,064 KCs, 97 MBONs, 354 DANs (PAM=reward-coding,
PPL/PPM=punishment-coding, per the literature). Building this circuit at
full scale is computationally impractical for a per-tick, per-fly colony
simulation, so it's built from a bounded, real subgraph (decisions.md
#26): the top-N KCs by total real KC->MBON synaptic weight (not an
arbitrary subsample -- the KCs whose real output actually matters most
for this circuit's function), their real top-K targets (auto-discovering
the MBON population rather than guessing it), and every DAN with a real
edge into that KC/MBON universe, each DAN's own top-K edges restricted
to that universe (not DAN's globally-broadest targets, which mostly lie
outside it -- DANs broadcast very widely in the real data).

Stimulus injection (percept attributes -> KC current) uses a fixed,
untrained random projection, the same simplification already used for
the escape circuit's direct current injection into DNp01 -- standing in
for the real ~6-random-PN-inputs-per-KC wiring (decisions.md #22 part 2).

MBON approach/avoid split: there is no real downstream-of-MBON
connectivity in this bounded subgraph to derive functional roles from
data, and mapping the dataset's numeric MBON type codes to published
literature roles is out of scope for verification here. This is an
explicit, documented simplification (`split_mbon_valence`) -- a
structural split by sorted body id parity, not a biological claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from world.env import Action, Observation, Percept
from world.items import ItemEncoder

from .circuit import Circuit, CircuitBlueprint, SynapseEdge
from .data import load_connectome_data

REWARD_TYPE_PATTERN = "PAM"
PUNISH_TYPE_PATTERN = "PPL|PPM"

DEFAULT_N_KC = 100  # top KCs by total real KC->MBON weight
DEFAULT_K_KC = 20  # each selected KC's own top-K real outgoing edges
DEFAULT_K_DAN = 30  # each contributing DAN's top-K real edges into the KC/MBON universe

# Reward/punishment channel weights combining Δhunger/Δhealth/Δstuck_ticks
# into one dopamine magnitude -- the "new small authored quantity" from
# decisions.md #22 part 5, tunable, not derived from anything.
CHANNEL_WEIGHTS = {"hunger": 1.0, "health": 3.0, "stuck_ticks": 0.5}

KC_TRACE_DECAY = 0.7  # eligibility trace decay per tick, bridges sense-then-outcome delay
LEARNING_RATE = 0.05
GAIN_BOUNDS = (0.0, 3.0)  # never negative -- a gain flips sign, real synapses don't
VALENCE_DEADZONE = 0.1  # ignore membrane-potential noise below this -- avoid constant twitchy movement
DAN_GAIN_RANGE = (0.01, 1.0)  # per-DAN-neuron fixed random sensitivity, see reinforce()


def build_plasticity_blueprint(
    weights: pd.DataFrame,
    annotations: pd.DataFrame,
    n_kc: int = DEFAULT_N_KC,
    k_kc: int = DEFAULT_K_KC,
    k_dan: int = DEFAULT_K_DAN,
) -> tuple[CircuitBlueprint, set[int], set[int], set[int], set[int]]:
    """Returns (blueprint, kc_ids, mbon_ids, pam_ids, ppl_ids) -- the real
    body ids selected for each functional group, each guaranteed to be a
    subset of blueprint.neuron_ids.
    """
    kc_all = set(annotations.loc[annotations["class"] == "Kenyon_Cell", "bodyId"])
    mbon_all = set(annotations.loc[annotations["type"].astype(str).str.contains("MBON", na=False), "bodyId"])
    pam_all = set(annotations.loc[annotations["type"].astype(str).str.contains(REWARD_TYPE_PATTERN, na=False), "bodyId"])
    ppl_all = set(annotations.loc[annotations["type"].astype(str).str.contains(PUNISH_TYPE_PATTERN, na=False), "bodyId"])
    dan_all = pam_all | ppl_all

    kc_to_mbon = weights[weights["body_pre"].isin(kc_all) & weights["body_post"].isin(mbon_all)]
    top_kc = kc_to_mbon.groupby("body_pre")["weight"].sum().sort_values(ascending=False)
    kc_ids = set(top_kc.head(n_kc).index)
    if len(kc_ids) < n_kc:
        raise ValueError(f"Only {len(kc_ids)} KCs have any real KC->MBON edge -- lower n_kc.")

    kc_edges = weights[weights["body_pre"].isin(kc_ids)]
    kc_top_edges = kc_edges.sort_values("weight", ascending=False).groupby("body_pre").head(k_kc)
    mbon_ids = set(kc_top_edges["body_post"]) & mbon_all
    if not mbon_ids:
        raise ValueError("No real MBON reached from the selected KCs -- raise k_kc.")

    target_universe = kc_ids | mbon_ids
    dan_edges = weights[weights["body_pre"].isin(dan_all) & weights["body_post"].isin(target_universe)]
    dan_top_edges = dan_edges.sort_values("weight", ascending=False).groupby("body_pre").head(k_dan)
    pam_ids = set(dan_top_edges["body_pre"]) & pam_all
    ppl_ids = set(dan_top_edges["body_pre"]) & ppl_all
    if not pam_ids or not ppl_ids:
        raise ValueError("No real PAM or PPL/PPM DAN reached the KC/MBON universe -- raise k_dan.")

    neuron_ids = kc_ids | mbon_ids | pam_ids | ppl_ids
    all_rows = pd.concat([kc_top_edges, dan_top_edges])
    final_rows = all_rows[all_rows["body_pre"].isin(neuron_ids) & all_rows["body_post"].isin(neuron_ids)]
    edges = [SynapseEdge(row.body_pre, row.body_post, row.weight) for row in final_rows.itertuples(index=False)]

    blueprint = CircuitBlueprint(neuron_ids=sorted(neuron_ids), edges=edges)
    return blueprint, kc_ids, mbon_ids, pam_ids, ppl_ids


def split_mbon_valence(mbon_ids: set[int]) -> tuple[list[int], list[int]]:
    """See module docstring -- a documented simplification, not a
    biological claim about which specific MBONs promote approach vs
    avoidance.
    """
    ordered = sorted(mbon_ids)
    return ordered[0::2], ordered[1::2]


@dataclass
class PlasticityCircuitTemplate:
    blueprint: CircuitBlueprint
    neurotransmitters: pd.Series
    kc_idx: list[int]
    mbon_approach_idx: list[int]
    mbon_avoid_idx: list[int]
    pam_idx: list[int]
    ppl_idx: list[int]
    kc_mbon_approach_edge_idx: np.ndarray  # plastic: KC -> approach-coding MBON
    kc_mbon_avoid_edge_idx: np.ndarray  # plastic: KC -> avoid-coding MBON
    projection: np.ndarray  # fixed random (n_kc, attribute_dim); percept attributes -> KC stimulus
    pam_gain: np.ndarray  # fixed, per-neuron random sensitivity -- see reinforce()'s docstring for why
    ppl_gain: np.ndarray


def build_plasticity_template(
    encoder: ItemEncoder,
    n_kc: int = DEFAULT_N_KC,
    k_kc: int = DEFAULT_K_KC,
    k_dan: int = DEFAULT_K_DAN,
    seed: int | None = 0,
) -> PlasticityCircuitTemplate:
    connectome = load_connectome_data()
    blueprint, kc_ids, mbon_ids, pam_ids, ppl_ids = build_plasticity_blueprint(
        connectome.weights, connectome.annotations, n_kc=n_kc, k_kc=k_kc, k_dan=k_dan,
    )
    index_of = {body_id: i for i, body_id in enumerate(blueprint.neuron_ids)}
    approach_ids, avoid_ids = split_mbon_valence(mbon_ids)
    approach_ids, avoid_ids = set(approach_ids), set(avoid_ids)

    kc_mbon_approach_edge_idx = np.array(
        [i for i, e in enumerate(blueprint.edges) if e.pre_body_id in kc_ids and e.post_body_id in approach_ids]
    )
    kc_mbon_avoid_edge_idx = np.array(
        [i for i, e in enumerate(blueprint.edges) if e.pre_body_id in kc_ids and e.post_body_id in avoid_ids]
    )

    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0, size=(len(kc_ids), encoder.output_dim))
    pam_gain = rng.uniform(DAN_GAIN_RANGE[0], DAN_GAIN_RANGE[1], size=len(pam_ids))
    ppl_gain = rng.uniform(DAN_GAIN_RANGE[0], DAN_GAIN_RANGE[1], size=len(ppl_ids))

    return PlasticityCircuitTemplate(
        blueprint=blueprint,
        neurotransmitters=connectome.neurotransmitters,
        kc_idx=[index_of[b] for b in sorted(kc_ids)],
        mbon_approach_idx=[index_of[b] for b in sorted(approach_ids)],
        mbon_avoid_idx=[index_of[b] for b in sorted(avoid_ids)],
        pam_idx=[index_of[b] for b in sorted(pam_ids)],
        ppl_idx=[index_of[b] for b in sorted(ppl_ids)],
        kc_mbon_approach_edge_idx=kc_mbon_approach_edge_idx,
        kc_mbon_avoid_edge_idx=kc_mbon_avoid_edge_idx,
        projection=projection,
        pam_gain=pam_gain,
        ppl_gain=ppl_gain,
    )


class PlasticityAgent:
    def __init__(
        self,
        template: PlasticityCircuitTemplate,
        initial_gains: np.ndarray | None = None,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self.template = template
        self.circuit = Circuit(template.blueprint, template.neurotransmitters)
        if initial_gains is not None:
            self.circuit.set_params(initial_gains)
        self.learning_rate = learning_rate
        self.kc_trace = np.zeros(self.circuit.n)
        # captured AFTER initial_gains is applied -- what THIS fly was born
        # with, whether default (gain=1.0) or inherited+mutated; never
        # updated again, see decisions.md #22 part 3 / #26
        self.prior_gains = self.circuit.get_params()

        # auditing (decisions.md #26): cheap running stats, not a per-tick log
        self.reinforcement_events = 0
        self.cumulative_dopamine = 0.0

    def reset(self) -> None:
        self.circuit.reset()
        self.kc_trace = np.zeros(self.circuit.n)

    def sense_and_decide(self, obs: Observation) -> Action:
        """Always runs, every tick, regardless of whether the escape
        circuit ends up overriding the resulting Action -- sensing and
        the eligibility trace continue even on ticks where the fly
        actually fled instead.

        Net valence reads MBON **membrane potential**, not spikes:
        empirically, a binary spike-count difference is too coarse to
        track gradual synaptic change (confirmed directly -- gain_drift
        climbed steadily tick after tick while the spike-count valence
        stayed frozen at 0 the whole time, only a real-valued membrane
        readout tracked the same learning smoothly, decisions.md #26).
        The circuit's own internal spiking dynamics (KC/DAN, and MBON's
        own spike-or-not state carried into the next tick) are unchanged
        -- only what this readout inspects is different.
        """
        current = np.zeros(self.circuit.n)
        for percept in obs.nearby:
            stimulus = self.template.projection @ percept.attributes
            current[self.template.kc_idx] += stimulus / (1.0 + percept.distance)
        spikes = self.circuit.step(current)

        kc_spike_mask = np.zeros(self.circuit.n)
        kc_spike_mask[self.template.kc_idx] = spikes[self.template.kc_idx].astype(float)
        self.kc_trace = self.kc_trace * KC_TRACE_DECAY + kc_spike_mask

        valence = self._mbon_valence()
        return self._valence_to_action(valence, obs.nearby)

    def _mbon_valence(self) -> float:
        approach = self.circuit.v[self.template.mbon_approach_idx].sum()
        avoid = self.circuit.v[self.template.mbon_avoid_idx].sum()
        return float(approach - avoid)

    def _valence_to_action(self, valence: float, nearby: list[Percept]) -> Action:
        if not nearby or abs(valence) < VALENCE_DEADZONE:
            return Action.STAY
        nearest = min(nearby, key=lambda p: p.distance)
        dx, dy = (nearest.dx, nearest.dy) if valence > 0 else (-nearest.dx, -nearest.dy)
        if abs(dx) >= abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        return Action.DOWN if dy > 0 else Action.UP

    def reinforce(self, deltas: dict[str, float]) -> None:
        """Call once per tick with the REAL state deltas that tick
        produced (world/results.py's blend, diffed by Colony) -- never
        from a percept's similarity score directly. That's the rule that
        keeps this consistent with decisions.md #22 part 2/5: the fly is
        reinforced only by what actually happened to it.

        The textbook mushroom-body rule is depression-only (dopamine
        always weakens whichever KC->MBON synapses were just active);
        that produces a behavioral shift in real flies because of real
        anatomy downstream of MBON that isn't part of this bounded
        subgraph -- here MBON spikes are read directly as approach minus
        avoid (decisions.md #26), so depressing both pathways equally for
        the same active KCs would never move that difference. The rule
        below is the adaptation that actually shifts net valence within
        that simplified readout: reward potentiates the approach pathway
        and depresses the avoid pathway for the KCs that were active;
        punishment does the reverse.

        DAN stimulus uses each neuron's fixed, per-neuron random gain
        (`template.pam_gain`/`ppl_gain`), not a uniform current -- found
        empirically, not assumed: injecting the same current into every
        neuron in a group makes that group's spike response an
        all-or-nothing saturating switch (any reward/punishment above a
        tiny floor makes the *entire* group spike), so `pam_signal -
        ppl_signal` collapsed to ~0 regardless of true magnitude,
        confirmed directly in a live Colony run where a real +47 hunger
        pickup produced zero reinforcement (decisions.md #26). Per-neuron
        gain turns the population into a genuine graded rate code: the
        fraction of a group crossing threshold grows smoothly with
        magnitude instead of jumping straight to 100%.
        """
        reward = max(0.0, deltas.get("hunger", 0.0)) * CHANNEL_WEIGHTS["hunger"]
        punishment = (
            max(0.0, -deltas.get("health", 0.0)) * CHANNEL_WEIGHTS["health"]
            + max(0.0, deltas.get("stuck_ticks", 0.0)) * CHANNEL_WEIGHTS["stuck_ticks"]
        )
        if reward == 0.0 and punishment == 0.0:
            return

        current = np.zeros(self.circuit.n)
        current[self.template.pam_idx] += reward * self.template.pam_gain
        current[self.template.ppl_idx] += punishment * self.template.ppl_gain
        spikes = self.circuit.step(current)

        pam_signal = float(spikes[self.template.pam_idx].mean()) if self.template.pam_idx else 0.0
        ppl_signal = float(spikes[self.template.ppl_idx].mean()) if self.template.ppl_idx else 0.0
        dopamine_signal = pam_signal - ppl_signal
        if dopamine_signal == 0.0:
            return

        approach_idx = self.template.kc_mbon_approach_edge_idx
        avoid_idx = self.template.kc_mbon_avoid_edge_idx
        approach_pre = self.circuit.pre_neuron_of_edge(approach_idx)
        avoid_pre = self.circuit.pre_neuron_of_edge(avoid_idx)

        self.circuit.synaptic_gain[approach_idx] = np.clip(
            self.circuit.synaptic_gain[approach_idx] + self.learning_rate * dopamine_signal * self.kc_trace[approach_pre],
            *GAIN_BOUNDS,
        )
        self.circuit.synaptic_gain[avoid_idx] = np.clip(
            self.circuit.synaptic_gain[avoid_idx] - self.learning_rate * dopamine_signal * self.kc_trace[avoid_pre],
            *GAIN_BOUNDS,
        )

        self.reinforcement_events += 1
        self.cumulative_dopamine += abs(dopamine_signal)

    def probe(self, attributes: np.ndarray) -> float:
        """Read-only diagnostic for auditing (decisions.md #26): net
        valence (approach - avoid MBON membrane potential, see
        `_mbon_valence`) this agent's CURRENT gains would produce for a
        given attribute vector, with zero side effects -- no trace
        update, no gain change, no disturbance to the live circuit's
        membrane state (a disposable circuit copy is stepped instead).
        Call with the same probe vector at intervals through a fly's
        life to see a learning curve.
        """
        probe_circuit = Circuit(self.template.blueprint, self.template.neurotransmitters)
        probe_circuit.set_params(self.circuit.get_params())
        current = np.zeros(probe_circuit.n)
        current[self.template.kc_idx] += self.template.projection @ attributes
        probe_circuit.step(current)  # KCs respond to direct stimulus this step
        probe_circuit.step(np.zeros(probe_circuit.n))  # KC->MBON propagates on the next
        approach = probe_circuit.v[self.template.mbon_approach_idx].sum()
        avoid = probe_circuit.v[self.template.mbon_avoid_idx].sum()
        return float(approach - avoid)

    def gain_drift(self) -> float:
        """L2 distance between this agent's current (live, possibly
        learned-from-experience) KC->MBON gains and its inherited prior
        -- a simple auditable "how much has this fly learned" scalar
        (decisions.md #26).
        """
        edge_idx = np.concatenate([self.template.kc_mbon_approach_edge_idx, self.template.kc_mbon_avoid_edge_idx])
        return float(np.linalg.norm(self.circuit.get_params()[edge_idx] - self.prior_gains[edge_idx]))

    def get_params(self) -> np.ndarray:
        """Live gains -- what THIS fly currently has, including whatever
        it has learned so far. NOT what gets inherited by offspring; see
        `prior_gains` and decisions.md #22 part 3.
        """
        return self.circuit.get_params()

    def set_params(self, theta: np.ndarray) -> None:
        self.circuit.set_params(theta)
