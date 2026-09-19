/**
 * Wire-format types (wiki/decisions.md #44), mirroring server/serialize.py
 * and server/app.py exactly -- field-for-field, not reinterpreted. Keep
 * this in sync with the backend by hand; there's no shared schema source
 * yet (a follow-up worth doing once the shape stabilizes).
 */

export interface ItemTypeInfo {
  name: string;
  description: string;
  strength: number;
}

export interface MobTypeInfo extends ItemTypeInfo {
  effect: string | null;
}

export interface WorldInitData {
  grid_size: number;
  max_hunger: number;
  max_health: number;
  max_stuck_ticks: number;
  max_energy: number;
  owners: string[];
  item_types: ItemTypeInfo[];
  tile_types: ItemTypeInfo[];
  mob_types: MobTypeInfo[];
}

export interface FlyPayload {
  id: number;
  x: number;
  y: number;
  hunger: number;
  health: number;
  owner: string;
  stuck: boolean;
  vulnerable: boolean;
}

export interface EntityPayload {
  id: number;
  type_name: string;
  x: number;
  y: number;
}

export interface TickEvents {
  deaths: { fly_id: number; cause: string }[];
  births: { new_id: number; parent_id: number }[];
  colony_extinct: Record<string, boolean>;
}

export interface TickData {
  tick: number;
  flies: FlyPayload[];
  items: EntityPayload[];
  tiles: EntityPayload[];
  mobs: EntityPayload[];
  corpses: EntityPayload[];
  energy: Record<string, number>;
  events: TickEvents;
}

export interface JoinedData {
  owner: string;
}

export interface RequestResultData {
  text: string;
  status: string | null;
}

export interface ErrorData {
  message: string;
}

/**
 * Per-fly diagnostic bundle (wiki/decisions.md #46), mirroring
 * game/colony.py's Colony.brain_snapshot() exactly. `action_source` is
 * which of the three tiers produced the fly's last action; the rest is
 * the real numbers behind that decision, straight from the agents'
 * own audit state -- nothing computed fresh in the frontend.
 */
export interface BrainSnapshot {
  fly_id: number;
  action_source: "escape" | "wander" | "plasticity" | null;
  escape: {
    danger_strength: number;
  };
  plasticity: {
    valence: number;
    gain_drift: number;
    reinforcement_events: number;
    cumulative_dopamine: number;
    probe_food: number;
    probe_danger: number;
  };
  wander: {
    persistence: number;
    current_direction: string | null;
  };
}

/**
 * Static, colony-wide connectome topology (wiki/decisions.md #47),
 * mirroring game/colony.py's Colony.brain_topology() exactly. The same
 * for every fly of a kind -- only synaptic gains differ per fly, never
 * topology -- so this arrives once per connection, not per tick.
 * `edges[].pre`/`post` and every index in `groups` are positions into
 * `neuron_ids` (real MaleCNS body ids, for reference/labels only).
 */
export interface TopologyEdge {
  pre: number;
  post: number;
  weight: number;
}

export interface CircuitTopology {
  neuron_ids: number[];
  edges: TopologyEdge[];
  groups: Record<string, number[]>;
}

export interface BrainTopology {
  escape: CircuitTopology;
  plasticity: CircuitTopology;
}

/**
 * Live per-neuron membrane potential + spike state for a watched fly's
 * two circuits (wiki/decisions.md #47), mirroring
 * Colony.neuron_state() exactly. `v`/`spikes` are index-aligned with
 * the matching circuit's `neuron_ids` in BrainTopology.
 */
export interface CircuitNeuronState {
  v: number[];
  spikes: boolean[];
}

export interface NeuronState {
  fly_id: number;
  escape: CircuitNeuronState;
  plasticity: CircuitNeuronState;
}

export type ServerMessage =
  | { type: "world_init"; data: WorldInitData }
  | { type: "tick"; data: TickData }
  | { type: "joined"; data: JoinedData }
  | { type: "request_result"; data: RequestResultData }
  | { type: "error"; data: ErrorData }
  | { type: "brain"; data: BrainSnapshot | null }
  | { type: "brain_topology"; data: BrainTopology }
  | { type: "neuron_state"; data: NeuronState | null };

export type ClientMessage =
  | { type: "join"; data: { owner: string } }
  | { type: "player_request"; data: { owner: string; text: string } }
  | { type: "watch"; data: { fly_id: number | null } };
