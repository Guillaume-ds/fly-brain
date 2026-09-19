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

export type ServerMessage =
  | { type: "world_init"; data: WorldInitData }
  | { type: "tick"; data: TickData }
  | { type: "joined"; data: JoinedData }
  | { type: "request_result"; data: RequestResultData }
  | { type: "error"; data: ErrorData };

export type ClientMessage =
  | { type: "join"; data: { owner: string } }
  | { type: "player_request"; data: { owner: string; text: string } };
