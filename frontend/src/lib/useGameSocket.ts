"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  BrainSnapshot,
  BrainTopology,
  ClientMessage,
  NeuronState,
  ServerMessage,
  TickData,
  WorldInitData,
} from "./protocol";

const DEFAULT_WS_URL = "ws://localhost:8000/ws";

export type ConnectionStatus = "connecting" | "open" | "closed";

export interface LogEntry {
  id: number;
  kind: "request" | "death" | "birth" | "error" | "info";
  text: string;
}

export interface GameSocket {
  status: ConnectionStatus;
  worldInit: WorldInitData | null;
  tick: TickData | null;
  owner: string | null;
  log: LogEntry[];
  join: (owner: string) => void;
  sendRequest: (text: string) => void;
  /** The fly id most recently sent via watch(), or null if not watching
   * anyone -- distinct from `brain`, which can lag behind by one message
   * (still the previous fly) right after switching.
   */
  watchedFlyId: number | null;
  /** Latest "brain" message: the watched fly's diagnostic snapshot, null
   * if not watching, or null once the watched fly has died (decisions.md
   * #46) -- server stops sending further updates for a dead fly, so this
   * stays null rather than going stale.
   */
  brain: BrainSnapshot | null;
  watch: (flyId: number | null) => void;
  /** Static connectome topology (wiki/decisions.md #47) -- arrives once
   * per connection, right after world_init, and never changes: every
   * fly of a kind shares the same subgraph.
   */
  brainTopology: BrainTopology | null;
  /** Latest "neuron_state" message: live per-neuron voltage/spike
   * arrays for the watched fly's two circuits, feeding the Tier 3
   * neuron-graph view -- null under the same conditions as `brain`
   * (not watching, or the watched fly died).
   */
  neuronState: NeuronState | null;
}

/** Owns the one WebSocket connection to server/app.py, parses every
 * message type in the contract, and exposes plain React state -- no
 * component below this needs to know the wire format exists.
 */
export function useGameSocket(wsUrl: string = process.env.NEXT_PUBLIC_WS_URL ?? DEFAULT_WS_URL): GameSocket {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [worldInit, setWorldInit] = useState<WorldInitData | null>(null);
  const [tick, setTick] = useState<TickData | null>(null);
  const [owner, setOwner] = useState<string | null>(null);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [watchedFlyId, setWatchedFlyId] = useState<number | null>(null);
  const [brain, setBrain] = useState<BrainSnapshot | null>(null);
  const [brainTopology, setBrainTopology] = useState<BrainTopology | null>(null);
  const [neuronState, setNeuronState] = useState<NeuronState | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const nextLogId = useRef(0);

  const appendLog = useCallback((kind: LogEntry["kind"], text: string) => {
    nextLogId.current += 1;
    setLog((prev) => [...prev.slice(-199), { id: nextLogId.current, kind, text }]);
  }, []);

  useEffect(() => {
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setStatus("open");
      appendLog("info", "connected");
    };
    socket.onclose = () => {
      setStatus("closed");
      appendLog("info", "disconnected");
    };
    socket.onerror = () => appendLog("error", "connection error");

    socket.onmessage = (event: MessageEvent<string>) => {
      const message = JSON.parse(event.data) as ServerMessage;
      switch (message.type) {
        case "world_init":
          setWorldInit(message.data);
          break;
        case "tick":
          setTick(message.data);
          for (const death of message.data.events.deaths) {
            appendLog("death", `fly #${death.fly_id} died (${death.cause})`);
          }
          for (const birth of message.data.events.births) {
            appendLog("birth", `fly #${birth.new_id} born (parent #${birth.parent_id})`);
          }
          break;
        case "joined":
          appendLog("info", `joined as ${message.data.owner}`);
          break;
        case "request_result":
          appendLog(
            "request",
            message.data.status ? `"${message.data.text}" -> ${message.data.status}` : `"${message.data.text}" -> no matching action`,
          );
          break;
        case "error":
          appendLog("error", message.data.message);
          break;
        case "brain":
          setBrain(message.data);
          break;
        case "brain_topology":
          setBrainTopology(message.data);
          break;
        case "neuron_state":
          setNeuronState(message.data);
          break;
      }
    };

    return () => socket.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- appendLog is stable (useCallback, no deps that change)
  }, [wsUrl]);

  const send = useCallback((message: ClientMessage) => {
    socketRef.current?.send(JSON.stringify(message));
  }, []);

  const join = useCallback(
    (newOwner: string) => {
      // Set locally right away rather than waiting on the server's
      // "joined" ack -- on a same-machine/local server this round trip
      // is real but pointless to block the UI on, and the ack still
      // arrives and gets logged for visibility.
      setOwner(newOwner);
      send({ type: "join", data: { owner: newOwner } });
    },
    [send],
  );

  const sendRequest = useCallback(
    (text: string) => {
      if (!owner || !text.trim()) return;
      send({ type: "player_request", data: { owner, text } });
    },
    [owner, send],
  );

  const watch = useCallback(
    (flyId: number | null) => {
      setWatchedFlyId(flyId);
      // clear stale data from whichever fly (if any) was watched before
      setBrain(null);
      setNeuronState(null);
      send({ type: "watch", data: { fly_id: flyId } });
    },
    [send],
  );

  return {
    status, worldInit, tick, owner, log, join, sendRequest,
    watchedFlyId, brain, watch, brainTopology, neuronState,
  };
}
