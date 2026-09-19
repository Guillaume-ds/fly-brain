"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { BrainTopology, CircuitTopology, NeuronState } from "@/lib/protocol";

const CANVAS_SIZE = 440;
const PADDING = 24;

type CircuitKind = "escape" | "plasticity";

interface Position {
  x: number;
  y: number;
}

const ESCAPE_GROUP_COLORS: Record<string, string> = {
  seed: "#f87171", // danger sensed here (DNp01)
  motor: "#34d399", // flee decision comes out here (TTMn)
  other: "#64748b",
};

const PLASTICITY_GROUP_COLORS: Record<string, string> = {
  kc: "#38bdf8", // Kenyon Cells -- sense
  mbon_approach: "#34d399", // MBON -- approach-coding
  mbon_avoid: "#f87171", // MBON -- avoid-coding
  pam: "#fbbf24", // reward-coding dopamine neurons
  ppl: "#e879f9", // punishment-coding dopamine neurons
};

const ESCAPE_LEGEND: [string, string][] = [
  ["seed (senses danger)", ESCAPE_GROUP_COLORS.seed],
  ["other (~2 hops between)", ESCAPE_GROUP_COLORS.other],
  ["motor (triggers flee)", ESCAPE_GROUP_COLORS.motor],
];

const PLASTICITY_LEGEND: [string, string][] = [
  ["Kenyon Cells (sense)", PLASTICITY_GROUP_COLORS.kc],
  ["MBON approach", PLASTICITY_GROUP_COLORS.mbon_approach],
  ["MBON avoid", PLASTICITY_GROUP_COLORS.mbon_avoid],
  ["dopamine: reward", PLASTICITY_GROUP_COLORS.pam],
  ["dopamine: punishment", PLASTICITY_GROUP_COLORS.ppl],
];

/** Column layout, computed once per topology (it never changes: every
 * fly of a kind shares the same connectome subgraph, wiki/decisions.md
 * #47) -- not a force-directed layout. A real force-directed layout was
 * considered and set aside for v1: at plasticity's scale (~480 neurons,
 * thousands of edges) it would mostly produce a hairball, where the
 * real functional clusters this template already carries (KC/MBON/DAN)
 * give a far more legible hint for free.
 */
// A dense group (e.g. plasticity's ~300 PAM neurons packed into one
// column) would otherwise stack points closer than their own radius,
// collapsing into a solid bar with no individual neuron visible at all
// -- a real rendering legibility problem, not a data one. A small
// deterministic sine-based x jitter (not random -- must be stable
// across redraws) turns that bar back into a legible scatter.
const JITTER_THRESHOLD = 25;
const JITTER_AMPLITUDE = 0.045;

function columnLayout(indices: number[], x: number, yMin: number, yMax: number, positions: Map<number, Position>) {
  const jitter = indices.length > JITTER_THRESHOLD;
  indices.forEach((idx, i) => {
    const y = indices.length <= 1 ? (yMin + yMax) / 2 : yMin + ((yMax - yMin) * i) / (indices.length - 1);
    const jx = jitter ? x + JITTER_AMPLITUDE * Math.sin(i * 2.4) : x;
    positions.set(idx, { x: jx, y });
  });
}

function layoutEscape(topology: CircuitTopology): Map<number, Position> {
  const positions = new Map<number, Position>();
  const seed = new Set(topology.groups.seed ?? []);
  const motor = new Set(topology.groups.motor ?? []);
  const other: number[] = [];
  for (let i = 0; i < topology.neuron_ids.length; i++) {
    if (!seed.has(i) && !motor.has(i)) other.push(i);
  }
  columnLayout(topology.groups.seed ?? [], 0.1, 0, 1, positions);
  columnLayout(other, 0.5, 0, 1, positions);
  columnLayout(topology.groups.motor ?? [], 0.9, 0, 1, positions);
  return positions;
}

function layoutPlasticity(topology: CircuitTopology): Map<number, Position> {
  const positions = new Map<number, Position>();
  columnLayout(topology.groups.kc ?? [], 0.12, 0, 1, positions);
  columnLayout(topology.groups.mbon_approach ?? [], 0.52, 0, 0.46, positions);
  columnLayout(topology.groups.mbon_avoid ?? [], 0.52, 0.54, 1, positions);
  columnLayout(topology.groups.pam ?? [], 0.9, 0, 0.46, positions);
  columnLayout(topology.groups.ppl ?? [], 0.9, 0.54, 1, positions);
  return positions;
}

function groupOf(topology: CircuitTopology, index: number): string {
  for (const [name, indices] of Object.entries(topology.groups)) {
    if (indices.includes(index)) return name;
  }
  return "other";
}

interface BrainGraphProps {
  topology: BrainTopology | null;
  neuronState: NeuronState | null;
  watchedFlyId: number | null;
}

/** Tier 3 of the brain inspector (wiki/decisions.md #47): the real
 * connectome subgraph a watched fly's circuit runs on, with live spike
 * animation -- goal 3 from the original feature request ("show this
 * isn't just a poor Minecraft clone") made concrete, on top of tiers
 * 1+2's summary numbers (decisions.md #46).
 */
export function BrainGraph({ topology, neuronState, watchedFlyId }: BrainGraphProps) {
  const [open, setOpen] = useState(true);
  const [circuit, setCircuit] = useState<CircuitKind>("escape");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const escapeLayout = useMemo(() => (topology ? layoutEscape(topology.escape) : null), [topology]);
  const plasticityLayout = useMemo(() => (topology ? layoutPlasticity(topology.plasticity) : null), [topology]);

  useEffect(() => {
    if (!open || !topology) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const circuitTopology = topology[circuit];
    const layout = circuit === "escape" ? escapeLayout : plasticityLayout;
    if (!layout) return;
    const state = neuronState ? neuronState[circuit] : null;

    const toPixel = (p: Position) => ({
      x: PADDING + p.x * (CANVAS_SIZE - 2 * PADDING),
      y: PADDING + p.y * (CANVAS_SIZE - 2 * PADDING),
    });

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    const maxWeight = Math.max(1, ...circuitTopology.edges.map((e) => e.weight));
    ctx.strokeStyle = "#ffffff";
    for (const edge of circuitTopology.edges) {
      const from = layout.get(edge.pre);
      const to = layout.get(edge.post);
      if (!from || !to) continue;
      const a = toPixel(from);
      const b = toPixel(to);
      ctx.globalAlpha = 0.04 + 0.16 * (edge.weight / maxWeight);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const colors = circuit === "escape" ? ESCAPE_GROUP_COLORS : PLASTICITY_GROUP_COLORS;
    for (let i = 0; i < circuitTopology.neuron_ids.length; i++) {
      const pos = layout.get(i);
      if (!pos) continue;
      const { x, y } = toPixel(pos);
      const spiking = state?.spikes[i] ?? false;
      const v = state?.v[i] ?? 0;
      const group = groupOf(circuitTopology, i);
      const color = colors[group] ?? "#94a3b8";

      if (spiking) {
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.9;
        ctx.stroke();
      }
      const intensity = Math.max(0, Math.min(1, v));
      ctx.globalAlpha = spiking ? 1 : 0.3 + intensity * 0.6;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, spiking ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }, [open, topology, neuronState, circuit, escapeLayout, plasticityLayout]);

  if (watchedFlyId === null) return null;

  const legend = circuit === "escape" ? ESCAPE_LEGEND : PLASTICITY_LEGEND;

  return (
    <div className="flex flex-col gap-2 rounded border border-neutral-700 bg-neutral-900 p-3 text-sm text-neutral-200">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-neutral-500">neuron graph (real connectome)</span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="rounded bg-neutral-800 px-2 py-0.5 text-xs text-neutral-300 hover:bg-neutral-700"
        >
          {open ? "hide" : "show"}
        </button>
      </div>

      {open && (
        <>
          <div className="flex gap-1">
            {(["escape", "plasticity"] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                onClick={() => setCircuit(kind)}
                className={
                  "rounded px-2 py-1 text-xs " +
                  (circuit === kind ? "bg-sky-800 text-sky-100" : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700")
                }
              >
                {kind === "escape" ? "escape (frozen)" : "plasticity (learned)"}
              </button>
            ))}
          </div>

          {!topology ? (
            <div className="text-xs text-neutral-500">loading topology…</div>
          ) : (
            <>
              <canvas ref={canvasRef} width={CANVAS_SIZE} height={CANVAS_SIZE} className="rounded" />
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-neutral-400">
                {legend.map(([label, color]) => (
                  <span key={label} className="flex items-center gap-1">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                    {label}
                  </span>
                ))}
              </div>
              {!neuronState && (
                <div className="text-xs text-neutral-600">no live spike data yet -- waiting on the next tick</div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
