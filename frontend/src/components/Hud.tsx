"use client";

import type { TickData, WorldInitData } from "@/lib/protocol";

interface HudProps {
  worldInit: WorldInitData;
  tick: TickData | null;
  myOwner: string;
}

export function Hud({ worldInit, tick, myOwner }: HudProps) {
  const population = tick?.flies.length ?? 0;
  const byOwner = new Map<string, number>();
  for (const fly of tick?.flies ?? []) {
    byOwner.set(fly.owner, (byOwner.get(fly.owner) ?? 0) + 1);
  }
  const myEnergy = tick?.energy[myOwner] ?? 0;
  const energyFraction = worldInit.max_energy > 0 ? Math.max(0, Math.min(1, myEnergy / worldInit.max_energy)) : 0;

  return (
    <div className="flex flex-col gap-3 rounded border border-neutral-700 bg-neutral-900 p-3 text-sm text-neutral-200">
      <div className="flex justify-between">
        <span className="text-neutral-400">tick</span>
        <span>{tick?.tick ?? "—"}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-neutral-400">population</span>
        <span>{population}</span>
      </div>
      <div className="flex flex-col gap-1">
        {Array.from(byOwner.entries()).map(([owner, count]) => (
          <div key={owner} className="flex justify-between text-xs text-neutral-400">
            <span>{owner === myOwner ? `${owner} (you)` : owner}</span>
            <span>{count}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="flex justify-between text-neutral-400">
          <span>energy</span>
          <span>
            {myEnergy.toFixed(1)} / {worldInit.max_energy}
          </span>
        </div>
        <div className="mt-1 h-2 w-full rounded bg-neutral-800">
          <div className="h-2 rounded bg-emerald-400" style={{ width: `${energyFraction * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
