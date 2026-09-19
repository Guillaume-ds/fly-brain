"use client";

import type { ReactNode } from "react";

import type { BrainSnapshot } from "@/lib/protocol";

interface BrainPanelProps {
  watchedFlyId: number | null;
  brain: BrainSnapshot | null;
}

const ACTION_SOURCE_LABEL: Record<string, string> = {
  escape: "Escape reflex",
  wander: "Wander",
  plasticity: "Learned response",
};

const ACTION_SOURCE_EXPLANATION: Record<string, string> = {
  escape: "A frozen, evolved circuit (trained offline, never changes during life) sensed danger and its flee reflex fired -- this always wins over anything else.",
  wander: "Nothing was in range to react to, so a fixed, evolved fallback kept the fly moving instead of freezing in place.",
  plasticity: "This fly's own live, dopamine-gated circuit -- shaped by what has actually happened to it so far -- chose this move.",
};

/** Live "brain inspector" (wiki/decisions.md #46) for whichever single
 * fly is currently watched -- goals 1 and 2 from the feature request:
 * "why did it do that" (action_source + the real numbers behind it) and
 * "how does it differ from others" (click a different fly to compare;
 * no generational history, live flies only, per the agreed scope).
 */
export function BrainPanel({ watchedFlyId, brain }: BrainPanelProps) {
  if (watchedFlyId === null) {
    return (
      <div className="rounded border border-neutral-700 bg-neutral-900 p-3 text-sm text-neutral-500">
        click a fly to inspect its brain
      </div>
    );
  }

  // Server sends `data: null` both right after a watch() for a fly that
  // no longer exists and once a previously-live watched fly dies
  // (decisions.md #46) -- either way there's simply nothing to show.
  if (!brain) {
    return (
      <div className="rounded border border-neutral-700 bg-neutral-900 p-3 text-sm text-neutral-500">
        no data for fly #{watchedFlyId} (it may have died, or hasn&apos;t reported in yet)
      </div>
    );
  }

  const sourceLabel = brain.action_source ? ACTION_SOURCE_LABEL[brain.action_source] : "—";
  const sourceExplanation = brain.action_source ? ACTION_SOURCE_EXPLANATION[brain.action_source] : null;

  return (
    <div className="flex flex-col gap-3 rounded border border-neutral-700 bg-neutral-900 p-3 text-sm text-neutral-200">
      <div className="flex items-center justify-between">
        <span className="font-semibold">fly #{brain.fly_id}</span>
        <span
          className={
            "rounded px-2 py-0.5 text-xs font-medium " +
            (brain.action_source === "escape"
              ? "bg-red-900 text-red-200"
              : brain.action_source === "wander"
                ? "bg-sky-900 text-sky-200"
                : "bg-violet-900 text-violet-200")
          }
        >
          {sourceLabel}
        </span>
      </div>
      {sourceExplanation && <p className="text-xs text-neutral-500">{sourceExplanation}</p>}

      <Section title="escape reflex (frozen, evolved)">
        <Stat
          label="danger sensed"
          value={brain.escape.danger_strength.toFixed(3)}
          hint="0 = nothing nearby resembles a threat"
        />
      </Section>

      <Section title="learned response (live, dopamine-gated)">
        <Stat
          label="current pull"
          value={signed(brain.plasticity.valence)}
          hint="positive = drawn toward what's nearby, negative = repelled"
        />
        <Stat
          label="learned so far"
          value={brain.plasticity.gain_drift.toFixed(3)}
          hint="distance from this fly's inherited starting point -- 0 means no learning yet"
        />
        <Stat label="reinforcement events" value={String(brain.plasticity.reinforcement_events)} />
        <Stat label="cumulative dopamine" value={brain.plasticity.cumulative_dopamine.toFixed(3)} />
        <Stat
          label="learned pull toward food-like things"
          value={signed(brain.plasticity.probe_food)}
          hint="positive = has learned to approach, negative = has learned to avoid"
        />
        <Stat
          label="learned pull toward danger-like things"
          value={signed(brain.plasticity.probe_danger)}
        />
      </Section>

      <Section title="wander (evolved fallback)">
        <Stat
          label="persistence"
          value={brain.wander.persistence.toFixed(2)}
          hint="higher = holds a direction longer before changing"
        />
        <Stat label="currently heading" value={brain.wander.current_direction ?? "— (reactive right now)"} />
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 border-t border-neutral-800 pt-2">
      <span className="text-xs uppercase tracking-wide text-neutral-500">{title}</span>
      {children}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2" title={hint}>
      <span className="text-neutral-400">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}

function signed(value: number): string {
  const rounded = value.toFixed(3);
  return value > 0 ? `+${rounded}` : rounded;
}
