"use client";

import { useEffect, useRef } from "react";

import type { LogEntry } from "@/lib/useGameSocket";

const KIND_STYLES: Record<LogEntry["kind"], string> = {
  request: "text-sky-300",
  death: "text-red-400",
  birth: "text-emerald-400",
  error: "text-amber-400",
  info: "text-neutral-500",
};

export function RequestLog({ entries }: { entries: LogEntry[] }) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [entries.length]);

  return (
    <div className="flex h-48 flex-col overflow-y-auto rounded border border-neutral-700 bg-neutral-900 p-2 text-xs">
      {entries.length === 0 && <div className="text-neutral-600">nothing yet</div>}
      {entries.map((entry) => (
        <div key={entry.id} className={KIND_STYLES[entry.kind]}>
          {entry.text}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
