"use client";

import { useState } from "react";

export function JoinScreen({ onJoin }: { onJoin: (owner: string) => void }) {
  const [name, setName] = useState("player");

  return (
    <form
      className="flex flex-col items-center gap-4 rounded border border-neutral-700 bg-neutral-900 p-8"
      onSubmit={(event) => {
        event.preventDefault();
        if (name.trim()) onJoin(name.trim());
      }}
    >
      <h1 className="text-lg text-neutral-100">kill-the-flies</h1>
      <p className="max-w-sm text-center text-sm text-neutral-400">
        Pick a name for your colony, then type instructions to shape the world -- create items, tiles,
        mobs, or nudge the spawn rates.
      </p>
      <input
        className="w-64 rounded border border-neutral-600 bg-neutral-800 px-3 py-2 text-neutral-100"
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="your colony's name"
        autoFocus
      />
      <button className="rounded bg-sky-500 px-4 py-2 font-medium text-neutral-950 hover:bg-sky-400" type="submit">
        Join
      </button>
    </form>
  );
}
