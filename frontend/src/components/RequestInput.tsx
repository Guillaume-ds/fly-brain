"use client";

import { useState } from "react";

export function RequestInput({ onSend }: { onSend: (text: string) => void }) {
  const [text, setText] = useState("");

  return (
    <form
      className="flex gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (!text.trim()) return;
        onSend(text.trim());
        setText("");
      }}
    >
      <input
        className="flex-1 rounded border border-neutral-600 bg-neutral-800 px-3 py-2 text-sm text-neutral-100"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="e.g. create a venomous spider mob, or more food"
      />
      <button className="rounded bg-sky-500 px-4 py-2 text-sm font-medium text-neutral-950 hover:bg-sky-400" type="submit">
        Send
      </button>
    </form>
  );
}
