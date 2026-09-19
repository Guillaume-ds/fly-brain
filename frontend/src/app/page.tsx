"use client";

import { BrainPanel } from "@/components/BrainPanel";
import { GameCanvas } from "@/components/GameCanvas";
import { Hud } from "@/components/Hud";
import { JoinScreen } from "@/components/JoinScreen";
import { RequestInput } from "@/components/RequestInput";
import { RequestLog } from "@/components/RequestLog";
import { useGameSocket } from "@/lib/useGameSocket";

export default function Home() {
  const { status, worldInit, tick, owner, log, join, sendRequest, watchedFlyId, brain, watch } = useGameSocket();

  return (
    <div className="flex min-h-screen flex-col items-center gap-4 bg-neutral-950 p-6 text-neutral-100">
      <div className="flex w-full max-w-5xl items-center justify-between">
        <h1 className="text-lg font-semibold">kill-the-flies</h1>
        <span className="text-xs text-neutral-500">
          server: <span className={status === "open" ? "text-emerald-400" : "text-amber-400"}>{status}</span>
        </span>
      </div>

      {!worldInit || !owner ? (
        <div className="flex flex-1 items-center justify-center">
          {!worldInit ? (
            <p className="text-neutral-500">connecting…</p>
          ) : (
            <JoinScreen onJoin={join} />
          )}
        </div>
      ) : (
        <div className="flex w-full max-w-5xl gap-4">
          <GameCanvas worldInit={worldInit} tick={tick} myOwner={owner} onFlyClick={watch} watchedFlyId={watchedFlyId} />
          <div className="flex w-72 flex-col gap-4">
            <Hud worldInit={worldInit} tick={tick} myOwner={owner} />
            <BrainPanel watchedFlyId={watchedFlyId} brain={brain} />
            <RequestLog entries={log} />
            <RequestInput onSend={sendRequest} />
          </div>
        </div>
      )}
    </div>
  );
}
