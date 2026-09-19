"use client";

import { useEffect, useRef, useState } from "react";
import type Phaser from "phaser";

import type { EntityPayload, FlyPayload, TickData, WorldInitData } from "@/lib/protocol";

const CANVAS_SIZE = 640;
const OWNER_COLORS = [0x60a5fa, 0xfacc15, 0xf472b6, 0x34d399, 0xfb923c, 0xa78bfa];
const KIND_COLORS: Record<"item" | "tile" | "mob" | "corpse", number> = {
  item: 0x4ade80,
  tile: 0x38bdf8,
  mob: 0xf87171,
  corpse: 0x78716c,
};

interface HoverInfo {
  fly: FlyPayload;
  screenX: number;
  screenY: number;
}

interface GameCanvasProps {
  worldInit: WorldInitData;
  tick: TickData | null;
  myOwner: string;
  /** Fired with a fly's id on click (decisions.md #46) -- the frontend's
   * only entry point into "watch this fly's brain". Not called for a
   * click that doesn't land on any fly.
   */
  onFlyClick?: (flyId: number) => void;
  watchedFlyId?: number | null;
}

/** A thin wrapper around one Phaser.Game: redraws the grid from scratch
 * on every tick (deliberate -- full-state redraw, not incremental
 * diffing, matches the wire contract's own "full state, not a diff"
 * choice, decisions.md #44). Hover detection is plain pointer-position
 * math against the latest tick's fly list, not a Phaser input object per
 * fly -- simpler, and this is a "simple colored shapes" v1
 * (decisions.md #45).
 */
export function GameCanvas({ worldInit, tick, myOwner, onFlyClick, watchedFlyId = null }: GameCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const gameRef = useRef<Phaser.Game | null>(null);
  const graphicsRef = useRef<Phaser.GameObjects.Graphics | null>(null);
  const tickRef = useRef<TickData | null>(tick);
  const ownerColorsRef = useRef<Map<string, number>>(new Map());
  const onFlyClickRef = useRef(onFlyClick);
  const watchedFlyIdRef = useRef(watchedFlyId);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  onFlyClickRef.current = onFlyClick;
  watchedFlyIdRef.current = watchedFlyId;

  const cellSize = CANVAS_SIZE / worldInit.grid_size;

  const colorForOwner = (owner: string): number => {
    const known = ownerColorsRef.current;
    if (!known.has(owner)) {
      known.set(owner, OWNER_COLORS[known.size % OWNER_COLORS.length]);
    }
    return known.get(owner) ?? 0xffffff;
  };

  // Mount Phaser once. Dynamically imported so it only ever runs
  // client-side -- Phaser needs a real canvas/window, which doesn't
  // exist during Next.js's server render.
  useEffect(() => {
    let destroyed = false;

    import("phaser").then((PhaserModule) => {
      if (destroyed || !containerRef.current) return;
      const Phaser = PhaserModule.default;

      class GridScene extends Phaser.Scene {
        create() {
          graphicsRef.current = this.add.graphics();
          this.input.on("pointermove", (pointer: Phaser.Input.Pointer) => {
            const current = tickRef.current;
            if (!current) return;
            const gx = Math.floor(pointer.x / cellSize);
            const gy = Math.floor(pointer.y / cellSize);
            const found = current.flies.find((fly) => fly.x === gx && fly.y === gy);
            setHover(found ? { fly: found, screenX: pointer.x, screenY: pointer.y } : null);
          });
          this.input.on("pointerdown", (pointer: Phaser.Input.Pointer) => {
            const current = tickRef.current;
            if (!current) return;
            const gx = Math.floor(pointer.x / cellSize);
            const gy = Math.floor(pointer.y / cellSize);
            const found = current.flies.find((fly) => fly.x === gx && fly.y === gy);
            if (found) onFlyClickRef.current?.(found.id);
          });
        }
      }

      const game = new Phaser.Game({
        type: Phaser.AUTO,
        width: CANVAS_SIZE,
        height: CANVAS_SIZE,
        parent: containerRef.current,
        backgroundColor: "#111827",
        scene: GridScene,
      });
      gameRef.current = game;
    });

    return () => {
      destroyed = true;
      gameRef.current?.destroy(true);
      gameRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount once; redraws happen via the effect below, not by recreating the game
  }, []);

  // Redraw whenever a new tick arrives.
  useEffect(() => {
    tickRef.current = tick;
    const graphics = graphicsRef.current;
    if (!graphics || !tick) return;

    graphics.clear();
    drawGrid(graphics, worldInit.grid_size, cellSize);
    drawEntities(graphics, tick.corpses, cellSize, KIND_COLORS.corpse);
    drawEntities(graphics, tick.tiles, cellSize, KIND_COLORS.tile);
    drawEntities(graphics, tick.items, cellSize, KIND_COLORS.item);
    drawEntities(graphics, tick.mobs, cellSize, KIND_COLORS.mob);
    for (const fly of tick.flies) {
      drawFly(
        graphics, fly, cellSize, colorForOwner(fly.owner), fly.owner === myOwner, worldInit.max_hunger,
        fly.id === watchedFlyId,
      );
    }
  }, [tick, worldInit.grid_size, worldInit.max_hunger, cellSize, myOwner, watchedFlyId]);

  return (
    <div className="relative inline-block">
      <div ref={containerRef} style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }} />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded bg-black/80 px-2 py-1 text-xs text-white"
          style={{ left: hover.screenX + 12, top: hover.screenY + 12 }}
        >
          <div>owner: {hover.fly.owner}</div>
          <div>hunger: {hover.fly.hunger}</div>
          <div>health: {hover.fly.health}</div>
          {hover.fly.stuck && <div>stuck</div>}
          {hover.fly.vulnerable && <div>vulnerable</div>}
        </div>
      )}
    </div>
  );
}

function drawGrid(graphics: Phaser.GameObjects.Graphics, gridSize: number, cellSize: number) {
  graphics.lineStyle(1, 0x1f2937, 1);
  for (let i = 0; i <= gridSize; i++) {
    graphics.lineBetween(i * cellSize, 0, i * cellSize, gridSize * cellSize);
    graphics.lineBetween(0, i * cellSize, gridSize * cellSize, i * cellSize);
  }
}

function drawEntities(graphics: Phaser.GameObjects.Graphics, entities: EntityPayload[], cellSize: number, color: number) {
  const size = cellSize * 0.6;
  const offset = (cellSize - size) / 2;
  graphics.fillStyle(color, 1);
  for (const entity of entities) {
    graphics.fillRect(entity.x * cellSize + offset, entity.y * cellSize + offset, size, size);
  }
}

function drawFly(
  graphics: Phaser.GameObjects.Graphics,
  fly: FlyPayload,
  cellSize: number,
  color: number,
  isMine: boolean,
  maxHunger: number,
  isWatched: boolean,
) {
  const cx = fly.x * cellSize + cellSize / 2;
  const cy = fly.y * cellSize + cellSize / 2;
  const radius = cellSize * 0.32;
  graphics.fillStyle(color, 1);
  graphics.fillCircle(cx, cy, radius);
  if (isMine) {
    graphics.lineStyle(2, 0xffffff, 1);
    graphics.strokeCircle(cx, cy, radius + 2);
  }
  if (isWatched) {
    // A color deliberately outside OWNER_COLORS so the watched ring never
    // blends into a same-colored owner ring (decisions.md #46).
    graphics.lineStyle(2, 0x22d3ee, 1);
    graphics.strokeCircle(cx, cy, radius + (isMine ? 5 : 2));
  }
  // A thin hunger bar under the fly -- the one always-visible stat, the rest live in the hover tooltip.
  const barWidth = cellSize * 0.7;
  const barX = cx - barWidth / 2;
  const barY = cy + radius + 3;
  graphics.fillStyle(0x1f2937, 1);
  graphics.fillRect(barX, barY, barWidth, 3);
  graphics.fillStyle(0xfacc15, 1);
  graphics.fillRect(barX, barY, barWidth * Math.max(0, Math.min(1, fly.hunger / maxHunger)), 3);
}
