"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// ─── Types ──────────────────────────────────────────────────────────────────

interface AnimationFrame {
  index: number;
  x: number;
  y: number;
  w: number;
  h: number;
  /** v2 only — per-frame hold time in milliseconds */
  duration_ms?: number;
  /** v2 only — horizontal CSS translate offset in pixels */
  offset_x?: number;
  /** v2 only — vertical CSS translate offset in pixels */
  offset_y?: number;
}

/** Parsed from animation.json (v1 or v2 schema) */
interface AnimationMeta {
  version?: number;
  fps: number;
  frame_count: number;
  frame_width: number;
  frame_height: number;
  loop: boolean;
  frames: AnimationFrame[];
}

interface SpritePreviewProps {
  /** URL of the horizontal sprite sheet PNG */
  spriteUrl: string;
  /**
   * URL of the animation.json metadata file.
   * When provided, all frame dimensions/timing are driven by the JSON.
   * Legacy numeric props are used as fallback only.
   */
  animationJsonUrl?: string;
  /** Total number of frames — used as fallback when animationJsonUrl is absent */
  frameCount?: number;
  /** Width of each individual frame (px) — fallback */
  frameWidth?: number;
  /** Height of each individual frame (px) — fallback */
  frameHeight?: number;
  /** Playback speed in frames per second — fallback */
  fps?: number;
  /** Display size (CSS pixels). When absent the player fills available width. */
  displaySize?: number;
  /** Whether to loop the animation — fallback */
  loop?: boolean;
  className?: string;
}

// ─── Component ───────────────────────────────────────────────────────────────

/**
 * Sprite-sheet animation player using CSS background-position stepping.
 *
 * The sprite sheet is a single horizontal row:
 *   [frame0][frame1][frame2]…[frameN]
 *
 * When `animationJsonUrl` is provided the component fetches the JSON on mount
 * and uses per-frame `duration_ms` (v2) or uniform `1000/fps` (v1).
 * Falls back to caller props when the URL is absent or the fetch fails.
 */
export default function SpritePreview({
  spriteUrl,
  animationJsonUrl,
  frameCount: propFrameCount = 17,
  frameWidth: propFrameWidth = 512,
  frameHeight: propFrameHeight = 512,
  fps: propFps = 16,
  displaySize,
  loop: propLoop = true,
  className = "",
}: SpritePreviewProps) {
  // ── Metadata state ─────────────────────────────────────────────────────────
  const [meta, setMeta] = useState<AnimationMeta | null>(null);
  const [metaLoading, setMetaLoading] = useState(!!animationJsonUrl);

  // Resolved values: JSON wins over props when available
  const frameCount  = meta?.frame_count  ?? propFrameCount;
  const frameWidth  = meta?.frame_width  ?? propFrameWidth;
  const frameHeight = meta?.frame_height ?? propFrameHeight;
  const fps         = meta?.fps          ?? propFps;
  const loop        = meta?.loop         ?? propLoop;

  // Per-frame durations (ms). Falls back to uniform 1000/fps.
  const getFrameDuration = useCallback(
    (index: number): number => {
      const frame = meta?.frames[index];
      return frame?.duration_ms ?? Math.max(16, Math.round(1000 / fps));
    },
    [meta, fps],
  );

  // Per-frame offsets (CSS translate). Defaults to (0, 0).
  const getFrameOffset = useCallback(
    (index: number): { x: number; y: number } => {
      const frame = meta?.frames[index];
      return { x: frame?.offset_x ?? 0, y: frame?.offset_y ?? 0 };
    },
    [meta],
  );

  // ── Fetch animation.json ───────────────────────────────────────────────────
  useEffect(() => {
    if (!animationJsonUrl) {
      setMetaLoading(false);
      return;
    }

    let cancelled = false;
    setMetaLoading(true);

    fetch(animationJsonUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<AnimationMeta>;
      })
      .then((data) => {
        if (!cancelled) {
          setMeta(data);
          setMetaLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.warn(
            "[SpritePreview] Failed to fetch animation.json, falling back to props:",
            err,
          );
          setMeta(null);
          setMetaLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [animationJsonUrl]);

  // ── Animation state ────────────────────────────────────────────────────────
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);

  // Refs for rAF loop
  const rafRef       = useRef<number | null>(null);
  const lastTimeRef  = useRef<number | null>(null);
  const elapsedRef   = useRef<number>(0);
  const frameRef     = useRef<number>(0);
  const playingRef   = useRef<boolean>(true);

  // Keep refs in sync with state
  useEffect(() => { frameRef.current = frame; }, [frame]);
  useEffect(() => { playingRef.current = playing; }, [playing]);

  // ── requestAnimationFrame loop ─────────────────────────────────────────────
  const tick = useCallback(
    (timestamp: number) => {
      if (!playingRef.current) return;

      if (lastTimeRef.current === null) {
        lastTimeRef.current = timestamp;
      }

      const delta = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;
      elapsedRef.current += delta;

      const currentFrame = frameRef.current;
      const holdMs = getFrameDuration(currentFrame);

      if (elapsedRef.current >= holdMs) {
        elapsedRef.current -= holdMs;
        const next = currentFrame + 1;

        if (next >= frameCount) {
          if (loop) {
            setFrame(0);
            frameRef.current = 0;
          } else {
            setFrame(frameCount - 1);
            setPlaying(false);
            return; // stop loop
          }
        } else {
          setFrame(next);
          frameRef.current = next;
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    },
    [frameCount, loop, getFrameDuration],
  );

  // Start / stop the rAF loop based on playing + metadata ready
  useEffect(() => {
    if (metaLoading) return; // wait for JSON before starting

    if (playing) {
      lastTimeRef.current = null;
      elapsedRef.current = 0;
      rafRef.current = requestAnimationFrame(tick);
    } else {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      lastTimeRef.current = null;
    }

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [playing, tick, metaLoading]);

  // ── Layout calculations ────────────────────────────────────────────────────
  // If displaySize provided, use it; otherwise fill container (natural frame width)
  const displayW = displaySize ?? frameWidth;
  const aspectRatio = frameHeight > 0 ? frameWidth / frameHeight : 1;
  const displayH = Math.round(displayW / aspectRatio);

  // Scale factor and full sprite sheet width in CSS px
  const scale = displayW / frameWidth;
  const sheetDisplayW = frameWidth * frameCount * scale;

  // Background offset for current frame
  const bgOffsetX = -(frame * displayW);

  // Per-frame CSS translate offset
  const offset = getFrameOffset(frame);

  // ── Render ─────────────────────────────────────────────────────────────────
  if (metaLoading) {
    return (
      <div
        className={`flex flex-col items-center gap-2 ${className}`}
        style={{ width: displayW }}
      >
        <div
          style={{
            width: displayW,
            height: displayH,
            borderRadius: 8,
            backgroundColor: "rgba(0,0,0,0.3)",
          }}
          className="animate-pulse"
        />
        <span className="text-xs text-gray-500">Cargando animación…</span>
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center gap-2 ${className}`}>
      {/* Sprite container — centered, maintains aspect ratio */}
      <div
        style={{ width: displayW, maxWidth: "100%", height: displayH, position: "relative", overflow: "hidden" }}
      >
        {/* Inner div applies per-frame CSS translate offset */}
        <div
          title={`Frame ${frame + 1}/${frameCount}`}
          style={{
            width: displayW,
            height: displayH,
            backgroundImage: `url(${spriteUrl})`,
            backgroundSize: `${sheetDisplayW}px ${displayH}px`,
            backgroundPosition: `${bgOffsetX}px 0px`,
            backgroundRepeat: "no-repeat",
            imageRendering: "auto",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 8,
            backgroundColor: "rgba(0,0,0,0.3)",
            cursor: "pointer",
            transform: `translate(${offset.x}px, ${offset.y}px)`,
            // Center the character within the container
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          onClick={() => setPlaying((p) => !p)}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <button
          onClick={() => {
            setFrame(0);
            frameRef.current = 0;
            elapsedRef.current = 0;
            lastTimeRef.current = null;
            setPlaying(true);
          }}
          className="hover:text-white transition-colors"
          title="Reiniciar"
        >
          ⏮
        </button>
        <button
          onClick={() => setPlaying((p) => !p)}
          className="hover:text-white transition-colors text-base"
          title={playing ? "Pausar" : "Reproducir"}
        >
          {playing ? "⏸" : "▶"}
        </button>
        <span>
          {frame + 1}/{frameCount} · {fps}fps
          {meta?.version === 2 && " · v2"}
        </span>
      </div>

      <p className="text-xs text-gray-600">Click en la imagen para pausar/reanudar</p>
    </div>
  );
}
