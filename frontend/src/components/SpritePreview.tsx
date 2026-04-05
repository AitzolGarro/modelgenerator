"use client";

import { useEffect, useRef, useState } from "react";

interface SpritePreviewProps {
  /** URL of the horizontal sprite sheet PNG */
  spriteUrl: string;
  /** Total number of frames */
  frameCount: number;
  /** Width of each individual frame (px) */
  frameWidth: number;
  /** Height of each individual frame (px) */
  frameHeight: number;
  /** Playback speed in frames per second */
  fps: number;
  /** Display size (CSS pixels). Sprite is scaled to fit. */
  displaySize?: number;
  /** Whether to loop the animation */
  loop?: boolean;
  className?: string;
}

/**
 * Sprite-sheet animation player using CSS background-position stepping.
 *
 * The sprite sheet is a single horizontal row:
 *   [frame0][frame1][frame2]…[frameN]
 *
 * We cycle through frames by shifting background-position-x by frameWidth each tick.
 */
export default function SpritePreview({
  spriteUrl,
  frameCount,
  frameWidth,
  frameHeight,
  fps,
  displaySize = 256,
  loop = true,
  className = "",
}: SpritePreviewProps) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Aspect ratio of a single frame
  const aspectRatio = frameHeight > 0 ? frameWidth / frameHeight : 1;
  const displayW = displaySize;
  const displayH = Math.round(displaySize / aspectRatio);

  // Scale factor: how many CSS px per sprite px
  const scale = displayW / frameWidth;
  // Full sprite sheet width in CSS px
  const sheetDisplayW = frameWidth * frameCount * scale;

  useEffect(() => {
    if (!playing) return;

    const ms = Math.max(16, Math.round(1000 / fps));
    intervalRef.current = setInterval(() => {
      setFrame((prev) => {
        const next = prev + 1;
        if (next >= frameCount) {
          if (!loop) {
            setPlaying(false);
            return frameCount - 1;
          }
          return 0;
        }
        return next;
      });
    }, ms);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, fps, frameCount, loop]);

  const bgOffsetX = -(frame * displayW);

  return (
    <div className={`flex flex-col items-center gap-2 ${className}`}>
      {/* Frame display */}
      <div
        title={`Frame ${frame + 1}/${frameCount}`}
        style={{
          width: displayW,
          height: displayH,
          backgroundImage: `url(${spriteUrl})`,
          backgroundSize: `${sheetDisplayW}px ${displayH}px`,
          backgroundPosition: `${bgOffsetX}px 0px`,
          backgroundRepeat: "no-repeat",
          imageRendering: "pixelated",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 8,
          backgroundColor: "rgba(0,0,0,0.3)",
          cursor: "pointer",
        }}
        onClick={() => setPlaying((p) => !p)}
      />

      {/* Controls */}
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <button
          onClick={() => { setFrame(0); setPlaying(true); }}
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
        </span>
      </div>

      <p className="text-xs text-gray-600">Click en la imagen para pausar/reanudar</p>
    </div>
  );
}
