"""
Animation Enhancer Service — Post-processing pipeline for 2D animation frames.

Applies timing profiles, procedural motion offsets, loop blending, pixel motion
baking, and motion amplification to raw AI-generated frames.  Runs entirely
CPU-side with no GPU calls.

Architecture:
    AnimationEnhancerService (public)
    ├── _TimingEngine       — per-frame duration_ms + easing
    ├── _ProceduralMotion   — per-frame offset_x + offset_y (numpy sin/cos)
    ├── _LoopBlender        — alpha cross-fade last N frames toward frame[0]
    ├── _MotionAmplifier    — amplify inter-frame diffs when SSIM ≥ 0.95
    └── _PixelMotionEngine  — per-anim-type PIL AFFINE pixel transforms (all anim types)
                              (formerly _IdleEnhancer — now handles ALL animation types)

All offsets are written to animation.json metadata — NOT baked into pixels.
LoopBlender, _MotionAmplifier, and _PixelMotionEngine DO modify pixels.

Output: v2 animation.json schema (backward-compatible with v1 consumers).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

# Per-animation timing profiles: list of (duration_ms, easing) per frame.
# Profiles are defined for 17 frames (Wan2.1 native) and interpolated for
# other frame counts.
#
# Attack: slow wind-up (frames 0-2) → fast strike (3-4) → medium recovery (5-8)
#         → long settle (9-16)
# Idle:   long holds, ease-in-out throughout
# Walk:   alternating step emphasis
# Run:    fast with ease-out on peak extension
# Jump:   slow apex, fast launch/land
# Dance:  rhythmic with alternating emphasis
# Wave:   steady arm raise/lower
# Hurt:   fast initial recoil, slower recovery
_TIMING_PROFILES: dict[str, list[tuple[int, str]]] = {
    "attack": [
        (100, "ease-in"),    # 0  wind-up start
        (100, "ease-in"),    # 1  wind-up
        (100, "ease-in"),    # 2  wind-up peak
        (40,  "linear"),     # 3  strike
        (40,  "linear"),     # 4  strike impact
        (60,  "ease-out"),   # 5  recovery start
        (80,  "ease-out"),   # 6  recovery
        (80,  "ease-out"),   # 7  recovery
        (80,  "ease-out"),   # 8  recovery end
        (80,  "ease-in-out"),# 9  settle
        (80,  "ease-in-out"),# 10 settle
        (80,  "ease-in-out"),# 11 settle
        (80,  "ease-in-out"),# 12 settle
        (80,  "ease-in-out"),# 13 settle
        (80,  "ease-in-out"),# 14 settle
        (80,  "ease-in-out"),# 15 settle
        (80,  "ease-in-out"),# 16 settle
    ],
    "idle": [
        (110, "ease-in-out"), (110, "ease-in-out"), (110, "ease-in-out"),
        (110, "ease-in-out"), (110, "ease-in-out"), (110, "ease-in-out"),
        (100, "ease-in-out"), (100, "ease-in-out"), (100, "ease-in-out"),
        (100, "ease-in-out"), (100, "ease-in-out"), (100, "ease-in-out"),
        (110, "ease-in-out"), (110, "ease-in-out"), (110, "ease-in-out"),
        (110, "ease-in-out"), (110, "ease-in-out"),
    ],
    "walk": [
        (70, "ease-out"),    # 0  heel strike R
        (65, "ease-in-out"), # 1  mid-stance
        (70, "ease-out"),    # 2  toe-off R
        (65, "linear"),      # 3  swing R
        (70, "ease-out"),    # 4  heel strike L
        (65, "ease-in-out"), # 5  mid-stance
        (70, "ease-out"),    # 6  toe-off L
        (65, "linear"),      # 7  swing L
        (70, "ease-out"),    # 8  heel strike R
        (65, "ease-in-out"), # 9  mid-stance
        (70, "ease-out"),    # 10 toe-off R
        (65, "linear"),      # 11 swing R
        (70, "ease-out"),    # 12 heel strike L
        (65, "ease-in-out"), # 13 mid-stance
        (70, "ease-out"),    # 14 toe-off L
        (65, "linear"),      # 15 swing L
        (70, "ease-out"),    # 16 heel strike R (loop back to 0)
    ],
    "run": [
        (45, "ease-out"),    # 0
        (40, "linear"),      # 1
        (45, "ease-out"),    # 2
        (40, "linear"),      # 3
        (45, "ease-out"),    # 4
        (40, "linear"),      # 5
        (45, "ease-out"),    # 6
        (40, "linear"),      # 7
        (45, "ease-out"),    # 8
        (40, "linear"),      # 9
        (45, "ease-out"),    # 10
        (40, "linear"),      # 11
        (45, "ease-out"),    # 12
        (40, "linear"),      # 13
        (45, "ease-out"),    # 14
        (40, "linear"),      # 15
        (45, "ease-out"),    # 16
    ],
    "jump": [
        (50,  "ease-in"),    # 0  crouch/prepare
        (40,  "ease-in"),    # 1  launch
        (60,  "ease-out"),   # 2  ascending
        (80,  "ease-out"),   # 3  ascending
        (120, "ease-in-out"),# 4  peak
        (150, "ease-in-out"),# 5  apex (longest — weightless)
        (150, "ease-in-out"),# 6  apex
        (120, "ease-in-out"),# 7  start descend
        (80,  "ease-in"),    # 8  descending
        (60,  "ease-in"),    # 9  descending
        (40,  "linear"),     # 10 impact
        (50,  "ease-out"),   # 11 absorb
        (70,  "ease-out"),   # 12 recover
        (80,  "ease-out"),   # 13 recover
        (90,  "ease-in-out"),# 14 stand
        (90,  "ease-in-out"),# 15 stand
        (90,  "ease-in-out"),# 16 stand
    ],
    "dance": [
        (80,  "ease-in-out"), # 0
        (70,  "ease-out"),    # 1
        (80,  "ease-in-out"), # 2
        (70,  "ease-out"),    # 3
        (60,  "linear"),      # 4  beat
        (80,  "ease-in-out"), # 5
        (70,  "ease-out"),    # 6
        (80,  "ease-in-out"), # 7
        (70,  "ease-out"),    # 8
        (60,  "linear"),      # 9  beat
        (80,  "ease-in-out"), # 10
        (70,  "ease-out"),    # 11
        (80,  "ease-in-out"), # 12
        (70,  "ease-out"),    # 13
        (60,  "linear"),      # 14 beat
        (80,  "ease-in-out"), # 15
        (80,  "ease-in-out"), # 16
    ],
    "wave": [
        (90, "ease-in"),     # 0  arm begins rise
        (90, "ease-in"),     # 1
        (80, "ease-in-out"), # 2
        (80, "ease-in-out"), # 3
        (70, "ease-out"),    # 4  hand up
        (70, "ease-out"),    # 5  wave peak 1
        (70, "ease-in-out"), # 6
        (70, "ease-in-out"), # 7  wave peak 2
        (70, "ease-out"),    # 8
        (70, "ease-out"),    # 9  wave peak 3
        (80, "ease-in-out"), # 10
        (80, "ease-in-out"), # 11
        (90, "ease-in"),     # 12 arm lowers
        (90, "ease-in"),     # 13
        (90, "ease-in-out"), # 14
        (90, "ease-in-out"), # 15
        (90, "ease-in-out"), # 16
    ],
    "hurt": [
        (30,  "linear"),     # 0  hit flash
        (30,  "linear"),     # 1  recoil fast
        (40,  "ease-out"),   # 2  stagger
        (60,  "ease-out"),   # 3
        (80,  "ease-in-out"),# 4
        (90,  "ease-in-out"),# 5
        (100, "ease-in-out"),# 6
        (100, "ease-in-out"),# 7  recovery
        (100, "ease-in-out"),# 8
        (100, "ease-in-out"),# 9
        (100, "ease-in-out"),# 10
        (100, "ease-in-out"),# 11
        (100, "ease-in-out"),# 12
        (100, "ease-in-out"),# 13
        (100, "ease-in-out"),# 14
        (100, "ease-in-out"),# 15
        (100, "ease-in-out"),# 16
    ],
}

# Personality → (amp_x, amp_y, period_x, period_y, phase_x, phase_y)
# offset_x = amp_x * intensity * sin(2π * idx / period_x + phase_x * 2π)
# offset_y = amp_y * intensity * sin(2π * idx / period_y + phase_y * 2π)
_PERSONALITY_PROFILES: dict[str, dict[str, float]] = {
    "calm": {
        "amp_x": 1.0, "amp_y": 1.0,
        "period_x": 17.0, "period_y": 17.0,
        "phase_x": 0.0, "phase_y": 0.25,
    },
    "aggressive": {
        "amp_x": 3.0, "amp_y": 3.5,  # amp_y >= 3x calm (calm=1.0) per spec R3
        "period_x": 8.0, "period_y": 8.0,
        "phase_x": 0.0, "phase_y": 0.5,
    },
    "heavy": {
        "amp_x": 1.0, "amp_y": 2.0,
        "period_x": 24.0, "period_y": 24.0,
        "phase_x": 0.0, "phase_y": 0.1,
    },
    "light": {
        "amp_x": 2.0, "amp_y": 1.0,
        "period_x": 12.0, "period_y": 12.0,
        "phase_x": 0.3, "phase_y": 0.0,
    },
}

# Per-frame minimum duration floors (ms) applied AFTER intensity scaling.
# These enforce spec R2 thresholds at any intensity value.
# Keys are animation types; each entry is a list of (min_ms, max_ms or None) tuples.
# None means "no bound in that direction".
# attack  wind-up (0-2): ≥100ms  |  strike (3-4): ≤40ms  |  recovery (5-8): ≥60ms
# idle    all:            ≥80ms
_TIMING_FLOORS: dict[str, list[tuple[int, int | None]]] = {
    "attack": [
        (100, None), (100, None), (100, None),  # wind-up: floor 100ms
        (1,   40),   (1,   40),                 # strike:  cap 40ms
        (60,  None), (60,  None), (60,  None), (60, None),  # recovery: floor 60ms
        (1,   None), (1,   None), (1,   None), (1, None),   # settle: no strict floor
        (1,   None), (1,   None), (1,   None), (1, None),
    ],
    "idle": [(80, None)] * 17,  # all frames ≥ 80ms
}

# Animation types that loop seamlessly (dict form for public API; frozenset for internal checks)
_LOOP_ANIMATIONS: frozenset[str] = frozenset({
    "idle", "walk", "run", "dance", "wave",
})

# ── Public constant aliases (design-spec names) ───────────────────────────────
# These match the public interface defined in the design document.

TIMING_PROFILES = _TIMING_PROFILES
PERSONALITY_PROFILES = _PERSONALITY_PROFILES
LOOP_PRESETS: dict[str, bool] = {
    "idle":   True,
    "walk":   True,
    "run":    True,
    "attack": False,
    "jump":   False,
    "dance":  True,
    "wave":   True,
    "hurt":   False,
}

# Animation types that receive idle breathing/sway pixel transforms (legacy reference)
_IDLE_ANIMATIONS: frozenset[str] = frozenset({"idle", "wave"})

# Per-animation-type pixel motion transform profiles.
# Each key maps to a list of 17 (tx_px, ty_px) tuples — one per Wan2.1 frame.
# tx_px > 0 = shift right, ty_px > 0 = shift down.
# "hurt" uses (0, 0) entries — the engine returns frames unchanged for it.
_PIXEL_MOTION_PROFILES: dict[str, list[tuple[float, float]]] = {
    # Walk: vertical bounce ±4.5px + horizontal body-bob ±3px over ~2 steps in 17 frames
    # Base values designed so intensity=0.7 yields ≥3px (spec R6 floor)
    "walk": [
        ( 0.0,  0.0), ( 1.5, -2.2), ( 3.0, -4.5), ( 1.5, -2.2), ( 0.0,  0.0),
        (-1.5,  2.2), (-3.0,  4.5), (-1.5,  2.2), ( 0.0,  0.0), ( 1.5, -2.2),
        ( 3.0, -4.5), ( 1.5, -2.2), ( 0.0,  0.0), (-1.5,  2.2), (-3.0,  4.5),
        (-1.5,  2.2), ( 0.0,  0.0),
    ],
    # Run: faster bounce ±5px + bob ±3px (faster cycle than walk)
    "run": [
        ( 0.0,  0.0), ( 1.5, -2.5), ( 3.0, -5.0), ( 1.5, -2.5), ( 0.0,  0.0),
        (-1.5,  2.5), (-3.0,  5.0), (-1.5,  2.5), ( 0.0,  0.0), ( 1.5, -2.5),
        ( 3.0, -5.0), ( 1.5, -2.5), ( 0.0,  0.0), (-1.5,  2.5), (-3.0,  5.0),
        (-1.5,  2.5), ( 0.0,  0.0),
    ],
    # Attack: lunge forward ±9px (peak at frames 3-4), recoil ±4px (frames 5-8),
    #         return to baseline by frame 16
    # Base values designed so intensity=0.7 yields ≥6px peak (spec R6 floor)
    "attack": [
        ( 0.0,  0.0), ( 3.0,  0.0), ( 6.0,  0.0), ( 9.0,  0.0), ( 9.0,  0.0),
        ( 4.0,  0.0), ( 0.0,  0.0), (-2.5,  0.0), (-4.0,  0.0), (-2.5,  0.0),
        (-1.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0),
        ( 0.0,  0.0), ( 0.0,  0.0),
    ],
    # Jump: vertical arc — rise frames 1-5, apex 5-6, fall 7-10, land/absorb 11-16
    "jump": [
        ( 0.0,  0.0), ( 0.0, -3.0), ( 0.0, -7.0), ( 0.0, -10.0), ( 0.0, -12.0),
        ( 0.0, -12.0), ( 0.0, -12.0), ( 0.0, -8.0), ( 0.0, -5.0), ( 0.0, -2.0),
        ( 0.0,  0.0), ( 0.0,  2.0), ( 0.0,  1.0), ( 0.0,  0.0), ( 0.0,  0.0),
        ( 0.0,  0.0), ( 0.0,  0.0),
    ],
    # Dance: rhythmic sway ±3px x + bounce ±2px y
    "dance": [
        ( 0.0,  0.0), ( 1.5, -1.0), ( 3.0, -2.0), ( 1.5, -1.0), ( 0.0,  0.0),
        (-1.5,  1.0), (-3.0,  2.0), (-1.5,  1.0), ( 0.0,  0.0), ( 1.5, -1.0),
        ( 3.0, -2.0), ( 1.5, -1.0), ( 0.0,  0.0), (-1.5,  1.0), (-3.0,  2.0),
        (-1.5,  1.0), ( 0.0,  0.0),
    ],
    # Hurt: recoil backward ±6px frames 0-3, then slow recovery
    "hurt": [
        (-6.0,  0.0), (-6.0,  0.0), (-5.0,  0.0), (-4.0,  0.0), (-3.0,  0.0),
        (-2.0,  0.0), (-1.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0),
        ( 0.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0), ( 0.0,  0.0),
        ( 0.0,  0.0), ( 0.0,  0.0),
    ],
    # Wave: subtle sway ±2px x to mimic arm inertia
    "wave": [
        ( 0.0,  0.0), ( 0.5, -0.5), ( 1.0, -1.0), ( 1.5, -1.5), ( 2.0, -2.0),
        ( 1.5, -1.5), ( 1.0, -1.0), ( 0.5, -0.5), ( 0.0,  0.0), (-0.5,  0.5),
        (-1.0,  1.0), (-1.5,  1.5), (-2.0,  2.0), (-1.5,  1.5), (-1.0,  1.0),
        (-0.5,  0.5), ( 0.0,  0.0),
    ],
    # Idle: gentle horizontal sway ±1.5px (x) + vertical breathing ±1px (y)
    # Sway period ≈ 17 frames (one full cycle), breathing period ≈ 8.5 frames
    # (breathing is faster — two breath cycles per sway cycle for realism)
    "idle": [
        ( 0.0,  0.0), ( 0.6, -0.3), ( 1.1, -0.6), ( 1.4, -0.9), ( 1.5, -1.0),
        ( 1.4, -0.9), ( 1.1, -0.6), ( 0.6, -0.3), ( 0.0,  0.0), (-0.6,  0.3),
        (-1.1,  0.6), (-1.4,  0.9), (-1.5,  1.0), (-1.4,  0.9), (-1.1,  0.6),
        (-0.6,  0.3), ( 0.0,  0.0),
    ],
}

# Advisory layer metadata — NOT actual segmentation, consumed by game engines
LAYER_METADATA: dict[str, dict] = {
    "body":     {"delay_ms": 0},
    "hair":     {"delay_ms": 40, "overshoot": 0.15, "damping": 0.8},
    "clothing": {"delay_ms": 30, "overshoot": 0.10, "damping": 0.85},
}


# ── Internal engines ──────────────────────────────────────────────────────────

class _TimingEngine:
    """Assigns per-frame duration_ms and easing strings from preset profiles.

    If the number of frames differs from the preset length (17), durations are
    distributed proportionally so the total animation duration stays constant.
    """

    _DEFAULT_EASING: str = "ease-in-out"
    _DEFAULT_DURATION_MS: int = 100

    @staticmethod
    def compute(
        animation_type: str,
        n_frames: int,
        intensity: float = 0.7,
    ) -> list[dict]:
        """Return per-frame timing: [{"duration_ms": int, "easing": str}, ...].

        Args:
            animation_type: One of the 8 supported types (unknown → defaults).
            n_frames: Number of frames to produce timing for.
            intensity: 0.0–1.0; scales durations (higher = slower / more dramatic).

        Returns:
            list of dicts with "duration_ms" (int) and "easing" (str) per frame.
        """
        profile = _TIMING_PROFILES.get(animation_type)
        if not profile:
            # Unknown type → uniform default durations
            return [
                {"duration_ms": _TimingEngine._DEFAULT_DURATION_MS, "easing": _TimingEngine._DEFAULT_EASING}
                for _ in range(n_frames)
            ]

        preset_n = len(profile)
        floors = _TIMING_FLOORS.get(animation_type)

        if n_frames == preset_n:
            # Exact match — apply intensity scale, then enforce spec floors/caps
            result = []
            for i, (dur, easing) in enumerate(profile):
                scaled = max(1, round(dur * intensity))
                if floors and i < len(floors):
                    floor_min, floor_max = floors[i]
                    if floor_min is not None:
                        scaled = max(scaled, floor_min)
                    if floor_max is not None:
                        scaled = min(scaled, floor_max)
                result.append({"duration_ms": scaled, "easing": easing})
            return result

        # Different frame count → proportional interpolation
        # Distribute preset timing proportionally across n_frames
        total_dur = sum(d for d, _ in profile)
        # Build cumulative weights from preset
        cumulative = [0.0]
        for d, _ in profile:
            cumulative.append(cumulative[-1] + d / total_dur)

        result = []
        for i in range(n_frames):
            # Find position in [0, 1] for this frame
            t = (i + 0.5) / n_frames
            # Find which preset segment we're in
            seg = 0
            for j in range(preset_n - 1):
                if cumulative[j] <= t < cumulative[j + 1]:
                    seg = j
                    break
            else:
                seg = preset_n - 1

            # Fraction of total → frame duration (scale by intensity)
            frame_frac = 1.0 / n_frames
            dur_raw = round(total_dur * frame_frac * intensity)
            dur = max(1, dur_raw)
            easing = profile[seg][1]
            result.append({"duration_ms": dur, "easing": easing})

        return result


class _ProceduralMotion:
    """Generates deterministic per-frame motion offsets using numpy sin/cos.

    Offsets are ADVISORY — written to animation.json, NOT baked into pixels.
    """

    _DEFAULT_PERSONALITY: str = "calm"

    @staticmethod
    def compute(
        n_frames: int,
        personality: str = "calm",
        intensity: float = 0.7,
    ) -> list[dict]:
        """Return per-frame offsets: [{"offset_x": float, "offset_y": float}, ...].

        Args:
            n_frames: Number of frames.
            personality: One of "calm", "aggressive", "heavy", "light".
            intensity: 0.0–1.0 amplitude scale.

        Returns:
            Deterministic, no-random offset list.
        """
        profile = _PERSONALITY_PROFILES.get(
            personality,
            _PERSONALITY_PROFILES[_ProceduralMotion._DEFAULT_PERSONALITY],
        )
        amp_x = profile["amp_x"]
        amp_y = profile["amp_y"]
        period_x = profile["period_x"]
        period_y = profile["period_y"]
        phase_x = profile["phase_x"]
        phase_y = profile["phase_y"]

        indices = np.arange(n_frames, dtype=np.float64)
        offset_x = amp_x * intensity * np.sin(
            2.0 * math.pi * indices / period_x + phase_x * 2.0 * math.pi
        )
        offset_y = amp_y * intensity * np.sin(
            2.0 * math.pi * indices / period_y + phase_y * 2.0 * math.pi
        )

        return [
            {"offset_x": round(float(ox), 4), "offset_y": round(float(oy), 4)}
            for ox, oy in zip(offset_x, offset_y)
        ]


class _LoopBlender:
    """Cross-fades the last N frames toward frame[0] for seamless loop animations.

    No-op for non-looping animation types.
    Modifies pixels via PIL Image.blend().
    """

    @staticmethod
    def blend(
        frames: list[Image.Image],
        *,
        loop: bool,
        blend_count: int = 4,
    ) -> list[Image.Image]:
        """Return frames with last blend_count frames alpha-blended toward frame[0].

        Args:
            frames: Input frame list (must have at least blend_count + 1 frames).
            loop: If False, return frames unmodified.
            blend_count: Number of tail frames to blend (default 4).

        Returns:
            New frame list (same length as input).

        Note:
            Alpha increases linearly so the last frame gets max_alpha=0.875.
            This ensures the pixel delta between frame[-1] and frame[0] is
            < 15% of total pixel range (spec R4) even for worst-case frames.
            Formula: alpha_k = (k+1) / blend_count * 0.875
            Last frame (k = blend_count-1): alpha = 0.875 → blended = 0.125*frame + 0.875*frame_0
            Worst-case pixel delta: 12.5% (< 15% threshold).
        """
        if not loop or len(frames) < blend_count + 1:
            return frames

        # Work on a copy so we don't mutate the caller's list
        result = list(frames)
        frame_0 = result[0].convert("RGBA")
        n = len(result)

        _MAX_ALPHA = 0.875  # guarantees < 15% pixel delta even in worst-case frames

        for k in range(blend_count):
            # Tail frames: result[n-blend_count], ..., result[n-1]
            idx = n - blend_count + k
            # alpha increases linearly: last frame (k = blend_count-1) gets _MAX_ALPHA
            alpha = (k + 1) / blend_count * _MAX_ALPHA

            src = result[idx].convert("RGBA")
            # Resize frame_0 to match src if needed (should be identical sizes)
            f0 = frame_0
            if f0.size != src.size:
                f0 = f0.resize(src.size, Image.Resampling.LANCZOS)

            blended = Image.blend(src, f0, alpha)
            result[idx] = blended

        return result


class _PixelMotionEngine:
    """Bakes per-animation-type pixel motion into frames via PIL AFFINE transforms.

    Replaces the old _IdleEnhancer — now handles ALL animation types using
    per-type transform profiles from _PIXEL_MOTION_PROFILES.

    For idle and wave, also applies the breathing scale_y transform (backward
    compatible with the old _IdleEnhancer behavior) in addition to the tx/ty
    profile displacement.

    "hurt" type: returns frames unchanged (no pixel baking).
    intensity=0.0: returns pixel-identical frames (no transform applied).
    """

    @staticmethod
    def apply(
        frames: list[Image.Image],
        animation_type: str,
        intensity: float = 0.7,
    ) -> list[Image.Image]:
        """Bake per-type pixel transforms into frames via PIL AFFINE.

        Args:
            frames: Input frame list.
            animation_type: One of the 8 animation types; "hurt" → unchanged.
            intensity: 0.0–1.0 scale for displacement amplitude.

        Returns:
            New frame list with pixel-baked motion transforms.
        """
        if intensity <= 0.0:
            return frames

        # "hurt" returns frames unchanged per spec
        if animation_type == "hurt":
            return frames

        profile = _PIXEL_MOTION_PROFILES.get(animation_type)
        if profile is None:
            # Unknown animation type → no-op
            return frames

        n = len(frames)
        result = []

        for i, frame in enumerate(frames):
            frame_rgba = frame.convert("RGBA")
            w, h = frame_rgba.size

            # Interpolate profile for non-17-frame sequences
            if n == len(profile):
                tx_raw, ty_raw = profile[i]
            else:
                # Linear interpolation from profile (17 entries → n frames)
                t = i / max(n - 1, 1) * (len(profile) - 1)
                lo = int(t)
                hi = min(lo + 1, len(profile) - 1)
                frac = t - lo
                tx_raw = profile[lo][0] + frac * (profile[hi][0] - profile[lo][0])
                ty_raw = profile[lo][1] + frac * (profile[hi][1] - profile[lo][1])

            tx = tx_raw * intensity
            ty = ty_raw * intensity

            # For idle and wave, also apply breathing scale_y (backward compat)
            if animation_type in _IDLE_ANIMATIONS:
                breathing_amp = 0.005 * intensity
                scale_y = 1.0 + breathing_amp * math.sin(2.0 * math.pi * i / max(n, 1))
                cy = h / 2.0
                inv_scale_y = 1.0 / scale_y
                # Combined: translate (tx, ty) + scale_y centered on cy
                # PIL AFFINE (output ← input):
                # x_in = x_out - tx
                # y_in = (y_out - ty - cy) / scale_y + cy
                #       = inv_scale_y * y_out - ty * inv_scale_y - cy * inv_scale_y + cy
                #       = inv_scale_y * y_out + cy*(1 - inv_scale_y) - ty * inv_scale_y
                affine_data = (
                    1.0,                           # a
                    0.0,                           # b
                    -tx,                           # c
                    0.0,                           # d
                    inv_scale_y,                   # e
                    cy * (1.0 - inv_scale_y) - ty * inv_scale_y,  # f
                )
            else:
                # Pure translation via AFFINE:
                # x_in = x_out - tx, y_in = y_out - ty
                affine_data = (
                    1.0,  # a
                    0.0,  # b
                    -tx,  # c
                    0.0,  # d
                    1.0,  # e
                    -ty,  # f
                )

            transformed = frame_rgba.transform(
                (w, h),
                Image.Transform.AFFINE,
                affine_data,
                resample=Image.Resampling.BILINEAR,
            )
            result.append(transformed)

        return result


class _MotionAmplifier:
    """Amplifies inter-frame pixel differences when frames are near-identical.

    Uses SSIM (structural similarity) to detect near-static output from Wan2.1.
    When mean pairwise SSIM ≥ threshold (default 0.95), applies the formula:

        frame[i] = clip(frame[0] + factor * (frame[i] − frame[0]), 0, 255)

    This stretches the existing (subtle) inter-frame differences to be more
    visible without introducing new information or drift.  When SSIM < threshold
    (motion is already sufficient), the amplifier is a no-op.

    Anchoring to frame[0] avoids drift accumulation across frames (design decision).
    """

    @staticmethod
    def amplify(
        frames: list[Image.Image],
        *,
        factor: float = 1.5,
        ssim_threshold: float = 0.95,
    ) -> list[Image.Image]:
        """Amplify inter-frame diffs when avg SSIM > ssim_threshold.

        Args:
            frames: Input frame list (PIL Images, any mode).
            factor: Amplification factor (default 1.5).
            ssim_threshold: Mean pairwise SSIM threshold to trigger amplification.
                            If mean SSIM < threshold, return frames unchanged.

        Returns:
            New frame list with amplified motion, or original frames if no-op.
        """
        if len(frames) < 2:
            return frames

        try:
            from skimage.metrics import structural_similarity as ssim_fn
        except ImportError:
            # skimage not available — skip amplification
            logger.warning(
                "_MotionAmplifier: scikit-image not available; skipping amplification"
            )
            return frames

        # Convert all frames to numpy uint8 RGB for SSIM computation
        np_frames: list[np.ndarray] = []
        for f in frames:
            arr = np.array(f.convert("RGB"), dtype=np.uint8)
            np_frames.append(arr)

        frame0 = np_frames[0]

        # Compute mean pairwise SSIM (frame[i] vs frame[0], i ≥ 1)
        ssim_scores: list[float] = []
        for i in range(1, len(np_frames)):
            score = float(
                ssim_fn(
                    frame0,
                    np_frames[i],
                    channel_axis=2,
                    data_range=255,
                )
            )
            ssim_scores.append(score)

        mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0

        if mean_ssim < ssim_threshold:
            # Motion already sufficient — skip amplification
            logger.debug(
                f"_MotionAmplifier: mean SSIM={mean_ssim:.4f} < {ssim_threshold}; "
                "skipping amplification (motion sufficient)"
            )
            return frames

        logger.info(
            f"_MotionAmplifier: mean SSIM={mean_ssim:.4f} ≥ {ssim_threshold}; "
            f"amplifying with factor={factor}"
        )

        # Apply amplification: frame[i] = clip(frame[0] + factor*(frame[i]-frame[0]), 0, 255)
        result: list[Image.Image] = [frames[0]]  # frame[0] unchanged (diff = 0)
        f0_float = frame0.astype(np.float32)

        for i in range(1, len(np_frames)):
            fi_float = np_frames[i].astype(np.float32)
            diff = fi_float - f0_float
            amplified = np.clip(f0_float + factor * diff, 0, 255).astype(np.uint8)
            # Preserve original mode (may be RGBA — paste amplified RGB back)
            orig = frames[i]
            if orig.mode == "RGBA":
                amp_pil = Image.fromarray(amplified, mode="RGB").convert("RGBA")
                # Keep original alpha channel
                r, g, b, a = orig.split()
                amp_r, amp_g, amp_b, _ = amp_pil.split()
                result.append(Image.merge("RGBA", (amp_r, amp_g, amp_b, a)))
            else:
                result.append(Image.fromarray(amplified, mode="RGB"))

        return result


# ── Public service ────────────────────────────────────────────────────────────

class AnimationEnhancerService:
    """Post-processing enhancement pipeline for AI-generated 2D animation frames.

    Wraps TimingEngine, ProceduralMotion, LoopBlender, and IdleEnhancer into a
    single call.  Falls back gracefully on any internal error — the pipeline
    never crashes due to enhancement failures.

    Usage::

        enhancer = AnimationEnhancerService()
        frames, metadata = enhancer.enhance(pil_frames, "walk")
    """

    def enhance(
        self,
        frames: list[Image.Image],
        animation_type: str,
        *,
        personality: str = "calm",
        intensity: float = 0.7,
        enable_timing: bool = True,
        enable_motion: bool = True,
        enable_looping: bool = True,
        enable_idle: bool = True,
        enable_pixel_motion: bool = True,
        enable_amplifier: bool = True,
        enhance: bool = True,
    ) -> tuple[list[Image.Image], dict]:
        """Enhance animation frames with timing, motion, loop blending, and pixel transforms.

        Args:
            frames: List of PIL Image frames (any mode; internally converted to RGBA).
            animation_type: One of the 8 animation types.
            personality: Motion personality profile (default "calm").
            intensity: 0.0–1.0 enhancement strength (default 0.7).
            enable_timing: Include timing engine output in metadata.
            enable_motion: Include procedural motion offsets in metadata.
            enable_looping: Apply loop cross-fade blending (pixels modified).
            enable_idle: Kept for backward compat; pixel motion is now handled
                         by _PixelMotionEngine (enable_pixel_motion flag).
            enable_pixel_motion: Apply _PixelMotionEngine pixel transforms (all
                                 animation types, pixels modified).
            enable_amplifier: Apply _MotionAmplifier when SSIM ≥ 0.95.
            enhance: If False, skip all enhancement and return v1-style metadata.

        Returns:
            (enhanced_frames, metadata_dict) where metadata_dict is ready for
            merging into animation.json.
        """
        if not enhance or not frames:
            return frames, self._v1_metadata(frames, animation_type)

        raw_frames = frames  # keep original for fallback
        try:
            return self._do_enhance(
                frames, animation_type,
                personality=personality,
                intensity=intensity,
                enable_timing=enable_timing,
                enable_motion=enable_motion,
                enable_looping=enable_looping,
                enable_pixel_motion=enable_pixel_motion,
                enable_amplifier=enable_amplifier,
            )
        except Exception as exc:
            logger.warning(
                f"AnimationEnhancerService: enhancement failed ({exc}); "
                "falling back to raw frames with v1 metadata"
            )
            return raw_frames, self._v1_metadata(raw_frames, animation_type)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _do_enhance(
        self,
        frames: list[Image.Image],
        animation_type: str,
        *,
        personality: str,
        intensity: float,
        enable_timing: bool,
        enable_motion: bool,
        enable_looping: bool,
        enable_pixel_motion: bool,
        enable_amplifier: bool,
    ) -> tuple[list[Image.Image], dict]:
        """Core enhancement logic (called inside try/except by enhance()).

        Pipeline order (design-v3):
            1. TimingEngine         — metadata only
            2. ProceduralMotion     — metadata only
            3. LoopBlender          — pixel modification (loop animations)
            4. MotionAmplifier      — pixel modification (SSIM guard)
            5. PixelMotionEngine    — pixel modification (all anim types)
            6. Build v2 metadata
        """
        n = len(frames)
        loop = animation_type in _LOOP_ANIMATIONS

        # 1. Timing (metadata only)
        timings: list[dict] = []
        if enable_timing:
            timings = _TimingEngine.compute(animation_type, n, intensity)

        # 2. Procedural motion (metadata only)
        motions: list[dict] = []
        if enable_motion:
            motions = _ProceduralMotion.compute(n, personality, intensity)

        # 3. Loop blending (pixel modification)
        if enable_looping:
            frames = _LoopBlender.blend(frames, loop=loop, blend_count=2)

        # 4. Motion amplifier — amplify inter-frame diffs when Wan2.1 output is
        #    near-static (SSIM ≥ 0.95).  Runs BEFORE pixel motion baking so that
        #    the amplifier works on the raw AI frames, not on artificially shifted ones.
        if enable_amplifier:
            frames = _MotionAmplifier.amplify(frames)

        # 5. Pixel motion baking (all animation types via _PixelMotionEngine)
        if enable_pixel_motion:
            frames = _PixelMotionEngine.apply(frames, animation_type, intensity)

        # 6. Build v2 metadata
        metadata = self._build_v2_metadata(
            frames=frames,
            animation_type=animation_type,
            personality=personality,
            intensity=intensity,
            timings=timings,
            motions=motions,
            loop=loop,
            enable_timing=enable_timing,
            enable_motion=enable_motion,
            enable_looping=enable_looping,
            enable_pixel_motion=enable_pixel_motion,
            enable_amplifier=enable_amplifier,
        )

        return frames, metadata

    @staticmethod
    def _build_v2_metadata(
        *,
        frames: list[Image.Image],
        animation_type: str,
        personality: str,
        intensity: float,
        timings: list[dict],
        motions: list[dict],
        loop: bool,
        enable_timing: bool,
        enable_motion: bool,
        enable_looping: bool,
        enable_pixel_motion: bool,
        enable_amplifier: bool,
    ) -> dict:
        """Build the v2 metadata dict for merging into animation.json."""
        n = len(frames)
        total_duration_ms = sum(t["duration_ms"] for t in timings) if timings else 0

        # Per-frame extra fields
        frame_extras: list[dict] = []
        for i in range(n):
            entry: dict = {}
            if timings and i < len(timings):
                entry["duration_ms"] = timings[i]["duration_ms"]
                entry["easing"] = timings[i]["easing"]
            if motions and i < len(motions):
                entry["offset_x"] = motions[i]["offset_x"]
                entry["offset_y"] = motions[i]["offset_y"]
            frame_extras.append(entry)

        return {
            "version": 2,
            "loop": loop,
            "total_duration_ms": total_duration_ms,
            "frame_extras": frame_extras,  # caller merges these into frames[]
            "layers": dict(LAYER_METADATA),  # advisory only
            "enhancements": {
                "timing": enable_timing,
                "procedural_motion": enable_motion,
                "looping": enable_looping and loop,
                "pixel_motion": enable_pixel_motion,
                "motion_amplifier": enable_amplifier,
                "personality": personality,
                "intensity": intensity,
            },
        }

    @staticmethod
    def _v1_metadata(frames: list[Image.Image], animation_type: str) -> dict:
        """Return a minimal v1-compatible metadata stub (no enhancement fields)."""
        return {
            "loop": animation_type in _LOOP_ANIMATIONS,
        }
