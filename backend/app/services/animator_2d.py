"""
2D Animator Service.

Generates sprite-sheet animations from a 2D character model (model.json + part PNGs).

Animation keyframes:
    Each keyframe contains: time, tx, ty, rot, scale_x, scale_y, skew_x, opacity
    - tx, ty: translation in pixels
    - rot: rotation in degrees (CCW)
    - scale_x, scale_y: simulate foreshortening / perspective depth
    - skew_x: horizontal shear to simulate body twist / perspective rotation
    - opacity: for fade/flash effects (0.0 = invisible, 1.0 = opaque)

Easing: ease-in-out (smoothstep) between adjacent keyframes.

Output:
    - sprite_sheet.png  — horizontal strip of FRAME_SIZE × FRAME_SIZE frames
    - animation.json    — metadata for game engines
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Keyframe type ─────────────────────────────────────────────

class Keyframe(NamedTuple):
    time: float          # seconds
    tx: float            # translate X pixels
    ty: float            # translate Y pixels
    rot: float           # rotation degrees (CCW)
    scale_x: float = 1.0 # X scale: <1 = narrower (foreshortening), >1 = wider
    scale_y: float = 1.0 # Y scale: <1 = squash, >1 = stretch
    skew_x: float = 0.0  # horizontal shear for perspective rotation / body twist
    opacity: float = 1.0 # 0.0–1.0 transparency


def _KF(
    time: float,
    tx: float = 0.0,
    ty: float = 0.0,
    rot: float = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
    skew: float = 0.0,
    opacity: float = 1.0,
) -> Keyframe:
    """Helper shorthand for building Keyframes with named defaults."""
    return Keyframe(time=time, tx=tx, ty=ty, rot=rot,
                    scale_x=sx, scale_y=sy, skew_x=skew, opacity=opacity)


# ── Animation presets ────────────────────────────────────────
#
# Each part has a list of Keyframes.  Parts not listed use zero keyframes.
# time 0.0 and time == duration should represent the same pose for loops.
#
# Depth notes:
#   scale_x < 1 → part turned away (narrower, foreshortening)
#   scale_x > 1 → part turned toward camera (wider / lunging forward)
#   skew_x > 0  → right side compresses (left turn / lean)
#   skew_x < 0  → left side compresses (right turn / lean)
#   scale_y < 1 → part squashed (impact, crouch, or angled view)
#   scale_y > 1 → part stretched (jump, lean back)

ANIMATION_PRESETS: dict[str, dict] = {

    # ── Idle: gentle breathing, slight head turn, depth oscillation ────────
    "idle": {
        "duration": 2.0, "fps": 60, "loop": True,
        "parts": {
            # Head: subtle tilt + tiny scale_x oscillation (slight head turn)
            "head": [
                _KF(0.0,  tx=0,  ty=0,   rot=0,    sx=1.00, sy=1.00, skew=0.000),
                _KF(0.5,  tx=0,  ty=-2,  rot=-1.5, sx=0.99, sy=1.00, skew=-0.005),
                _KF(1.0,  tx=0,  ty=-3,  rot=0,    sx=1.00, sy=1.00, skew=0.000),
                _KF(1.5,  tx=0,  ty=-2,  rot=1.5,  sx=1.01, sy=1.00, skew=0.005),
                _KF(2.0,  tx=0,  ty=0,   rot=0,    sx=1.00, sy=1.00, skew=0.000),
            ],
            # Hair: secondary motion (handled by spring physics in _render_frame)
            "hair": [
                _KF(0.0,  tx=0,  ty=0,   rot=0,    sx=1.00, sy=1.00),
                _KF(1.0,  tx=0,  ty=-2,  rot=1.0,  sx=1.00, sy=1.00),
                _KF(2.0,  tx=0,  ty=0,   rot=0,    sx=1.00, sy=1.00),
            ],
            # Torso: breathing = scale_y oscillation (chest expanding)
            "torso": [
                _KF(0.0,  tx=0,  ty=0,   rot=0, sx=1.00, sy=1.00, skew=0.000),
                _KF(1.0,  tx=0,  ty=-2,  rot=0, sx=1.00, sy=1.02, skew=0.000),
                _KF(2.0,  tx=0,  ty=0,   rot=0, sx=1.00, sy=1.00, skew=0.000),
            ],
            # Arms: subtle swing, scale_x varies (arm slightly toward/away from camera)
            "arm_left": [
                _KF(0.0,  tx=0,  ty=0,  rot=2,  sx=1.00, sy=1.00),
                _KF(1.0,  tx=0,  ty=-2, rot=3,  sx=0.98, sy=1.00),
                _KF(2.0,  tx=0,  ty=0,  rot=2,  sx=1.00, sy=1.00),
            ],
            "arm_right": [
                _KF(0.0,  tx=0,  ty=0,  rot=-2, sx=1.00, sy=1.00),
                _KF(1.0,  tx=0,  ty=-2, rot=-3, sx=1.02, sy=1.00),
                _KF(2.0,  tx=0,  ty=0,  rot=-2, sx=1.00, sy=1.00),
            ],
            # Legs: barely move, very subtle scale_y
            "leg_left": [
                _KF(0.0,  tx=0,  ty=0,  rot=0, sx=1.00, sy=1.00),
                _KF(1.0,  tx=0,  ty=-1, rot=0, sx=1.00, sy=1.01),
                _KF(2.0,  tx=0,  ty=0,  rot=0, sx=1.00, sy=1.00),
            ],
            "leg_right": [
                _KF(0.0,  tx=0,  ty=0,  rot=0, sx=1.00, sy=1.00),
                _KF(1.0,  tx=0,  ty=-1, rot=0, sx=1.00, sy=1.01),
                _KF(2.0,  tx=0,  ty=0,  rot=0, sx=1.00, sy=1.00),
            ],
        },
    },

    # ── Walk: body twist, foreshortening legs/arms, parallax depth ────────
    "walk": {
        "duration": 0.8, "fps": 60, "loop": True,
        "parts": {
            # Head: counter-sway (stays relatively still vs body twist)
            "head": [
                _KF(0.0,  tx=0,  ty=0,  rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.2,  tx=0,  ty=-2, rot=1,  sx=1.00, sy=1.00, skew=0.005),
                _KF(0.4,  tx=0,  ty=0,  rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.6,  tx=0,  ty=-2, rot=-1, sx=1.00, sy=1.00, skew=-0.005),
                _KF(0.8,  tx=0,  ty=0,  rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            "hair": [
                _KF(0.0,  tx=0,  ty=0,  rot=0,  sx=1.00, sy=1.00),
                _KF(0.4,  tx=1,  ty=0,  rot=2,  sx=1.00, sy=1.00),
                _KF(0.8,  tx=0,  ty=0,  rot=0,  sx=1.00, sy=1.00),
            ],
            # Torso: body twists with each step — core of the 3D illusion
            "torso": [
                _KF(0.0,  tx=0,  ty=0,  rot=2,  sx=0.98, sy=1.00, skew=0.015),
                _KF(0.4,  tx=0,  ty=0,  rot=-2, sx=1.02, sy=1.00, skew=-0.015),
                _KF(0.8,  tx=0,  ty=0,  rot=2,  sx=0.98, sy=1.00, skew=0.015),
            ],
            # Arms: opposite swing to legs; scale_x simulates depth (arm toward/away camera)
            "arm_left": [
                _KF(0.0,  tx=0,  ty=0,  rot=-28, sx=0.92, sy=1.00),  # arm swings back = narrower
                _KF(0.4,  tx=0,  ty=0,  rot=28,  sx=1.08, sy=1.00),  # arm swings forward = wider
                _KF(0.8,  tx=0,  ty=0,  rot=-28, sx=0.92, sy=1.00),
            ],
            "arm_right": [
                _KF(0.0,  tx=0,  ty=0,  rot=28,  sx=1.08, sy=1.00),  # forward = wider
                _KF(0.4,  tx=0,  ty=0,  rot=-28, sx=0.92, sy=1.00),  # back = narrower
                _KF(0.8,  tx=0,  ty=0,  rot=28,  sx=1.08, sy=1.00),
            ],
            # Legs: large rotation + foreshortening (scale_y) as leg swings + skew for perspective
            "leg_left": [
                _KF(0.0,  tx=0,  ty=0,  rot=-22, sx=1.00, sy=0.96, skew=0.010),
                _KF(0.2,  tx=0,  ty=-8, rot=-32, sx=0.97, sy=0.92, skew=0.015),  # knee up / foreshortening
                _KF(0.4,  tx=0,  ty=0,  rot=22,  sx=1.00, sy=1.04, skew=-0.010), # leg extends = stretch
                _KF(0.6,  tx=0,  ty=0,  rot=10,  sx=1.00, sy=1.00, skew=-0.005),
                _KF(0.8,  tx=0,  ty=0,  rot=-22, sx=1.00, sy=0.96, skew=0.010),
            ],
            "leg_right": [
                _KF(0.0,  tx=0,  ty=0,  rot=22,  sx=1.00, sy=1.04, skew=-0.010),
                _KF(0.2,  tx=0,  ty=0,  rot=10,  sx=1.00, sy=1.00, skew=-0.005),
                _KF(0.4,  tx=0,  ty=0,  rot=-22, sx=1.00, sy=0.96, skew=0.010),
                _KF(0.6,  tx=0,  ty=-8, rot=-32, sx=0.97, sy=0.92, skew=0.015),
                _KF(0.8,  tx=0,  ty=0,  rot=22,  sx=1.00, sy=1.04, skew=-0.010),
            ],
        },
    },

    # ── Run: extreme twist, forward lean, heavy foreshortening ────────────
    "run": {
        "duration": 0.5, "fps": 60, "loop": True,
        "parts": {
            "head": [
                _KF(0.0,  tx=0, ty=-2, rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.25, tx=0, ty=2,  rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.5,  tx=0, ty=-2, rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            "hair": [
                _KF(0.0,  tx=2,  ty=-2, rot=-4, sx=1.00, sy=1.00),
                _KF(0.25, tx=-2, ty=2,  rot=4,  sx=1.00, sy=1.00),
                _KF(0.5,  tx=2,  ty=-2, rot=-4, sx=1.00, sy=1.00),
            ],
            # Torso: forward lean = scale_y compressed, skew for twist
            "torso": [
                _KF(0.0,  tx=0, ty=0, rot=5,  sx=0.97, sy=0.95, skew=0.025),
                _KF(0.25, tx=0, ty=0, rot=-5, sx=1.03, sy=0.95, skew=-0.025),
                _KF(0.5,  tx=0, ty=0, rot=5,  sx=0.97, sy=0.95, skew=0.025),
            ],
            # Arms: bent elbows simulated by scale changes
            "arm_left": [
                _KF(0.0,  tx=0, ty=0, rot=-45, sx=0.85, sy=1.00),
                _KF(0.25, tx=0, ty=0, rot=45,  sx=1.15, sy=1.00),
                _KF(0.5,  tx=0, ty=0, rot=-45, sx=0.85, sy=1.00),
            ],
            "arm_right": [
                _KF(0.0,  tx=0, ty=0, rot=45,  sx=1.15, sy=1.00),
                _KF(0.25, tx=0, ty=0, rot=-45, sx=0.85, sy=1.00),
                _KF(0.5,  tx=0, ty=0, rot=45,  sx=1.15, sy=1.00),
            ],
            # Legs: bigger swing, heavy foreshortening
            "leg_left": [
                _KF(0.0,   tx=0, ty=0,   rot=-40, sx=1.00, sy=0.88, skew=0.020),
                _KF(0.125, tx=0, ty=-18, rot=-55, sx=0.93, sy=0.82, skew=0.025),
                _KF(0.25,  tx=0, ty=0,   rot=40,  sx=1.00, sy=1.08, skew=-0.020),
                _KF(0.5,   tx=0, ty=0,   rot=-40, sx=1.00, sy=0.88, skew=0.020),
            ],
            "leg_right": [
                _KF(0.0,   tx=0, ty=0,   rot=40,  sx=1.00, sy=1.08, skew=-0.020),
                _KF(0.25,  tx=0, ty=0,   rot=-40, sx=1.00, sy=0.88, skew=0.020),
                _KF(0.375, tx=0, ty=-18, rot=-55, sx=0.93, sy=0.82, skew=0.025),
                _KF(0.5,   tx=0, ty=0,   rot=40,  sx=1.00, sy=1.08, skew=-0.020),
            ],
        },
    },

    # ── Attack: wind-up → explosive strike → follow-through ──────────────
    "attack": {
        "duration": 0.9, "fps": 60, "loop": False,
        "parts": {
            "head": [
                _KF(0.0,  tx=0,   ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.15, tx=0,   ty=0, rot=-8, sx=0.98, sy=1.00, skew=-0.010),  # wind-up: turn away
                _KF(0.35, tx=0,   ty=0, rot=8,  sx=1.02, sy=1.00, skew=0.010),   # strike: face forward
                _KF(0.9,  tx=0,   ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            "hair": [
                _KF(0.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(0.15, tx=-4,  ty=0,  rot=-6,  sx=1.00, sy=1.00),
                _KF(0.35, tx=6,   ty=-2, rot=10,  sx=1.00, sy=1.00),
                _KF(0.6,  tx=-2,  ty=0,  rot=-3,  sx=1.00, sy=1.00),
                _KF(0.9,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
            ],
            # Torso: lean back then explosive forward lunge
            "torso": [
                _KF(0.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
                _KF(0.15, tx=-8,  ty=0,  rot=-12, sx=0.95, sy=1.00, skew=-0.050),  # wind-up lean back
                _KF(0.35, tx=8,   ty=0,  rot=20,  sx=1.05, sy=0.97, skew=0.080),   # explosive forward
                _KF(0.55, tx=4,   ty=0,  rot=8,   sx=1.02, sy=1.00, skew=0.030),   # follow through
                _KF(0.9,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
            ],
            # Attacking arm: scale_x goes 0.80 (drawn back) → 1.30 (lunging toward camera)
            "arm_right": [
                _KF(0.0,  tx=0,   ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(0.15, tx=-12, ty=0, rot=-70, sx=0.80, sy=0.95),  # wind-up: arm pulled back (far)
                _KF(0.30, tx=12,  ty=0, rot=80,  sx=1.30, sy=1.00),  # strike: arm lunges forward
                _KF(0.5,  tx=4,   ty=0, rot=30,  sx=1.10, sy=1.00),  # follow through
                _KF(0.9,  tx=0,   ty=0, rot=0,   sx=1.00, sy=1.00),
            ],
            # Support arm: counters the strike
            "arm_left": [
                _KF(0.0,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(0.15, tx=8,  ty=0, rot=30,  sx=1.05, sy=1.00),   # pulls forward slightly
                _KF(0.35, tx=-8, ty=0, rot=-10, sx=0.95, sy=1.00),   # pushes back on follow-through
                _KF(0.9,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
            ],
            "leg_left": [
                _KF(0.0,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(0.15, tx=-6, ty=0, rot=-8, sx=1.00, sy=1.00),
                _KF(0.9,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
            ],
            "leg_right": [
                _KF(0.0,  tx=0, ty=0, rot=0, sx=1.00, sy=1.00),
                _KF(0.15, tx=6, ty=0, rot=8, sx=1.00, sy=1.00),
                _KF(0.9,  tx=0, ty=0, rot=0, sx=1.00, sy=1.00),
            ],
        },
    },

    # ── Jump: crouch anticipation → air stretch → landing squash ─────────
    "jump": {
        "duration": 1.0, "fps": 60, "loop": False,
        "parts": {
            "head": [
                _KF(0.0, tx=0, ty=0,   rot=0, sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=4,   rot=0, sx=1.00, sy=0.95),  # anticipation: crouch
                _KF(0.4, tx=0, ty=-20, rot=0, sx=1.00, sy=1.05),  # air: stretch
                _KF(0.7, tx=0, ty=-18, rot=0, sx=1.00, sy=1.05),
                _KF(0.9, tx=0, ty=4,   rot=0, sx=1.00, sy=0.90),  # landing: squash
                _KF(1.0, tx=0, ty=0,   rot=0, sx=1.00, sy=1.00),
            ],
            "hair": [
                _KF(0.0, tx=0, ty=0,   rot=0,  sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=2,   rot=2,  sx=1.00, sy=1.00),
                _KF(0.4, tx=0, ty=-22, rot=-6, sx=1.00, sy=1.00),
                _KF(0.7, tx=0, ty=-20, rot=-4, sx=1.00, sy=1.00),
                _KF(0.9, tx=0, ty=6,   rot=8,  sx=1.00, sy=1.00),
                _KF(1.0, tx=0, ty=0,   rot=0,  sx=1.00, sy=1.00),
            ],
            # Torso: anticipation squash, air stretch, landing squash
            "torso": [
                _KF(0.0, tx=0, ty=0,   rot=0,  sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=6,   rot=5,  sx=1.02, sy=0.85),  # crouch = squash
                _KF(0.4, tx=0, ty=-20, rot=-5, sx=0.97, sy=1.10),  # air = stretch tall
                _KF(0.7, tx=0, ty=-18, rot=0,  sx=1.00, sy=1.10),
                _KF(0.9, tx=0, ty=6,   rot=0,  sx=1.05, sy=0.85),  # landing = squash wide
                _KF(1.0, tx=0, ty=0,   rot=0,  sx=1.00, sy=1.00),
            ],
            # Arms: pull up for jump, spread wide in air
            "arm_left": [
                _KF(0.0, tx=0, ty=0,  rot=5,   sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=0,  rot=50,  sx=0.90, sy=1.00),  # arms pull back (crouch)
                _KF(0.4, tx=0, ty=0,  rot=-40, sx=1.20, sy=1.00),  # spread wide in air
                _KF(0.7, tx=0, ty=0,  rot=-35, sx=1.20, sy=1.00),
                _KF(0.9, tx=0, ty=0,  rot=10,  sx=1.00, sy=1.00),
                _KF(1.0, tx=0, ty=0,  rot=5,   sx=1.00, sy=1.00),
            ],
            "arm_right": [
                _KF(0.0, tx=0, ty=0,  rot=-5,  sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=0,  rot=-50, sx=0.90, sy=1.00),
                _KF(0.4, tx=0, ty=0,  rot=40,  sx=1.20, sy=1.00),
                _KF(0.7, tx=0, ty=0,  rot=35,  sx=1.20, sy=1.00),
                _KF(0.9, tx=0, ty=0,  rot=-10, sx=1.00, sy=1.00),
                _KF(1.0, tx=0, ty=0,  rot=-5,  sx=1.00, sy=1.00),
            ],
            # Legs: crouch, extend in air, absorb landing
            "leg_left": [
                _KF(0.0, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=0, rot=30, sx=1.00, sy=0.85),  # crouch squash
                _KF(0.4, tx=0, ty=0, rot=-10, sx=0.97, sy=1.05), # extend in air
                _KF(0.7, tx=0, ty=0, rot=-5,  sx=0.97, sy=1.05),
                _KF(0.9, tx=0, ty=0, rot=20, sx=1.00, sy=0.85),  # absorb landing
                _KF(1.0, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00),
            ],
            "leg_right": [
                _KF(0.0, tx=0, ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(0.2, tx=0, ty=0, rot=-30, sx=1.00, sy=0.85),
                _KF(0.4, tx=0, ty=0, rot=10,  sx=1.03, sy=1.05),
                _KF(0.7, tx=0, ty=0, rot=5,   sx=1.03, sy=1.05),
                _KF(0.9, tx=0, ty=0, rot=-20, sx=1.00, sy=0.85),
                _KF(1.0, tx=0, ty=0, rot=0,   sx=1.00, sy=1.00),
            ],
        },
    },

    # ── Dance: Honkai Star Rail-style, exaggerated perspective rotation ───
    "dance": {
        "duration": 2.0, "fps": 60, "loop": True,
        "parts": {
            # Head: follows body with slight delay, head bobs with flair
            "head": [
                _KF(0.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
                _KF(0.25, tx=4,   ty=-4, rot=-12, sx=0.97, sy=1.00, skew=-0.020),
                _KF(0.5,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
                _KF(0.75, tx=-4,  ty=-4, rot=12,  sx=1.03, sy=1.00, skew=0.020),
                _KF(1.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
                _KF(1.25, tx=4,   ty=-4, rot=-12, sx=0.97, sy=1.00, skew=-0.020),
                _KF(1.5,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
                _KF(1.75, tx=-4,  ty=-4, rot=12,  sx=1.03, sy=1.00, skew=0.020),
                _KF(2.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00, skew=0.000),
            ],
            "hair": [
                _KF(0.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(0.25, tx=6,   ty=-2, rot=-15, sx=1.00, sy=1.00),
                _KF(0.5,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(0.75, tx=-6,  ty=-2, rot=15,  sx=1.00, sy=1.00),
                _KF(1.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(1.25, tx=6,   ty=-2, rot=-15, sx=1.00, sy=1.00),
                _KF(1.5,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(1.75, tx=-6,  ty=-2, rot=15,  sx=1.00, sy=1.00),
                _KF(2.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
            ],
            # Torso: core of the dance — body rotation with strong skew_x
            "torso": [
                _KF(0.0,  tx=0, ty=0,  rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.5,  tx=0, ty=-4, rot=8,  sx=0.97, sy=1.00, skew=-0.060),  # turn left
                _KF(1.0,  tx=0, ty=0,  rot=-8, sx=1.03, sy=1.00, skew=0.060),   # turn right
                _KF(1.5,  tx=0, ty=-4, rot=8,  sx=0.97, sy=1.00, skew=-0.060),
                _KF(2.0,  tx=0, ty=0,  rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            # Arms: wide sweeping movement, scale_x 0.7–1.3 (reaching toward/away camera)
            "arm_left": [
                _KF(0.0,  tx=0,  ty=0,   rot=10,  sx=1.00, sy=1.00),
                _KF(0.25, tx=-6, ty=-12, rot=-40, sx=0.75, sy=1.00),  # reach toward camera (wide)
                _KF(0.5,  tx=0,  ty=0,   rot=10,  sx=1.00, sy=1.00),
                _KF(0.75, tx=6,  ty=0,   rot=60,  sx=1.25, sy=1.00),  # sweep away from camera
                _KF(1.0,  tx=0,  ty=0,   rot=10,  sx=1.00, sy=1.00),
                _KF(1.25, tx=-6, ty=-12, rot=-40, sx=0.75, sy=1.00),
                _KF(1.5,  tx=0,  ty=0,   rot=10,  sx=1.00, sy=1.00),
                _KF(1.75, tx=6,  ty=0,   rot=60,  sx=1.25, sy=1.00),
                _KF(2.0,  tx=0,  ty=0,   rot=10,  sx=1.00, sy=1.00),
            ],
            "arm_right": [
                _KF(0.0,  tx=0,  ty=0,   rot=-10, sx=1.00, sy=1.00),
                _KF(0.25, tx=6,  ty=0,   rot=-60, sx=1.25, sy=1.00),
                _KF(0.5,  tx=0,  ty=0,   rot=-10, sx=1.00, sy=1.00),
                _KF(0.75, tx=-6, ty=-12, rot=40,  sx=0.75, sy=1.00),
                _KF(1.0,  tx=0,  ty=0,   rot=-10, sx=1.00, sy=1.00),
                _KF(1.25, tx=6,  ty=0,   rot=-60, sx=1.25, sy=1.00),
                _KF(1.5,  tx=0,  ty=0,   rot=-10, sx=1.00, sy=1.00),
                _KF(1.75, tx=-6, ty=-12, rot=40,  sx=0.75, sy=1.00),
                _KF(2.0,  tx=0,  ty=0,   rot=-10, sx=1.00, sy=1.00),
            ],
            # Legs: shifting weight side to side
            "leg_left": [
                _KF(0.0,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(0.25, tx=-8, ty=0, rot=-20, sx=0.97, sy=0.97),
                _KF(0.5,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(0.75, tx=8,  ty=0, rot=20,  sx=1.03, sy=1.03),
                _KF(1.0,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(1.5,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(2.0,  tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
            ],
            "leg_right": [
                _KF(0.0,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(0.25, tx=8,  ty=0, rot=20, sx=1.03, sy=1.03),
                _KF(0.5,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(0.75, tx=-8, ty=0, rot=-20, sx=0.97, sy=0.97),
                _KF(1.0,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(1.5,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(2.0,  tx=0,  ty=0, rot=0,  sx=1.00, sy=1.00),
            ],
        },
    },

    # ── Wave: friendly greeting with arm and head ─────────────────────────
    "wave": {
        "duration": 1.2, "fps": 60, "loop": True,
        "parts": {
            "head": [
                _KF(0.0, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.3, tx=0, ty=0, rot=-6, sx=0.99, sy=1.00, skew=-0.010),
                _KF(0.6, tx=0, ty=0, rot=6,  sx=1.01, sy=1.00, skew=0.010),
                _KF(1.2, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            "hair": [
                _KF(0.0, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(0.3, tx=2, ty=0, rot=-8, sx=1.00, sy=1.00),
                _KF(0.6, tx=-2, ty=0, rot=8, sx=1.00, sy=1.00),
                _KF(1.2, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00),
            ],
            "torso": [
                _KF(0.0, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
                _KF(0.6, tx=0, ty=0, rot=-4, sx=1.00, sy=1.00, skew=-0.015),
                _KF(1.2, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            # Right arm waves: scale_x varies (arm raised toward/away from camera)
            "arm_right": [
                _KF(0.0, tx=0, ty=0, rot=-60, sx=1.05, sy=1.00),
                _KF(0.3, tx=0, ty=0, rot=-90, sx=0.95, sy=1.00),
                _KF(0.6, tx=0, ty=0, rot=-60, sx=1.05, sy=1.00),
                _KF(0.9, tx=0, ty=0, rot=-90, sx=0.95, sy=1.00),
                _KF(1.2, tx=0, ty=0, rot=-60, sx=1.05, sy=1.00),
            ],
            "arm_left": [
                _KF(0.0, tx=0, ty=0, rot=8, sx=1.00, sy=1.00),
                _KF(1.2, tx=0, ty=0, rot=8, sx=1.00, sy=1.00),
            ],
            "leg_left":  [_KF(0.0, tx=0, ty=0, rot=0), _KF(1.2, tx=0, ty=0, rot=0)],
            "leg_right": [_KF(0.0, tx=0, ty=0, rot=0), _KF(1.2, tx=0, ty=0, rot=0)],
        },
    },

    # ── Hurt: recoil flash, opacity for impact ────────────────────────────
    "hurt": {
        "duration": 0.7, "fps": 60, "loop": False,
        "parts": {
            "head": [
                _KF(0.0,  tx=0,   ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000, opacity=1.0),
                _KF(0.05, tx=-8,  ty=-4, rot=15, sx=0.95, sy=0.95, skew=0.020, opacity=0.7),  # impact flash
                _KF(0.1,  tx=-8,  ty=-4, rot=15, sx=0.95, sy=0.95, skew=0.020, opacity=1.0),
                _KF(0.25, tx=6,   ty=0,  rot=-8, sx=1.00, sy=1.00, skew=-0.010),
                _KF(0.5,  tx=-2,  ty=0,  rot=4,  sx=1.00, sy=1.00),
                _KF(0.7,  tx=0,   ty=0,  rot=0,  sx=1.00, sy=1.00),
            ],
            "hair": [
                _KF(0.0,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(0.1,  tx=-10, ty=-2, rot=-12, sx=1.00, sy=1.00),
                _KF(0.3,  tx=6,   ty=0,  rot=8,   sx=1.00, sy=1.00),
                _KF(0.5,  tx=-3,  ty=0,  rot=-4,  sx=1.00, sy=1.00),
                _KF(0.7,  tx=0,   ty=0,  rot=0,   sx=1.00, sy=1.00),
            ],
            "torso": [
                _KF(0.0,  tx=0,   ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000, opacity=1.0),
                _KF(0.05, tx=-12, ty=0, rot=20, sx=0.92, sy=0.95, skew=0.030, opacity=0.75),
                _KF(0.1,  tx=-12, ty=0, rot=20, sx=0.92, sy=0.95, skew=0.030, opacity=1.0),
                _KF(0.3,  tx=6,   ty=0, rot=-8, sx=1.00, sy=1.00, skew=-0.015),
                _KF(0.7,  tx=0,   ty=0, rot=0,  sx=1.00, sy=1.00, skew=0.000),
            ],
            "arm_left": [
                _KF(0.0,  tx=0,  ty=0,  rot=0,   sx=1.00, sy=1.00),
                _KF(0.1,  tx=-8, ty=-4, rot=-40, sx=0.90, sy=1.00),
                _KF(0.3,  tx=4,  ty=0,  rot=15,  sx=1.00, sy=1.00),
                _KF(0.7,  tx=0,  ty=0,  rot=0,   sx=1.00, sy=1.00),
            ],
            "arm_right": [
                _KF(0.0,  tx=0, ty=0,  rot=0,  sx=1.00, sy=1.00),
                _KF(0.1,  tx=8, ty=-4, rot=40, sx=1.10, sy=1.00),
                _KF(0.3,  tx=-4, ty=0, rot=-15, sx=1.00, sy=1.00),
                _KF(0.7,  tx=0, ty=0,  rot=0,  sx=1.00, sy=1.00),
            ],
            "leg_left": [
                _KF(0.0, tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
                _KF(0.1, tx=-6, ty=0, rot=-10, sx=1.00, sy=0.95),
                _KF(0.7, tx=0,  ty=0, rot=0,   sx=1.00, sy=1.00),
            ],
            "leg_right": [
                _KF(0.0, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00),
                _KF(0.1, tx=6, ty=0, rot=10, sx=1.00, sy=0.95),
                _KF(0.7, tx=0, ty=0, rot=0,  sx=1.00, sy=1.00),
            ],
        },
    },
}

# ── Keyword → animation name mapping ────────────────────────

_KEYWORD_MAP: list[tuple[str, str]] = [
    (r"\b(run|sprint|dash|corriendo|correr)\b", "run"),
    (r"\b(walk|walking|caminar|caminando|march)\b", "walk"),
    (r"\b(attack|swing|slash|punch|hit|golpe|ataque)\b", "attack"),
    (r"\b(jump|leap|hop|salto|saltar)\b", "jump"),
    (r"\b(dance|dancing|baile|bailar|groove)\b", "dance"),
    (r"\b(wave|waving|greeting|saludar)\b", "wave"),
    (r"\b(hurt|pain|damage|hit|da[ñn]o|golpeado)\b", "hurt"),
    (r"\b(idle|stand|standing|breath|rest|esperar)\b", "idle"),
]


def _detect_animation(prompt: str) -> str:
    """Detect animation type from a text prompt. Falls back to 'idle'."""
    lower = prompt.lower()
    for pattern, anim_name in _KEYWORD_MAP:
        if re.search(pattern, lower):
            return anim_name
    return "idle"


# ── Easing ───────────────────────────────────────────────────

def _smoothstep(t: float) -> float:
    """Smoothstep ease-in-out: t must be in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interpolate(
    kf_list: list[Keyframe],
    t: float,
    loop: bool,
    duration: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Interpolate (tx, ty, rot, scale_x, scale_y, skew_x, opacity) from keyframe list at time t.

    Returns identity defaults for missing fields so old presets continue to work.
    """
    if not kf_list:
        return 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0

    if loop:
        t = t % duration

    # Find surrounding keyframes
    if t <= kf_list[0].time:
        kf = kf_list[0]
        return kf.tx, kf.ty, kf.rot, kf.scale_x, kf.scale_y, kf.skew_x, kf.opacity
    if t >= kf_list[-1].time:
        kf = kf_list[-1]
        return kf.tx, kf.ty, kf.rot, kf.scale_x, kf.scale_y, kf.skew_x, kf.opacity

    for i in range(len(kf_list) - 1):
        ka, kb = kf_list[i], kf_list[i + 1]
        if ka.time <= t <= kb.time:
            seg_len = kb.time - ka.time
            if seg_len < 1e-9:
                return kb.tx, kb.ty, kb.rot, kb.scale_x, kb.scale_y, kb.skew_x, kb.opacity
            raw = (t - ka.time) / seg_len
            s = _smoothstep(raw)
            tx      = ka.tx      + s * (kb.tx      - ka.tx)
            ty      = ka.ty      + s * (kb.ty      - ka.ty)
            rot     = ka.rot     + s * (kb.rot     - ka.rot)
            scale_x = ka.scale_x + s * (kb.scale_x - ka.scale_x)
            scale_y = ka.scale_y + s * (kb.scale_y - ka.scale_y)
            skew_x  = ka.skew_x  + s * (kb.skew_x  - ka.skew_x)
            opacity = ka.opacity + s * (kb.opacity  - ka.opacity)
            return tx, ty, rot, scale_x, scale_y, skew_x, opacity

    kf = kf_list[-1]
    return kf.tx, kf.ty, kf.rot, kf.scale_x, kf.scale_y, kf.skew_x, kf.opacity


# ── Perspective / affine transform ──────────────────────────

def _apply_perspective_transform(
    image: Image.Image,
    rot: float,
    scale_x: float,
    scale_y: float,
    skew_x: float,
    opacity: float,
    pivot_x: float,
    pivot_y: float,
) -> Image.Image:
    """Apply rotation + scale + skew around pivot to simulate 3D perspective.

    Uses PIL's AFFINE transform:
        x' = a*x + b*y + c
        y' = d*x + e*y + f

    The matrix is inverted (PIL applies the inverse mapping).

    Args:
        image: RGBA PIL image of the part.
        rot: Rotation in degrees (CCW positive in PIL convention).
        scale_x: Horizontal scale. <1 = narrower (foreshortening).
        scale_y: Vertical scale. <1 = squash, >1 = stretch.
        skew_x: Horizontal shear for perspective twist.
        opacity: 0.0–1.0 alpha multiplier.
        pivot_x: Pivot x in part-image pixel space.
        pivot_y: Pivot y in part-image pixel space.

    Returns:
        Transformed RGBA image (same size as input).
    """
    # PIL's rotate() sign convention: positive = CCW.  We want the same.
    # But PIL's AFFINE maps *destination* pixels back to source, so the matrix
    # is the inverse of the forward transform.
    # Forward:  p' = M * (p - pivot) + pivot
    # Inverse:  p  = M⁻¹ * (p' - pivot) + pivot

    # Check if we need any transform at all
    is_identity = (
        abs(rot) < 0.01
        and abs(scale_x - 1.0) < 0.001
        and abs(scale_y - 1.0) < 0.001
        and abs(skew_x) < 0.001
        and opacity >= 0.999
    )
    if is_identity:
        return image

    # Build the FORWARD 2x2 matrix: scale → skew → rotate
    # Using PIL CCW convention: positive rot = CCW
    rad = math.radians(rot)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    # Forward 2x2: rotate after scale+skew
    # [sx  0 ] then skew → [sx   skew_x*sy] then rotate
    # [0   sy]              [0    sy       ]
    # Combined scale+skew:
    m_a = scale_x
    m_b = skew_x * scale_y
    m_c = 0.0
    m_d = scale_y

    # Rotate: [cos -sin] * [m_a m_b]
    #         [sin  cos]   [m_c m_d]
    fwd_a = cos_r * m_a - sin_r * m_c
    fwd_b = cos_r * m_b - sin_r * m_d
    fwd_c = sin_r * m_a + cos_r * m_c
    fwd_d = sin_r * m_b + cos_r * m_d

    # Invert the 2x2 forward matrix: [[a b][c d]]^-1 = 1/det * [[d -b][-c a]]
    det = fwd_a * fwd_d - fwd_b * fwd_c
    if abs(det) < 1e-9:
        return image  # degenerate, skip
    inv_a = fwd_d / det
    inv_b = -fwd_b / det
    inv_c = -fwd_c / det
    inv_d = fwd_a / det

    # Translation so pivot stays fixed: t = pivot - M_inv * pivot
    c = pivot_x - inv_a * pivot_x - inv_b * pivot_y
    f = pivot_y - inv_c * pivot_x - inv_d * pivot_y

    result = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (inv_a, inv_b, c, inv_c, inv_d, f),
        resample=Image.Resampling.BICUBIC,
    )

    # Apply opacity to alpha channel
    if opacity < 0.999:
        r, g, b, a = result.split()
        a = a.point(lambda p: int(p * max(0.0, min(1.0, opacity))))
        result = Image.merge("RGBA", (r, g, b, a))

    return result


# ── Spring physics helper ────────────────────────────────────

class _SpringState:
    """Simple damped spring for secondary motion (hair, capes, etc.).

    Simulates a spring attached to a moving anchor. The spring lags behind
    the anchor and oscillates with exponential damping.
    """

    def __init__(self, stiffness: float = 8.0, damping: float = 4.5) -> None:
        self.stiffness = stiffness
        self.damping = damping
        self.position = 0.0   # current spring displacement
        self.velocity = 0.0   # current spring velocity

    def update(self, anchor: float, dt: float) -> float:
        """Step the spring physics and return current displacement from anchor."""
        if dt <= 0:
            return self.position
        # Spring force + damping
        displacement = self.position - anchor
        force = -self.stiffness * displacement - self.damping * self.velocity
        self.velocity += force * dt
        self.position += self.velocity * dt
        return self.position


# ── Main service ─────────────────────────────────────────────

class AffineAnimator2DService:
    """Generates 2D sprite sheet animations from a character 2D model using affine transforms."""

    def animate(
        self,
        model_json: dict,
        parts_dir: Path,
        prompt: str,
        output_dir: Path,
        frame_size: int | None = None,
    ) -> tuple[Path, Path]:
        """Generate a sprite sheet and metadata JSON for the given model + prompt.

        Args:
            model_json: The model dict loaded from model.json.
            parts_dir: Directory containing part PNGs.
            prompt: Text description of the desired animation.
            output_dir: Where to write sprite_sheet.png + animation.json.
            frame_size: Square frame size in pixels. Defaults to settings.SPRITE_SHEET_FRAME_SIZE.

        Returns:
            (sprite_sheet_path, metadata_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_size = frame_size or settings.SPRITE_SHEET_FRAME_SIZE
        fps = settings.SPRITE_SHEET_FPS

        anim_name = _detect_animation(prompt)
        logger.info(f"Animation type detected: {anim_name!r} from prompt: {prompt[:80]!r}")

        preset = ANIMATION_PRESETS.get(anim_name, ANIMATION_PRESETS["idle"])
        duration: float = preset["duration"]
        loop: bool = preset["loop"]
        part_keyframes: dict[str, list[Keyframe]] = preset["parts"]

        n_frames = max(1, int(round(duration * fps)))
        logger.info(f"Rendering {n_frames} frames at {fps}fps ({duration:.2f}s)")

        # Load part images
        parts_data = self._load_parts(model_json, parts_dir)

        # Determine character bounding box in the full image space
        canvas_w: int = model_json["bounds"]["width"]
        canvas_h: int = model_json["bounds"]["height"]

        # Initialize spring states for secondary-motion parts (per part, per axis)
        spring_states: dict[str, dict[str, _SpringState]] = {}
        for part in parts_data:
            if part.get("secondary_motion"):
                amp = part.get("secondary_amplitude", 0.5)
                # Higher amplitude → looser spring (lower stiffness)
                stiffness = max(2.0, 10.0 - amp * 8.0)
                spring_states[part["name"]] = {
                    "rot":  _SpringState(stiffness=stiffness, damping=4.0),
                    "tx":   _SpringState(stiffness=stiffness, damping=4.0),
                    "ty":   _SpringState(stiffness=stiffness, damping=4.0),
                }

        # Initialise body layer cache (reset per animate() call)
        self._body_cache = None

        # Generate frames
        dt = duration / n_frames
        frames: list[Image.Image] = []
        for frame_idx in range(n_frames):
            t = (frame_idx / n_frames) * duration
            frame_img = self._render_frame(
                parts_data=parts_data,
                part_keyframes=part_keyframes,
                t=t,
                dt=dt,
                loop=loop,
                duration=duration,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                frame_size=frame_size,
                spring_states=spring_states,
                model_json=model_json,
                parts_dir=parts_dir,
            )
            frames.append(frame_img)

        # Assemble sprite sheet
        from app.services.spritesheet_export import SpriteSheetExportService
        exporter = SpriteSheetExportService()
        sprite_path, meta_path = exporter.export(
            frames=frames,
            frame_size=frame_size,
            fps=fps,
            anim_name=anim_name,
            output_dir=output_dir,
        )
        return sprite_path, meta_path

    # ── Internals ──────────────────────────────────────────────

    def _load_parts(self, model_json: dict, parts_dir: Path) -> list[dict]:
        """Load PIL Images for each part, sorted by z_order ascending (draw back-to-front)."""
        parts = []
        for part_info in model_json.get("parts", []):
            img_path = parts_dir / part_info["image"]
            if not img_path.exists():
                logger.warning(f"Part image not found: {img_path}, skipping")
                continue
            img = Image.open(str(img_path)).convert("RGBA")
            parts.append({
                "name": part_info["name"],
                "image": img,
                "pivot": part_info["pivot"],          # [px, py] in part-image space
                "z_order": part_info.get("z_order", 0),
                "bounds": part_info["bounds"],         # {x, y, w, h} in canvas space
                # New depth / physics fields (gracefully default if absent)
                "z_depth": part_info.get("z_depth", 0.0),
                "secondary_motion": part_info.get("secondary_motion", False),
                "secondary_amplitude": part_info.get("secondary_amplitude", 0.5),
            })
        # Sort: lowest z_order drawn first (background)
        parts.sort(key=lambda p: p["z_order"])
        return parts

    def _render_frame(
        self,
        parts_data: list[dict],
        part_keyframes: dict[str, list[Keyframe]],
        t: float,
        dt: float,
        loop: bool,
        duration: float,
        canvas_w: int,
        canvas_h: int,
        frame_size: int,
        spring_states: dict[str, dict[str, _SpringState]],
        model_json: dict | None = None,
        parts_dir: Path | None = None,
    ) -> Image.Image:
        """Compose one animation frame with perspective transforms and parallax."""
        frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        # ── Body base layer (SDXL inpainted) — drawn FIRST ───────────────────
        # Check model_json for "body_layer" key (absent = pre-body-layer models)
        body_layer_file = (model_json or {}).get("body_layer")
        if body_layer_file and parts_dir is not None:
            # Lazy-load and cache body_base.png for the duration of this animate() call
            if self._body_cache is None:
                bp = parts_dir / body_layer_file
                if bp.exists():
                    try:
                        self._body_cache = Image.open(str(bp)).convert("RGBA")
                    except Exception as exc:
                        logger.warning(f"AffineAnimator2DService: could not load body layer ({exc})")
            if self._body_cache is not None:
                frame.paste(self._body_cache, (0, 0), self._body_cache)

        for part in parts_data:
            name = part["name"]
            kfs = part_keyframes.get(name, [])
            tx, ty, rot, sx, sy, skew, opacity = _interpolate(kfs, t, loop, duration)

            # ── 1. Parallax based on z_depth ──────────────────────────────────
            z = part.get("z_depth", 0.0)
            # Closer parts (positive z) move more; farther parts move less
            parallax_factor = 1.0 + z * 0.3
            parallax_tx = tx * parallax_factor
            parallax_ty = ty * parallax_factor

            # ── 2. Secondary motion (spring physics for hair, capes, etc.) ────
            if part.get("secondary_motion") and name in spring_states:
                springs = spring_states[name]
                amp = part.get("secondary_amplitude", 0.5)

                # Spring follows the keyframe transform with inertia/lag
                spring_rot = springs["rot"].update(rot * amp, dt)
                spring_tx  = springs["tx"].update(parallax_tx * amp * 0.5, dt)
                spring_ty  = springs["ty"].update(parallax_ty * amp * 0.5, dt)

                # Blend keyframe intent with spring offset
                rot           = rot + (spring_rot - rot * amp) * 0.4
                parallax_tx   = parallax_tx + spring_tx * 0.3
                parallax_ty   = parallax_ty + spring_ty * 0.3

            # ── 3. Apply perspective transform (scale + skew + rotate + opacity) ─
            pivot_px, pivot_py = part["pivot"]
            part_img = _apply_perspective_transform(
                part["image"],
                rot=rot,
                scale_x=sx,
                scale_y=sy,
                skew_x=skew,
                opacity=opacity,
                pivot_x=pivot_px,
                pivot_y=pivot_py,
            )

            # ── 4. Paste with parallax-adjusted offset ────────────────────────
            bounds = part["bounds"]
            paste_x = int(round(bounds["x"] + parallax_tx))
            paste_y = int(round(bounds["y"] + parallax_ty))
            frame.paste(part_img, (paste_x, paste_y), part_img)

        return _fit_to_square(frame, frame_size)


def _fit_to_square(image: Image.Image, size: int) -> Image.Image:
    """Fit image into a square of `size`×`size`, centered, transparent padding."""
    iw, ih = image.size
    scale = min(size / iw, size / ih)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    out.paste(resized, (off_x, off_y), resized)
    return out


# Backward-compatibility alias — kept for any code that imports Animator2DService directly
Animator2DService = AffineAnimator2DService
