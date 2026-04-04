"""
Animation service: auto-rigs a GLB mesh and adds skeletal animation.

Adaptive approach:
1. Classify body type from mesh geometry + prompt keywords
2. Use body-type-specific skeleton template (biped, quadruped, winged_biped, serpentine, compact)
3. Fit skeleton to actual mesh geometry
4. Compute skinning weights
5. Generate body-type-appropriate keyframe animations
6. Build generic GLB (works with any bone count)
"""

import math
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

from app.core.logging import get_logger
from app.services.base import AnimationService
from app.services.animation_utils import (
    BodyType,
    BoneDef,
    SKELETON_BIPED,
    SKELETON_QUADRUPED,
    SKELETON_WINGED_BIPED,
    SKELETON_SERPENTINE,
    SKELETON_COMPACT,
    SKELETONS,
    ANIMATION_PRESETS,
    _detect_preset,
    _quat,
    _qi,
    _ease,
    _ss,
    _quat_multiply,
    _classify_body_type,
    _analyze_mesh_slices,
    _fit_biped_skeleton,
    _fit_quadruped_skeleton,
    _fit_winged_biped_skeleton,
    _fit_serpentine_skeleton,
    _fit_compact_skeleton,
    _SKELETON_FITTERS,
    _compute_weights,
    _compute_weights_biped,
    _parse_attack_hand,
)

logger = get_logger(__name__)


# ── Per-body-type keyframe generators ─────────────────────────

def _gen_keyframes_biped(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Original biped keyframe generator — all 14 biped bones."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)
    s  = scale

    for i, t in enumerate(times):
        p    = (t / cycle) * 2 * math.pi
        dt_v = None

        if anim_type == "walk":
            r = {
                "root":        _quat([0,0,1], _ss(p) * 0.008),
                "hip":         _quat([0,1,0], _ss(p) * 0.03),
                "spine":       _quat([0,1,0], _ss(p) * 0.02),
                "chest":       _quat([0,1,0], _ss(p) * -0.015),
                "neck":        _quat([0,1,0], _ss(p) * -0.008),
                "head":        _quat([1,0,0], _ss(p*2) * 0.01),
                "upper_leg_l": _quat([1,0,0], _ss(p) * 0.28),
                "lower_leg_l": _quat([1,0,0], max(0, _ss(p - 0.6)) * 0.40),
                "upper_leg_r": _quat([1,0,0], _ss(p + math.pi) * 0.28),
                "lower_leg_r": _quat([1,0,0], max(0, _ss(p + math.pi - 0.6)) * 0.40),
                "upper_arm_l": _quat([1,0,0], _ss(p + math.pi) * 0.18),
                "lower_arm_l": _quat([1,0,0], max(0, _ss(p + math.pi - 0.2)) * 0.12),
                "upper_arm_r": _quat([1,0,0], _ss(p) * 0.18),
                "lower_arm_r": _quat([1,0,0], max(0, _ss(p - 0.2)) * 0.12),
            }
            if bone_name == "hip":
                dt_v = np.array([0, abs(_ss(p*2)) * s * 0.008, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "run":
            r = {
                "root":        _quat([1,0,0], -0.06),
                "hip":         _quat([0,1,0], _ss(p) * 0.04),
                "spine":       _quat([1,0,0], -0.03 + _ss(p) * 0.02),
                "chest":       _quat([1,0,0], -0.02 + _ss(p) * -0.015),
                "neck":        _quat([1,0,0], 0.04),
                "head":        _quat([1,0,0], 0.03 + _ss(p*2) * 0.012),
                "upper_leg_l": _quat([1,0,0], _ss(p) * 0.45),
                "lower_leg_l": _quat([1,0,0], max(0, _ss(p - 0.4)) * 0.65),
                "upper_leg_r": _quat([1,0,0], _ss(p + math.pi) * 0.45),
                "lower_leg_r": _quat([1,0,0], max(0, _ss(p + math.pi - 0.4)) * 0.65),
                "upper_arm_l": _quat([1,0,0], _ss(p + math.pi) * 0.35),
                "lower_arm_l": _quat([1,0,0], 0.3 + _ss(p + math.pi) * 0.15),
                "upper_arm_r": _quat([1,0,0], _ss(p) * 0.35),
                "lower_arm_r": _quat([1,0,0], 0.3 + _ss(p) * 0.15),
            }
            if bone_name == "hip":
                dt_v = np.array([0, abs(_ss(p*2)) * s * 0.014, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "breathing":
            sp = math.sin(p)
            r = {
                "root":        _quat([0,0,1], sp * 0.002),
                "hip":         _quat([1,0,0], sp * 0.003),
                "spine":       _quat([1,0,0], sp * 0.005),
                "chest":       _quat([1,0,0], sp * 0.008),
                "neck":        _quat([1,0,0], sp * -0.003),
                "head":        _quat([1,0,0], math.sin(p*0.5) * 0.01),
                "upper_arm_l": _quat([0,0,1], sp * 0.005),
                "lower_arm_l": _quat([1,0,0], sp * 0.003),
                "upper_arm_r": _quat([0,0,1], sp * -0.005),
                "lower_arm_r": _quat([1,0,0], sp * 0.003),
                "upper_leg_l": _quat([1,0,0], sp * 0.002),
                "lower_leg_l": _qi(),
                "upper_leg_r": _quat([1,0,0], sp * 0.002),
                "lower_leg_r": _qi(),
            }
            if bone_name == "chest":
                dt_v = np.array([0, sp * s * 0.003, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "gesture":
            r = {
                "root":        _quat([0,1,0], _ss(p*0.3) * 0.01),
                "hip":         _quat([0,1,0], _ss(p*0.5) * 0.01),
                "spine":       _quat([0,1,0], _ss(p) * 0.02),
                "chest":       _quat([0,1,0], _ss(p) * 0.015),
                "neck":        _quat([0,1,0], _ss(p) * 0.02),
                "head":        _quat([0,1,0], _ss(p*0.8) * 0.06),
                "upper_arm_l": _quat([0,0,1], _ss(p*0.5) * 0.03),
                "lower_arm_l": _quat([1,0,0], _ss(p*0.5) * 0.02),
                "upper_arm_r": _quat([0,0,1], -0.7 - 0.3 * _ease((math.sin(p)+1)/2)),
                "lower_arm_r": _quat([1,0,0], -0.3 + _ss(p*2) * 0.35),
                "upper_leg_l": _quat([1,0,0], _ss(p*0.3) * 0.01),
                "lower_leg_l": _qi(),
                "upper_leg_r": _quat([1,0,0], _ss(p*0.3) * 0.01),
                "lower_leg_r": _qi(),
            }
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "jump":
            tn = (t % cycle) / cycle
            e  = _ease(tn) if tn < 0.5 else _ease(1.0 - tn)
            r = {
                "root":        _qi(),
                "hip":         _quat([1,0,0], -e * 0.05),
                "spine":       _quat([1,0,0], -e * 0.04),
                "chest":       _quat([1,0,0], e * 0.02),
                "neck":        _quat([1,0,0], e * 0.03),
                "head":        _quat([1,0,0], e * 0.02),
                "upper_leg_l": _quat([1,0,0], -e * 0.30),
                "lower_leg_l": _quat([1,0,0], e * 0.40),
                "upper_leg_r": _quat([1,0,0], -e * 0.30),
                "lower_leg_r": _quat([1,0,0], e * 0.40),
                "upper_arm_l": _quat([0,0,1], e * 0.40),
                "lower_arm_l": _quat([1,0,0], -e * 0.15),
                "upper_arm_r": _quat([0,0,1], -e * 0.40),
                "lower_arm_r": _quat([1,0,0], -e * 0.15),
            }
            if bone_name == "root":
                dt_v = np.array([0, e * s * 0.08, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "rotation":
            angle = (t / cycle) * 2 * math.pi
            r = {
                "root":        _quat([0,1,0], angle),
                "hip":         _qi(),
                "spine":       _qi(),
                "chest":       _qi(),
                "neck":        _qi(),
                "head":        _qi(),
                "upper_arm_l": _quat([0,0,1], math.sin(angle) * 0.05),
                "lower_arm_l": _qi(),
                "upper_arm_r": _quat([0,0,1], -math.sin(angle) * 0.05),
                "lower_arm_r": _qi(),
                "upper_leg_l": _qi(),
                "lower_leg_l": _qi(),
                "upper_leg_r": _qi(),
                "lower_leg_r": _qi(),
            }
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "dance":
            r = {
                "root":        _quat([0,0,1], _ss(p*0.5) * 0.02),
                "hip":         _quat([0,1,0], _ss(p) * 0.12),
                "spine":       _quat([0,0,1], _ss(p) * 0.06),
                "chest":       _quat([0,1,0], _ss(p+0.5) * 0.05),
                "neck":        _quat([0,0,1], _ss(p) * 0.03),
                "head":        _quat([0,1,0], _ss(p*2) * 0.06),
                "upper_arm_l": _quat([0,0,1], 0.3 + _ss(p) * 0.40),
                "lower_arm_l": _quat([1,0,0], -0.2 + _ss(p*2) * 0.25),
                "upper_arm_r": _quat([0,0,1], -0.3 + _ss(p+1.5) * 0.40),
                "lower_arm_r": _quat([1,0,0], -0.2 + _ss(p*2+1) * 0.25),
                "upper_leg_l": _quat([1,0,0], _ss(p) * 0.15),
                "lower_leg_l": _quat([1,0,0], max(0, _ss(p-0.3)) * 0.20),
                "upper_leg_r": _quat([1,0,0], _ss(p+math.pi) * 0.15),
                "lower_leg_r": _quat([1,0,0], max(0, _ss(p+math.pi-0.3)) * 0.20),
            }
            if bone_name == "hip":
                dt_v = np.array([_ss(p) * s * 0.005, abs(_ss(p*2)) * s * 0.01, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "attack":
            tn   = (t % cycle) / cycle
            hand = _parse_attack_hand(prompt)
            ua   = f"upper_arm_{hand[0]}"
            la   = f"lower_arm_{hand[0]}"
            ua_o = "upper_arm_l" if hand == "right" else "upper_arm_r"
            la_o = "lower_arm_l" if hand == "right" else "lower_arm_r"
            sd   = 1 if hand == "right" else -1

            if tn < 0.3:
                e = _ease(tn / 0.3)
                r = {
                    "root":   _quat([0,1,0], sd*e*0.04), "hip": _quat([0,1,0], sd*e*0.03),
                    "spine":  _quat([0,1,0], sd*e*0.12), "chest": _quat([0,1,0], sd*e*0.06),
                    "neck":   _quat([0,1,0], sd*e*-0.03), "head": _quat([0,1,0], sd*e*-0.04),
                    ua: _quat([1,0,0], -e*0.7), la: _quat([1,0,0], -e*0.5),
                    ua_o: _quat([1,0,0], e*0.05), la_o: _quat([1,0,0], e*0.03),
                    "upper_leg_l": _quat([1,0,0], e*0.03), "lower_leg_l": _qi(),
                    "upper_leg_r": _quat([1,0,0], -e*0.03), "lower_leg_r": _qi(),
                }
            elif tn < 0.55:
                e = _ease((tn - 0.3) / 0.25)
                r = {
                    "root":  _quat([0,1,0], sd*(0.04-e*0.10)), "hip": _quat([0,1,0], sd*(0.03-e*0.06)),
                    "spine": _quat([0,1,0], sd*(0.12-e*0.24)), "chest": _quat([0,1,0], sd*(0.06-e*0.12)),
                    "neck":  _quat([0,1,0], sd*(-0.03+e*0.06)), "head": _quat([0,1,0], sd*(-0.04+e*0.02)),
                    ua: _quat([1,0,0], -0.7+e*1.2), la: _quat([1,0,0], -0.5+e*0.7),
                    ua_o: _quat([1,0,0], 0.05-e*0.05), la_o: _quat([1,0,0], 0.03-e*0.03),
                    "upper_leg_l": _quat([1,0,0], 0.03+e*0.02), "lower_leg_l": _quat([1,0,0], e*0.03),
                    "upper_leg_r": _quat([1,0,0], -0.03-e*0.02), "lower_leg_r": _quat([1,0,0], e*0.03),
                }
            else:
                e = _ease((tn - 0.55) / 0.45)
                r = {
                    "root":  _quat([0,1,0], sd*-0.06*(1-e)), "hip": _quat([0,1,0], sd*-0.03*(1-e)),
                    "spine": _quat([0,1,0], sd*-0.12*(1-e)), "chest": _quat([0,1,0], sd*-0.06*(1-e)),
                    "neck":  _quat([0,1,0], sd*0.03*(1-e)), "head": _quat([0,1,0], sd*-0.02*(1-e)),
                    ua: _quat([1,0,0], 0.5*(1-e)), la: _quat([1,0,0], 0.2*(1-e)),
                    ua_o: _qi(), la_o: _qi(),
                    "upper_leg_l": _quat([1,0,0], 0.05*(1-e)), "lower_leg_l": _quat([1,0,0], 0.03*(1-e)),
                    "upper_leg_r": _quat([1,0,0], -0.05*(1-e)), "lower_leg_r": _quat([1,0,0], 0.03*(1-e)),
                }
            if bone_name == "hip":
                dt_v = np.array([0, 0, sd * _ss(p*0.5) * s * 0.003])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "fly":
            r = {
                "root":        _quat([1,0,0], _ss(p*0.5) * 0.02),
                "hip":         _quat([1,0,0], _ss(p) * 0.01),
                "spine":       _quat([1,0,0], _ss(p) * 0.01),
                "chest":       _quat([1,0,0], _ss(p) * 0.02),
                "neck":        _quat([1,0,0], -_ss(p) * 0.01),
                "head":        _quat([1,0,0], _ss(p*0.5) * 0.01),
                "upper_arm_l": _quat([0,0,1], 0.5 + _ss(p*2) * 0.40),
                "lower_arm_l": _quat([0,0,1], _ss(p*2+0.3) * 0.20),
                "upper_arm_r": _quat([0,0,1], -0.5 - _ss(p*2) * 0.40),
                "lower_arm_r": _quat([0,0,1], -_ss(p*2+0.3) * 0.20),
                "upper_leg_l": _quat([1,0,0], 0.1 + _ss(p) * 0.05),
                "lower_leg_l": _quat([1,0,0], _ss(p) * 0.03),
                "upper_leg_r": _quat([1,0,0], 0.1 + _ss(p+0.5) * 0.05),
                "lower_leg_r": _quat([1,0,0], _ss(p+0.5) * 0.03),
            }
            if bone_name == "root":
                dt_v = np.array([0, math.sin(p) * s * 0.015, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "bounce":
            b = abs(math.sin(p))
            r = {
                "root":        _quat([0,0,1], _ss(p*0.5) * 0.005),
                "hip":         _quat([1,0,0], -b * 0.03),
                "spine":       _quat([1,0,0], -b * 0.02),
                "chest":       _quat([1,0,0], b * 0.015),
                "neck":        _quat([1,0,0], b * 0.01),
                "head":        _quat([1,0,0], b * 0.015),
                "upper_arm_l": _quat([0,0,1], b * 0.10),
                "lower_arm_l": _quat([1,0,0], b * 0.05),
                "upper_arm_r": _quat([0,0,1], -b * 0.10),
                "lower_arm_r": _quat([1,0,0], b * 0.05),
                "upper_leg_l": _quat([1,0,0], -b * 0.12),
                "lower_leg_l": _quat([1,0,0], b * 0.15),
                "upper_leg_r": _quat([1,0,0], -b * 0.12),
                "lower_leg_r": _quat([1,0,0], b * 0.15),
            }
            if bone_name == "root":
                dt_v = np.array([0, b * s * 0.04, 0])
            ro[i] = r.get(bone_name, _qi())

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


def _gen_keyframes_quadruped(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Quadruped keyframe generator — diagonal gait walk, gallop run."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)
    s  = scale

    for i, t in enumerate(times):
        p    = (t / cycle) * 2 * math.pi
        dt_v = None

        if anim_type in ("walk", "run"):
            # Diagonal gait: front-left + back-right together, then swap
            # Phase offsets: FL=0, BR=0, FR=π, BL=π
            amp = 0.35 if anim_type == "walk" else 0.55

            r = {
                "root":              _quat([0,0,1], _ss(p) * 0.005),
                "hip":               _quat([0,1,0], _ss(p) * 0.03),
                "spine":             _quat([0,1,0], _ss(p) * 0.02),
                "chest":             _quat([0,1,0], _ss(p) * -0.02),
                "neck":              _quat([1,0,0], _ss(p) * 0.02),
                "head":              _quat([1,0,0], _ss(p*2) * 0.01),
                "shoulder_l":        _quat([1,0,0], _ss(p + math.pi) * amp * 0.3),
                "upper_front_leg_l": _quat([1,0,0], _ss(p + math.pi) * amp),
                "lower_front_leg_l": _quat([1,0,0], max(0, _ss(p + math.pi - 0.5)) * amp * 0.8),
                "shoulder_r":        _quat([1,0,0], _ss(p) * amp * 0.3),
                "upper_front_leg_r": _quat([1,0,0], _ss(p) * amp),
                "lower_front_leg_r": _quat([1,0,0], max(0, _ss(p - 0.5)) * amp * 0.8),
                "upper_back_leg_l":  _quat([1,0,0], _ss(p) * amp),
                "lower_back_leg_l":  _quat([1,0,0], max(0, _ss(p - 0.5)) * amp * 0.8),
                "upper_back_leg_r":  _quat([1,0,0], _ss(p + math.pi) * amp),
                "lower_back_leg_r":  _quat([1,0,0], max(0, _ss(p + math.pi - 0.5)) * amp * 0.8),
                "tail_base":         _quat([0,1,0], _ss(p) * 0.15),
            }
            if bone_name == "hip":
                bob = abs(_ss(p*2)) * s * 0.008
                dt_v = np.array([0, bob, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "breathing":
            sp = math.sin(p)
            r = {
                "root":              _quat([0,0,1], sp * 0.001),
                "hip":               _quat([1,0,0], sp * 0.005),
                "spine":             _quat([1,0,0], sp * 0.006),
                "chest":             _quat([1,0,0], sp * 0.008),
                "neck":              _quat([1,0,0], sp * 0.004),
                "head":              _quat([1,0,0], math.sin(p*0.5) * 0.008),
                "tail_base":         _quat([0,1,0], sp * 0.05),
            }
            if bone_name == "chest":
                dt_v = np.array([0, sp * s * 0.003, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type in ("bounce", "jump"):
            b = abs(math.sin(p)) if anim_type == "bounce" else _ease(((t % cycle) / cycle))
            r = {
                "root":              _qi(),
                "hip":               _quat([1,0,0], -b * 0.06),
                "spine":             _quat([1,0,0], -b * 0.04),
                "chest":             _quat([1,0,0], b * 0.02),
                "neck":              _quat([1,0,0], b * 0.03),
                "head":              _quat([1,0,0], b * 0.02),
                "upper_front_leg_l": _quat([1,0,0], -b * 0.25),
                "lower_front_leg_l": _quat([1,0,0], b * 0.35),
                "upper_front_leg_r": _quat([1,0,0], -b * 0.25),
                "lower_front_leg_r": _quat([1,0,0], b * 0.35),
                "upper_back_leg_l":  _quat([1,0,0], -b * 0.25),
                "lower_back_leg_l":  _quat([1,0,0], b * 0.35),
                "upper_back_leg_r":  _quat([1,0,0], -b * 0.25),
                "lower_back_leg_r":  _quat([1,0,0], b * 0.35),
                "tail_base":         _quat([1,0,0], b * 0.20),
            }
            if bone_name == "root":
                dt_v = np.array([0, b * s * 0.06, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "rotation":
            angle = (t / cycle) * 2 * math.pi
            r = {"root": _quat([0,1,0], angle)}
            ro[i] = r.get(bone_name, _qi())

        else:
            # Fallback: breathing for unsupported types
            sp = math.sin(p)
            r = {
                "spine": _quat([1,0,0], sp * 0.005),
                "chest": _quat([1,0,0], sp * 0.008),
                "head":  _quat([1,0,0], math.sin(p*0.5) * 0.01),
                "tail_base": _quat([0,1,0], sp * 0.08),
            }
            ro[i] = r.get(bone_name, _qi())

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


def _gen_keyframes_winged_biped(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Winged biped: biped base + wing flap + tail sway."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)

    # Wings and tail get special treatment; other bones delegate to biped
    WING_BONES = {"wing_l", "wing_tip_l", "wing_r", "wing_tip_r", "tail_base"}

    for i, t in enumerate(times):
        p = (t / cycle) * 2 * math.pi
        dt_v = None

        if bone_name in WING_BONES:
            if anim_type in ("fly", "walk", "run", "breathing"):
                flap_amp = 0.8 if anim_type == "fly" else 0.3
                wing_r = {
                    "wing_l":      _quat([0,0,1],  0.3 + _ss(p*2) * flap_amp),
                    "wing_tip_l":  _quat([0,0,1],  0.2 + _ss(p*2 + 0.5) * flap_amp * 0.5),
                    "wing_r":      _quat([0,0,1], -0.3 - _ss(p*2) * flap_amp),
                    "wing_tip_r":  _quat([0,0,1], -0.2 - _ss(p*2 + 0.5) * flap_amp * 0.5),
                    "tail_base":   _quat([1,0,0], _ss(p) * 0.15),
                }
                ro[i] = wing_r.get(bone_name, _qi())
                if bone_name == "wing_l" and anim_type == "fly":
                    dt_v = np.array([0, math.sin(p*2) * scale * 0.01, 0])
            else:
                # Idle / other: gentle wing droop
                r = {
                    "wing_l":    _quat([0,0,1], 0.1 + math.sin(p * 0.5) * 0.05),
                    "wing_tip_l":_quat([0,0,1], 0.05),
                    "wing_r":    _quat([0,0,1], -0.1 - math.sin(p * 0.5) * 0.05),
                    "wing_tip_r":_quat([0,0,1], -0.05),
                    "tail_base": _quat([0,1,0], _ss(p * 0.5) * 0.08),
                }
                ro[i] = r.get(bone_name, _qi())
        else:
            # Delegate to biped for non-wing bones
            bt_single, br_single = _gen_keyframes_biped(
                bone_name, anim_type, times[i:i+1], cycle, scale, prompt
            )
            ro[i] = br_single[0]
            if np.any(bt_single[0] != 0):
                dt_v = bt_single[0]

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


def _gen_keyframes_serpentine(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Serpentine animation: wave propagation along body segments.
    Walk/slither: horizontal S-wave (Y-axis rotation)
    Fly: vertical undulation (X-axis rotation) + horizontal sway
    Idle: gentle breathing wave
    """
    n = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)

    name_to_idx = {
        "root": 0, "segment_1": 1, "segment_2": 2, "segment_3": 3,
        "segment_4": 4, "segment_5": 5, "segment_6": 6, "head": 7,
    }
    bi = name_to_idx.get(bone_name, 0)
    num_seg = 8
    # Phase offset: wave propagates from tail (root=0) to head (7)
    # Each segment is offset so the wave travels along the body
    phase_offset = bi * (2 * math.pi / num_seg)
    # Amplitude decreases slightly toward head (tail whips more)
    amp_scale = 1.0 - bi * 0.04  # root=1.0, head=0.72

    for i, t in enumerate(times):
        p = (t / cycle) * 2 * math.pi

        if anim_type in ("walk", "run"):
            # Slither: horizontal S-wave (rotate around Y)
            amp_h = (0.20 if anim_type == "walk" else 0.30) * amp_scale
            wave_h = _ss(p - phase_offset)

            # Also slight vertical undulation
            amp_v = amp_h * 0.3
            wave_v = _ss(p - phase_offset + math.pi * 0.5)

            # Combine: Y-rotation (horizontal sway) + X-rotation (vertical bob)
            ry = wave_h * amp_h
            rx = wave_v * amp_v

            # Head: more pitch, less yaw — looking where it's going
            if bone_name == "head":
                rx = _ss(p * 0.5) * 0.06
                ry = wave_h * amp_h * 0.5
                tr[i] = np.array([0, abs(wave_v) * scale * 0.004, 0])

            # Root: also slight lateral translation for ground contact feel
            if bone_name == "root":
                tr[i] = np.array([wave_h * scale * 0.008, 0, 0])

            ro[i] = _quat_multiply(
                _quat([0, 1, 0], ry),
                _quat([1, 0, 0], rx),
            )

        elif anim_type == "fly":
            # Flying serpent/dragon: primary vertical undulation + horizontal sway
            # Body makes flowing S-curves in 3D
            amp_v = 0.25 * amp_scale  # vertical (pitch)
            amp_h = 0.12 * amp_scale  # horizontal (yaw)
            amp_r = 0.06 * amp_scale  # roll

            wave_v = _ss(p - phase_offset)
            wave_h = _ss(p - phase_offset + math.pi * 0.3)
            wave_r = _ss(p * 0.5 - phase_offset * 0.5)

            rx = wave_v * amp_v
            ry = wave_h * amp_h
            rz = wave_r * amp_r

            # Root: vertical bobbing for altitude variation
            if bone_name == "root":
                tr[i] = np.array([0, _ss(p) * scale * 0.015, 0])

            # Head: look forward/up, less side-to-side
            if bone_name == "head":
                rx = _ss(p * 0.5) * 0.08  # gentle pitch
                ry = wave_h * amp_h * 0.3
                rz = 0

            ro[i] = _quat_multiply(
                _quat_multiply(
                    _quat([1, 0, 0], rx),
                    _quat([0, 1, 0], ry),
                ),
                _quat([0, 0, 1], rz),
            )

        elif anim_type == "breathing":
            sp = math.sin(p + bi * math.pi * 0.3)
            rx = sp * 0.015
            ry = sp * 0.02
            if bone_name == "head":
                rx = math.sin(p * 0.5) * 0.03
                ry = math.sin(p * 0.3) * 0.015
            ro[i] = _quat_multiply(
                _quat([1, 0, 0], rx),
                _quat([0, 1, 0], ry),
            )

        elif anim_type == "attack":
            # Lunge forward then recoil
            tn = (t % cycle) / cycle
            if tn < 0.3:
                e = _ease(tn / 0.3)
                # Coil back (segments curve backward)
                ang = -e * 0.15 * (num_seg - bi) / num_seg
                ro[i] = _quat([1, 0, 0], ang)
                if bone_name == "root":
                    tr[i] = np.array([0, 0, -e * scale * 0.02])
            elif tn < 0.5:
                e = _ease((tn - 0.3) / 0.2)
                # Strike forward
                ang = (-0.15 + e * 0.4) * (num_seg - bi) / num_seg
                ro[i] = _quat([1, 0, 0], ang)
                if bone_name == "root":
                    tr[i] = np.array([0, 0, (-0.02 + e * 0.06) * scale])
                if bone_name == "head":
                    ro[i] = _quat([1, 0, 0], e * 0.3)  # head lunges forward/down
            else:
                e = _ease((tn - 0.5) / 0.5)
                ang = 0.25 * (1 - e) * (num_seg - bi) / num_seg
                ro[i] = _quat([1, 0, 0], ang)
                if bone_name == "root":
                    tr[i] = np.array([0, 0, 0.04 * (1 - e) * scale])

        elif anim_type == "rotation":
            if bone_name == "root":
                ro[i] = _quat([0, 1, 0], (t / cycle) * 2 * math.pi)

        else:
            # Default: gentle flowing sway
            wave = _ss(p - phase_offset)
            ro[i] = _quat_multiply(
                _quat([0, 1, 0], wave * 0.12 * amp_scale),
                _quat([1, 0, 0], wave * 0.04 * amp_scale),
            )

    return tr, ro


def _gen_keyframes_compact(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Compact: squash-and-stretch + gentle wobble."""
    n  = len(times)
    tr = np.zeros((n, 3), dtype=np.float32)
    ro = np.tile(_qi(), (n, 1)).astype(np.float32)

    for i, t in enumerate(times):
        p  = (t / cycle) * 2 * math.pi
        sp = math.sin(p)
        b  = abs(math.sin(p))
        dt_v = None

        if anim_type in ("walk", "run", "bounce"):
            r = {
                "root": _quat([0,0,1], sp * 0.02),
                "body": _quat([0,0,1], sp * 0.05),
                "top":  _quat([1,0,0], sp * 0.04),
                "base": _quat([1,0,0], -sp * 0.02),
            }
            if bone_name == "body":
                dt_v = np.array([0, b * scale * 0.04, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "breathing":
            r = {
                "root": _quat([0,0,1], sp * 0.005),
                "body": _quat([1,0,0], sp * 0.008),
                "top":  _quat([1,0,0], sp * 0.01),
                "base": _qi(),
            }
            if bone_name == "body":
                dt_v = np.array([0, sp * scale * 0.005, 0])
            ro[i] = r.get(bone_name, _qi())

        elif anim_type == "rotation":
            angle = (t / cycle) * 2 * math.pi
            if bone_name == "root":
                ro[i] = _quat([0,1,0], angle)

        else:
            r = {
                "body": _quat([0,1,0], sp * 0.08),
                "top":  _quat([0,0,1], sp * 0.05),
            }
            ro[i] = r.get(bone_name, _qi())

        if dt_v is not None:
            tr[i] = dt_v

    return tr, ro


# Dispatch table: (body_type) → keyframe generator
_KF_GENERATORS = {
    BodyType.BIPED:        _gen_keyframes_biped,
    BodyType.QUADRUPED:    _gen_keyframes_quadruped,
    BodyType.WINGED_BIPED: _gen_keyframes_winged_biped,
    BodyType.SERPENTINE:   _gen_keyframes_serpentine,
    BodyType.COMPACT:      _gen_keyframes_compact,
}


# ── Backwards-compatible wrapper (kept for any external callers) ───────

def _gen_keyframes(
    bone_name: str, anim_type: str,
    times: np.ndarray, cycle: float, scale: float,
    prompt: str = "",
    body_type: str = BodyType.BIPED,
) -> tuple[np.ndarray, np.ndarray]:
    """Route to the correct per-body-type keyframe generator."""
    gen = _KF_GENERATORS.get(body_type, _gen_keyframes_biped)
    return gen(bone_name, anim_type, times, cycle, scale, prompt)


# ── Main service ──────────────────────────────────────────────

class ProceduralAnimationService(AnimationService):

    def load_model(self) -> None:
        logger.info("Procedural animation service ready (multi-body-type)")

    def animate(
        self, glb_path: Path, prompt: str, output_path: Path,
        duration: float = 3.0, fps: int = 30,
    ) -> Path:
        logger.info(f"Animating {glb_path.name}: '{prompt[:60]}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        anim_name, preset = _detect_preset(prompt)
        logger.info(f"Animation: {anim_name} (cycle={preset['cycle']}s)")

        # 1. Load mesh
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No meshes found")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            raise ValueError(f"Unsupported: {type(scene)}")

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # 2. Classify body type
        body_type = _classify_body_type(mesh.vertices, prompt)
        logger.info(f"Detected body type: {body_type}")

        # 3. Select skeleton template and fit to mesh
        bone_hierarchy = SKELETONS[body_type]
        fitter = _SKELETON_FITTERS[body_type]
        logger.info(f"Fitting {body_type} skeleton ({len(bone_hierarchy)} bones) to mesh geometry...")
        bone_positions, segment_info = fitter(mesh.vertices)

        for bi, bd in enumerate(bone_hierarchy):
            logger.info(f"  {bd.name:24s} pos={bone_positions[bi].round(3).tolist()}")

        bmin, bmax = mesh.bounds
        bsize       = bmax - bmin
        model_height = float(bsize[1])

        # 4. Compute weights
        logger.info("Computing skinning weights...")
        weights = _compute_weights(mesh.vertices, bone_positions, segment_info, bmin, bsize)

        # Log weight distribution
        num_bones = len(bone_hierarchy)
        primary   = weights.argmax(axis=1)
        for bi in range(num_bones):
            count = (primary == bi).sum()
            pct   = 100 * count / len(primary)
            logger.info(f"  {bone_hierarchy[bi].name:24s} {count:5d} verts ({pct:4.1f}%)")

        # 5. Compute local (parent-relative) positions for bind pose
        local_positions = np.zeros_like(bone_positions)
        for bi, bd in enumerate(bone_hierarchy):
            if bd.parent_idx is not None:
                local_positions[bi] = bone_positions[bi] - bone_positions[bd.parent_idx]
            else:
                local_positions[bi] = bone_positions[bi]

        # 6. Generate keyframes
        num_frames  = int(duration * fps)
        times       = np.linspace(0, duration, num_frames, dtype=np.float32)
        gen         = _KF_GENERATORS[body_type]

        all_trans, all_rots = [], []
        for bi, bd in enumerate(bone_hierarchy):
            bt_delta, br = gen(
                bd.name, preset["type"], times, preset["cycle"], model_height, prompt
            )
            # Translation = bind pose local position + animation delta
            # (glTF animation REPLACES node translation, so bind pose must be in keyframe)
            bt = bt_delta + local_positions[bi].astype(np.float32)
            all_trans.append(bt)
            all_rots.append(br)

        # 7. Build GLB
        logger.info("Building animated GLB...")
        glb = self._build_gltf(
            mesh, bone_hierarchy, bone_positions, weights,
            times, all_trans, all_rots, anim_name
        )

        output_path.write_bytes(glb)
        logger.info(f"Animated GLB: {output_path} ({len(glb)} bytes)")
        return output_path

    def _build_gltf(
        self, mesh, bone_hierarchy: list[BoneDef], bone_positions: np.ndarray,
        weights: np.ndarray, times: np.ndarray,
        bone_trans: list, bone_rots: list, anim_name: str
    ) -> bytes:
        """Generic glTF builder — works with any skeleton (any bone count)."""
        vertices = mesh.vertices.astype(np.float32)
        normals  = mesh.vertex_normals.astype(np.float32)
        indices  = mesh.faces.astype(np.uint32).flatten()
        nv       = len(vertices)
        nf       = len(times)
        nb       = len(bone_hierarchy)  # dynamic bone count

        # Skinning data (top-4 bone influences per vertex)
        jd = np.zeros((nv, 4), dtype=np.uint16)
        wd = np.zeros((nv, 4), dtype=np.float32)
        for v in range(nv):
            t4 = np.argsort(weights[v])[-4:][::-1]
            for k in range(4):
                jd[v, k] = t4[k]
                wd[v, k] = weights[v, t4[k]]
            ws = wd[v].sum()
            if ws > 0:
                wd[v] /= ws

        # Inverse bind matrices
        ibms = np.zeros((nb, 16), dtype=np.float32)
        for bi in range(nb):
            ibm = np.eye(4, dtype=np.float32)
            ibm[0, 3] = -bone_positions[bi][0]
            ibm[1, 3] = -bone_positions[bi][1]
            ibm[2, 3] = -bone_positions[bi][2]
            ibms[bi]  = ibm.T.flatten()

        # Binary buffer construction
        buf = bytearray()
        bvs, accs = [], []

        def pad():
            while len(buf) % 4: buf.append(0)

        def add(d, tgt=None):
            pad()
            off = len(buf)
            buf.extend(d)
            bv = {"buffer": 0, "byteOffset": off, "byteLength": len(d)}
            if tgt:
                bv["target"] = tgt
            bvs.append(bv)
            return len(bvs) - 1

        def acc(bvi, ct, cnt, at, mn=None, mx=None):
            a = {"bufferView": bvi, "componentType": ct, "count": cnt, "type": at}
            if mn: a["min"] = mn
            if mx: a["max"] = mx
            accs.append(a)
            return len(accs) - 1

        pa  = acc(add(vertices.tobytes(), 34962), 5126, nv, "VEC3",
                  vertices.min(0).tolist(), vertices.max(0).tolist())
        na  = acc(add(normals.tobytes(),  34962), 5126, nv, "VEC3")
        ia  = acc(add(indices.tobytes(),  34963), 5125, len(indices), "SCALAR",
                  [int(indices.min())], [int(indices.max())])
        ja  = acc(add(jd.tobytes(),       34962), 5123, nv, "VEC4")
        wa  = acc(add(wd.tobytes(),       34962), 5126, nv, "VEC4")
        ima = acc(add(ibms.tobytes()),             5126, nb, "MAT4")
        ta  = acc(add(times.tobytes()),            5126, nf, "SCALAR",
                  [float(times[0])], [float(times[-1])])

        chs, sms = [], []
        si = 0
        for bi in range(nb):
            bt  = bone_trans[bi]
            bta = acc(add(bt.tobytes()), 5126, nf, "VEC3",
                      bt.min(0).tolist(), bt.max(0).tolist())
            sms.append({"input": ta, "output": bta, "interpolation": "LINEAR"})
            chs.append({"sampler": si, "target": {"node": bi + 1, "path": "translation"}})
            si += 1

            br  = bone_rots[bi]
            bra = acc(add(br.tobytes()), 5126, nf, "VEC4",
                      br.min(0).tolist(), br.max(0).tolist())
            sms.append({"input": ta, "output": bra, "interpolation": "LINEAR"})
            chs.append({"sampler": si, "target": {"node": bi + 1, "path": "rotation"}})
            si += 1

        # Nodes: mesh node + one node per bone
        mn  = {"name": "mesh", "mesh": 0, "skin": 0}
        bns = []
        for bi, bd in enumerate(bone_hierarchy):
            lp = bone_positions[bi].copy()
            if bd.parent_idx is not None:
                lp = bone_positions[bi] - bone_positions[bd.parent_idx]
            nd = {"name": bd.name, "translation": lp.tolist()}
            ch = [j for j, b in enumerate(bone_hierarchy) if b.parent_idx == bi]
            if ch:
                nd["children"] = [c + 1 for c in ch]
            bns.append(nd)

        nodes = [mn] + bns
        roots = [bi + 1 for bi, b in enumerate(bone_hierarchy) if b.parent_idx is None]

        gltf = {
            "asset":      {"version": "2.0", "generator": "ModelGenerator"},
            "scene":      0,
            "scenes":     [{"nodes": [0] + roots}],
            "nodes":      nodes,
            "meshes":     [{"primitives": [{"attributes": {
                "POSITION": pa, "NORMAL": na,
                "JOINTS_0": ja, "WEIGHTS_0": wa
            }, "indices": ia}]}],
            "skins":      [{"joints": list(range(1, nb + 1)),
                            "inverseBindMatrices": ima, "skeleton": 1}],
            "animations": [{"name": anim_name, "channels": chs, "samplers": sms}],
            "buffers":    [{"byteLength": len(buf)}],
            "bufferViews": bvs,
            "accessors":   accs,
        }

        # Optional vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            try:
                c  = mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0
                ca = acc(add(c.tobytes(), 34962), 5126, nv, "VEC4")
                gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = ca
            except Exception:
                pass

        pad()
        gltf["buffers"][0]["byteLength"] = len(buf)
        jb = json.dumps(gltf, separators=(",", ":")).encode()
        while len(jb) % 4:
            jb += b" "

        tot = 12 + 8 + len(jb) + 8 + len(buf)
        o = bytearray()
        o.extend(struct.pack("<III", 0x46546C67, 2, tot))
        o.extend(struct.pack("<II", len(jb), 0x4E4F534A))
        o.extend(jb)
        o.extend(struct.pack("<II", len(buf), 0x004E4942))
        o.extend(buf)
        return bytes(o)

    def unload_model(self):
        pass
