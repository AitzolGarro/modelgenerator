"""
Generates synthetic CMU-format BVH files for mocap animation testing.
Run once: python generate_bvh.py
"""
import math
import os

# CMU skeleton hierarchy
CMU_HIERARCHY = """\
HIERARCHY
ROOT Hips
{
	CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
	JOINT LHipJoint
	{
		OFFSET 0.00	0.00	0.00
		CHANNELS 3 Zrotation Xrotation Yrotation
		JOINT LeftUpLeg
		{
			OFFSET 2.32	-0.00	-0.00
			CHANNELS 3 Zrotation Xrotation Yrotation
			JOINT LeftLeg
			{
				OFFSET 2.04	-7.71	-0.00
				CHANNELS 3 Zrotation Xrotation Yrotation
				JOINT LeftFoot
				{
					OFFSET 0.00	-14.08	0.00
					CHANNELS 3 Zrotation Xrotation Yrotation
					JOINT LeftToeBase
					{
						OFFSET 0.00	-0.00	4.75
						CHANNELS 3 Zrotation Xrotation Yrotation
						End Site
						{
							OFFSET 0.00	0.00	3.12
						}
					}
				}
			}
		}
	}
	JOINT RHipJoint
	{
		OFFSET 0.00	0.00	0.00
		CHANNELS 3 Zrotation Xrotation Yrotation
		JOINT RightUpLeg
		{
			OFFSET -2.32	-0.00	-0.00
			CHANNELS 3 Zrotation Xrotation Yrotation
			JOINT RightLeg
			{
				OFFSET -2.04	-7.71	-0.00
				CHANNELS 3 Zrotation Xrotation Yrotation
				JOINT RightFoot
				{
					OFFSET 0.00	-14.08	0.00
					CHANNELS 3 Zrotation Xrotation Yrotation
					JOINT RightToeBase
					{
						OFFSET 0.00	-0.00	4.75
						CHANNELS 3 Zrotation Xrotation Yrotation
						End Site
						{
							OFFSET 0.00	0.00	3.12
						}
					}
				}
			}
		}
	}
	JOINT LowerBack
	{
		OFFSET 0.00	0.00	0.00
		CHANNELS 3 Zrotation Xrotation Yrotation
		JOINT Spine
		{
			OFFSET 0.00	5.69	0.00
			CHANNELS 3 Zrotation Xrotation Yrotation
			JOINT Spine1
			{
				OFFSET 0.00	5.69	0.00
				CHANNELS 3 Zrotation Xrotation Yrotation
				JOINT Neck
				{
					OFFSET 0.00	5.69	0.00
					CHANNELS 3 Zrotation Xrotation Yrotation
					JOINT Neck1
					{
						OFFSET 0.00	2.84	0.00
						CHANNELS 3 Zrotation Xrotation Yrotation
						JOINT Head
						{
							OFFSET 0.00	2.84	0.00
							CHANNELS 3 Zrotation Xrotation Yrotation
							End Site
							{
								OFFSET 0.00	7.10	0.00
							}
						}
					}
				}
				JOINT LeftShoulder
				{
					OFFSET 3.44	5.69	0.00
					CHANNELS 3 Zrotation Xrotation Yrotation
					JOINT LeftArm
					{
						OFFSET 5.08	0.00	0.00
						CHANNELS 3 Zrotation Xrotation Yrotation
						JOINT LeftForeArm
						{
							OFFSET 0.00	-11.48	0.00
							CHANNELS 3 Zrotation Xrotation Yrotation
							JOINT LeftHand
							{
								OFFSET 0.00	-9.80	0.00
								CHANNELS 3 Zrotation Xrotation Yrotation
								End Site
								{
									OFFSET 0.00	-4.80	0.00
								}
							}
						}
					}
				}
				JOINT RightShoulder
				{
					OFFSET -3.44	5.69	0.00
					CHANNELS 3 Zrotation Xrotation Yrotation
					JOINT RightArm
					{
						OFFSET -5.08	0.00	0.00
						CHANNELS 3 Zrotation Xrotation Yrotation
						JOINT RightForeArm
						{
							OFFSET 0.00	-11.48	0.00
							CHANNELS 3 Zrotation Xrotation Yrotation
							JOINT RightHand
							{
								OFFSET 0.00	-9.80	0.00
								CHANNELS 3 Zrotation Xrotation Yrotation
								End Site
								{
									OFFSET 0.00	-4.80	0.00
								}
							}
						}
					}
				}
			}
		}
	}
}
"""

# Channel order for each joint matches the hierarchy above
# Root: X Y Z Zrot Xrot Yrot  → first 3 = translation, last 3 = ZXY rotation
# Others: Zrot Xrot Yrot

# Joint order as they appear in the hierarchy (matches channel count)
JOINT_NAMES = [
    "Hips",          # root: 6ch (XYZ + ZXY rot)
    "LHipJoint",     # 3ch ZXY
    "LeftUpLeg",     # 3ch ZXY
    "LeftLeg",       # 3ch ZXY
    "LeftFoot",      # 3ch ZXY
    "LeftToeBase",   # 3ch ZXY
    "RHipJoint",     # 3ch ZXY
    "RightUpLeg",    # 3ch ZXY
    "RightLeg",      # 3ch ZXY
    "RightFoot",     # 3ch ZXY
    "RightToeBase",  # 3ch ZXY
    "LowerBack",     # 3ch ZXY
    "Spine",         # 3ch ZXY
    "Spine1",        # 3ch ZXY
    "Neck",          # 3ch ZXY
    "Neck1",         # 3ch ZXY
    "Head",          # 3ch ZXY
    "LeftShoulder",  # 3ch ZXY
    "LeftArm",       # 3ch ZXY
    "LeftForeArm",   # 3ch ZXY
    "LeftHand",      # 3ch ZXY
    "RightShoulder", # 3ch ZXY
    "RightArm",      # 3ch ZXY
    "RightForeArm",  # 3ch ZXY
    "RightHand",     # 3ch ZXY
]

# Indices into JOINT_NAMES for easy access
HIP_IDX         = 0
LHIPJOINT_IDX   = 1
L_UPLEG_IDX     = 2
L_LEG_IDX       = 3
L_FOOT_IDX      = 4
L_TOEBASE_IDX   = 5
RHIPJOINT_IDX   = 6
R_UPLEG_IDX     = 7
R_LEG_IDX       = 8
R_FOOT_IDX      = 9
R_TOEBASE_IDX   = 10
LOWERBACK_IDX   = 11
SPINE_IDX       = 12
SPINE1_IDX      = 13
NECK_IDX        = 14
NECK1_IDX       = 15
HEAD_IDX        = 16
L_SHOULDER_IDX  = 17
L_ARM_IDX       = 18
L_FOREARM_IDX   = 19
L_HAND_IDX      = 20
R_SHOULDER_IDX  = 21
R_ARM_IDX       = 22
R_FOREARM_IDX   = 23
R_HAND_IDX      = 24

def ss(phase):
    """Smooth sine."""
    return math.sin(phase) * (1.0 - 0.1 * math.sin(phase * 2))

def make_frame_walk(t, cycle=1.0):
    """Walk cycle: realistic biped gait."""
    p = (t / cycle) * 2 * math.pi
    # 25 joints: root=6ch, rest=3ch → 6 + 24*3 = 78 values
    v = [0.0] * 78

    # Root translation (indices 0,1,2) + rotation ZXY (3,4,5)
    hip_bob = abs(ss(p * 2)) * 0.8   # vertical bob
    hip_sway = ss(p) * 0.6           # lateral sway
    v[0] = hip_sway    # Xpos (lateral)
    v[1] = 94.0 + hip_bob  # Ypos (height ~94cm)
    v[2] = 0.0         # Zpos (no forward translation for cycle)
    v[3] = ss(p) * 1.5  # Zrot
    v[4] = 0.0          # Xrot
    v[5] = ss(p) * 2.0  # Yrot (hip rotation)

    def set3(idx, z, x, y):
        base = 6 + idx * 3
        v[base] = z; v[base+1] = x; v[base+2] = y

    # LHipJoint
    set3(LHIPJOINT_IDX - 1, 0, 0, 0)
    # LeftUpLeg: forward swing
    set3(L_UPLEG_IDX - 1, ss(p) * 2.0, ss(p) * 25.0, 2.0)
    # LeftLeg: knee bend (only forward)
    knee_l = max(0, ss(p - 0.5)) * 30.0
    set3(L_LEG_IDX - 1, 0, knee_l, 0)
    # LeftFoot: ankle flex
    set3(L_FOOT_IDX - 1, 0, -ss(p) * 10.0, 0)
    set3(L_TOEBASE_IDX - 1, 0, 0, 0)

    # RHipJoint
    set3(RHIPJOINT_IDX - 1, 0, 0, 0)
    # RightUpLeg: opposite phase
    set3(R_UPLEG_IDX - 1, -ss(p) * 2.0, ss(p + math.pi) * 25.0, -2.0)
    knee_r = max(0, ss(p + math.pi - 0.5)) * 30.0
    set3(R_LEG_IDX - 1, 0, knee_r, 0)
    set3(R_FOOT_IDX - 1, 0, -ss(p + math.pi) * 10.0, 0)
    set3(R_TOEBASE_IDX - 1, 0, 0, 0)

    # LowerBack
    set3(LOWERBACK_IDX - 1, ss(p) * 0.5, 0, -ss(p) * 1.5)
    # Spine
    set3(SPINE_IDX - 1, ss(p) * 0.3, 0, -ss(p) * 1.0)
    # Spine1
    set3(SPINE1_IDX - 1, 0, 0, ss(p) * 0.5)
    # Neck
    set3(NECK_IDX - 1, 0, 0, -ss(p) * 0.5)
    # Neck1
    set3(NECK1_IDX - 1, 0, 0, 0)
    # Head
    set3(HEAD_IDX - 1, 0, ss(p * 2) * 1.0, 0)

    # LeftShoulder
    set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
    # LeftArm: opposite to left leg
    set3(L_ARM_IDX - 1, 0, ss(p + math.pi) * 18.0, 0)
    # LeftForeArm
    set3(L_FOREARM_IDX - 1, 0, max(0, ss(p + math.pi - 0.2)) * 12.0, 0)
    set3(L_HAND_IDX - 1, 0, 0, 0)

    # RightShoulder
    set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
    # RightArm: opposite to right leg
    set3(R_ARM_IDX - 1, 0, ss(p) * 18.0, 0)
    set3(R_FOREARM_IDX - 1, 0, max(0, ss(p - 0.2)) * 12.0, 0)
    set3(R_HAND_IDX - 1, 0, 0, 0)

    return v


def make_frame_run(t, cycle=0.6):
    """Run cycle: faster, higher knees, more arm swing."""
    p = (t / cycle) * 2 * math.pi
    v = [0.0] * 78

    hip_bob = abs(ss(p * 2)) * 1.5
    v[0] = ss(p) * 0.8
    v[1] = 94.0 + hip_bob
    v[2] = 0.0
    v[3] = ss(p) * 2.5
    v[4] = -3.0
    v[5] = ss(p) * 3.5

    def set3(idx, z, x, y):
        base = 6 + idx * 3
        v[base] = z; v[base+1] = x; v[base+2] = y

    set3(LHIPJOINT_IDX - 1, 0, 0, 0)
    set3(L_UPLEG_IDX - 1, ss(p) * 3.0, ss(p) * 40.0, 2.5)
    knee_l = max(0, ss(p - 0.4)) * 55.0
    set3(L_LEG_IDX - 1, 0, knee_l, 0)
    set3(L_FOOT_IDX - 1, 0, -ss(p) * 15.0, 0)
    set3(L_TOEBASE_IDX - 1, 0, 0, 0)

    set3(RHIPJOINT_IDX - 1, 0, 0, 0)
    set3(R_UPLEG_IDX - 1, -ss(p) * 3.0, ss(p + math.pi) * 40.0, -2.5)
    knee_r = max(0, ss(p + math.pi - 0.4)) * 55.0
    set3(R_LEG_IDX - 1, 0, knee_r, 0)
    set3(R_FOOT_IDX - 1, 0, -ss(p + math.pi) * 15.0, 0)
    set3(R_TOEBASE_IDX - 1, 0, 0, 0)

    set3(LOWERBACK_IDX - 1, ss(p) * 0.8, -1.5, -ss(p) * 2.5)
    set3(SPINE_IDX - 1, ss(p) * 0.5, -1.0, -ss(p) * 1.5)
    set3(SPINE1_IDX - 1, 0, -0.5, ss(p) * 0.8)
    set3(NECK_IDX - 1, 0, 2.0, -ss(p) * 0.8)
    set3(NECK1_IDX - 1, 0, 1.0, 0)
    set3(HEAD_IDX - 1, 0, ss(p * 2) * 1.5, 0)

    set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
    set3(L_ARM_IDX - 1, 0, ss(p + math.pi) * 32.0, 0)
    set3(L_FOREARM_IDX - 1, 0, 25.0 + ss(p + math.pi) * 15.0, 0)
    set3(L_HAND_IDX - 1, 0, 0, 0)

    set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
    set3(R_ARM_IDX - 1, 0, ss(p) * 32.0, 0)
    set3(R_FOREARM_IDX - 1, 0, 25.0 + ss(p) * 15.0, 0)
    set3(R_HAND_IDX - 1, 0, 0, 0)

    return v


def make_frame_idle(t, cycle=3.0):
    """Idle: standing with breathing and subtle sways."""
    p = (t / cycle) * 2 * math.pi
    v = [0.0] * 78

    breath = math.sin(p)
    v[0] = 0.0
    v[1] = 93.5 + breath * 0.3
    v[2] = 0.0
    v[3] = 0.0; v[4] = 0.0; v[5] = 0.0

    def set3(idx, z, x, y):
        base = 6 + idx * 3
        v[base] = z; v[base+1] = x; v[base+2] = y

    set3(LHIPJOINT_IDX - 1, 0, 0, 0)
    set3(L_UPLEG_IDX - 1, 0, breath * 0.3, 0)
    set3(L_LEG_IDX - 1, 0, breath * 0.2, 0)
    set3(L_FOOT_IDX - 1, 0, 0, 0)
    set3(L_TOEBASE_IDX - 1, 0, 0, 0)

    set3(RHIPJOINT_IDX - 1, 0, 0, 0)
    set3(R_UPLEG_IDX - 1, 0, breath * 0.3, 0)
    set3(R_LEG_IDX - 1, 0, breath * 0.2, 0)
    set3(R_FOOT_IDX - 1, 0, 0, 0)
    set3(R_TOEBASE_IDX - 1, 0, 0, 0)

    set3(LOWERBACK_IDX - 1, 0, breath * 0.3, 0)
    set3(SPINE_IDX - 1, 0, breath * 0.5, 0)
    set3(SPINE1_IDX - 1, 0, breath * 0.8, 0)
    set3(NECK_IDX - 1, 0, breath * -0.3, 0)
    set3(NECK1_IDX - 1, 0, 0, 0)
    set3(HEAD_IDX - 1, 0, math.sin(p * 0.5) * 1.0, 0)

    set3(L_SHOULDER_IDX - 1, 5.0 + breath * 0.2, 0, 0)
    set3(L_ARM_IDX - 1, breath * 0.5, 0, 0)
    set3(L_FOREARM_IDX - 1, breath * 0.3, 0, 0)
    set3(L_HAND_IDX - 1, 0, 0, 0)

    set3(R_SHOULDER_IDX - 1, -5.0 - breath * 0.2, 0, 0)
    set3(R_ARM_IDX - 1, -breath * 0.5, 0, 0)
    set3(R_FOREARM_IDX - 1, -breath * 0.3, 0, 0)
    set3(R_HAND_IDX - 1, 0, 0, 0)

    return v


def make_frame_jump(t, cycle=1.2):
    """Jump: crouch → launch → apex → land."""
    p = (t / cycle) * 2 * math.pi
    tn = (t % cycle) / cycle
    v = [0.0] * 78

    # Vertical position: sine arch
    height_offset = max(0, math.sin(tn * math.pi)) * 25.0
    v[0] = 0.0
    v[1] = 93.5 + height_offset
    v[2] = 0.0
    v[3] = 0.0; v[4] = 0.0; v[5] = 0.0

    # Crouch at start/end, extend at apex
    if tn < 0.2:
        e = tn / 0.2
        crouch = (1 - e) * 0.5
    elif tn > 0.8:
        e = (tn - 0.8) / 0.2
        crouch = e * 0.5
    else:
        crouch = 0.0

    def set3(idx, z, x, y):
        base = 6 + idx * 3
        v[base] = z; v[base+1] = x; v[base+2] = y

    set3(LHIPJOINT_IDX - 1, 0, 0, 0)
    set3(L_UPLEG_IDX - 1, 0, -crouch * 25.0, 0)
    set3(L_LEG_IDX - 1, 0, crouch * 45.0, 0)
    set3(L_FOOT_IDX - 1, 0, crouch * 20.0, 0)
    set3(L_TOEBASE_IDX - 1, 0, 0, 0)

    set3(RHIPJOINT_IDX - 1, 0, 0, 0)
    set3(R_UPLEG_IDX - 1, 0, -crouch * 25.0, 0)
    set3(R_LEG_IDX - 1, 0, crouch * 45.0, 0)
    set3(R_FOOT_IDX - 1, 0, crouch * 20.0, 0)
    set3(R_TOEBASE_IDX - 1, 0, 0, 0)

    set3(LOWERBACK_IDX - 1, 0, -crouch * 5.0, 0)
    set3(SPINE_IDX - 1, 0, -crouch * 5.0, 0)
    set3(SPINE1_IDX - 1, 0, -crouch * 3.0, 0)
    set3(NECK_IDX - 1, 0, crouch * 2.0, 0)
    set3(NECK1_IDX - 1, 0, 0, 0)
    set3(HEAD_IDX - 1, 0, crouch * 2.0, 0)

    arm_raise = max(0, math.sin(tn * math.pi)) * 30.0
    set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
    set3(L_ARM_IDX - 1, arm_raise * 0.5, arm_raise, 0)
    set3(L_FOREARM_IDX - 1, 0, -arm_raise * 0.3, 0)
    set3(L_HAND_IDX - 1, 0, 0, 0)

    set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
    set3(R_ARM_IDX - 1, -arm_raise * 0.5, arm_raise, 0)
    set3(R_FOREARM_IDX - 1, 0, -arm_raise * 0.3, 0)
    set3(R_HAND_IDX - 1, 0, 0, 0)

    return v


def make_frame_attack(t, cycle=1.0):
    """Attack: wind-up then right punch strike."""
    p = (t / cycle) * 2 * math.pi
    tn = (t % cycle) / cycle
    v = [0.0] * 78

    v[0] = 0.0; v[1] = 93.5; v[2] = 0.0
    v[3] = 0.0; v[4] = 0.0; v[5] = 0.0

    def ease(t_):
        return 4*t_*t_*t_ if t_ < 0.5 else 1 - (-2*t_+2)**3/2

    def set3(idx, z, x, y):
        base = 6 + idx * 3
        v[base] = z; v[base+1] = x; v[base+2] = y

    if tn < 0.3:
        e = ease(tn / 0.3)
        # Wind-up: turn right
        set3(LHIPJOINT_IDX - 1, 0, 0, 0)
        set3(L_UPLEG_IDX - 1, 0, e * 3.0, 0)
        set3(L_LEG_IDX - 1, 0, e * 2.0, 0)
        set3(L_FOOT_IDX - 1, 0, 0, 0)
        set3(L_TOEBASE_IDX - 1, 0, 0, 0)

        set3(RHIPJOINT_IDX - 1, 0, 0, 0)
        set3(R_UPLEG_IDX - 1, 0, -e * 3.0, 0)
        set3(R_LEG_IDX - 1, 0, e * 2.0, 0)
        set3(R_FOOT_IDX - 1, 0, 0, 0)
        set3(R_TOEBASE_IDX - 1, 0, 0, 0)

        set3(LOWERBACK_IDX - 1, e * 8.0, 0, 0)
        set3(SPINE_IDX - 1, e * 6.0, 0, 0)
        set3(SPINE1_IDX - 1, e * 5.0, 0, 0)
        set3(NECK_IDX - 1, 0, 0, 0)
        set3(NECK1_IDX - 1, 0, 0, 0)
        set3(HEAD_IDX - 1, 0, 0, 0)

        # Left arm guard position
        set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
        set3(L_ARM_IDX - 1, 0, e * 40.0, 0)
        set3(L_FOREARM_IDX - 1, 0, e * 60.0, 0)
        set3(L_HAND_IDX - 1, 0, 0, 0)

        # Right arm cocking back
        set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
        set3(R_ARM_IDX - 1, 0, -e * 60.0, 0)
        set3(R_FOREARM_IDX - 1, 0, -e * 40.0, 0)
        set3(R_HAND_IDX - 1, 0, 0, 0)

    elif tn < 0.55:
        e = ease((tn - 0.3) / 0.25)
        # Strike forward
        set3(LHIPJOINT_IDX - 1, 0, 0, 0)
        set3(L_UPLEG_IDX - 1, 0, 3.0 - e * 3.0, 0)
        set3(L_LEG_IDX - 1, 0, 2.0, 0)
        set3(L_FOOT_IDX - 1, 0, 0, 0)
        set3(L_TOEBASE_IDX - 1, 0, 0, 0)

        set3(RHIPJOINT_IDX - 1, 0, 0, 0)
        set3(R_UPLEG_IDX - 1, 0, -3.0 + e * 3.0, 0)
        set3(R_LEG_IDX - 1, 0, 2.0, 0)
        set3(R_FOOT_IDX - 1, 0, 0, 0)
        set3(R_TOEBASE_IDX - 1, 0, 0, 0)

        set3(LOWERBACK_IDX - 1, 8.0 - e * 16.0, 0, 0)
        set3(SPINE_IDX - 1, 6.0 - e * 12.0, 0, 0)
        set3(SPINE1_IDX - 1, 5.0 - e * 10.0, 0, 0)
        set3(NECK_IDX - 1, 0, 0, 0)
        set3(NECK1_IDX - 1, 0, 0, 0)
        set3(HEAD_IDX - 1, 0, 0, 0)

        set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
        set3(L_ARM_IDX - 1, 0, 40.0 - e * 35.0, 0)
        set3(L_FOREARM_IDX - 1, 0, 60.0 - e * 50.0, 0)
        set3(L_HAND_IDX - 1, 0, 0, 0)

        # Right arm punching forward
        set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
        set3(R_ARM_IDX - 1, 0, -60.0 + e * 110.0, 0)
        set3(R_FOREARM_IDX - 1, 0, -40.0 + e * 60.0, 0)
        set3(R_HAND_IDX - 1, 0, 0, 0)

    else:
        e = ease((tn - 0.55) / 0.45)
        # Recovery
        set3(LHIPJOINT_IDX - 1, 0, 0, 0)
        set3(L_UPLEG_IDX - 1, 0, 0, 0)
        set3(L_LEG_IDX - 1, 0, 2.0 * (1-e), 0)
        set3(L_FOOT_IDX - 1, 0, 0, 0)
        set3(L_TOEBASE_IDX - 1, 0, 0, 0)

        set3(RHIPJOINT_IDX - 1, 0, 0, 0)
        set3(R_UPLEG_IDX - 1, 0, 0, 0)
        set3(R_LEG_IDX - 1, 0, 2.0 * (1-e), 0)
        set3(R_FOOT_IDX - 1, 0, 0, 0)
        set3(R_TOEBASE_IDX - 1, 0, 0, 0)

        set3(LOWERBACK_IDX - 1, -8.0 * (1-e), 0, 0)
        set3(SPINE_IDX - 1, -6.0 * (1-e), 0, 0)
        set3(SPINE1_IDX - 1, -5.0 * (1-e), 0, 0)
        set3(NECK_IDX - 1, 0, 0, 0)
        set3(NECK1_IDX - 1, 0, 0, 0)
        set3(HEAD_IDX - 1, 0, 0, 0)

        set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
        set3(L_ARM_IDX - 1, 0, 5.0 * (1-e), 0)
        set3(L_FOREARM_IDX - 1, 0, 10.0 * (1-e), 0)
        set3(L_HAND_IDX - 1, 0, 0, 0)

        set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
        set3(R_ARM_IDX - 1, 0, 50.0 * (1-e), 0)
        set3(R_FOREARM_IDX - 1, 0, 20.0 * (1-e), 0)
        set3(R_HAND_IDX - 1, 0, 0, 0)

    return v


def make_frame_dance(t, cycle=1.6):
    """Dance: hip sway with arm flourishes."""
    p = (t / cycle) * 2 * math.pi
    v = [0.0] * 78

    hip_sway = ss(p) * 4.0
    hip_bob = abs(ss(p * 2)) * 1.2
    v[0] = hip_sway * 0.1
    v[1] = 93.5 + hip_bob
    v[2] = 0.0
    v[3] = ss(p) * 3.0
    v[4] = 0.0
    v[5] = ss(p) * 8.0

    def set3(idx, z, x, y):
        base = 6 + idx * 3
        v[base] = z; v[base+1] = x; v[base+2] = y

    set3(LHIPJOINT_IDX - 1, 0, 0, 0)
    set3(L_UPLEG_IDX - 1, ss(p) * 2.0, ss(p) * 15.0, 2.0)
    set3(L_LEG_IDX - 1, 0, max(0, ss(p - 0.3)) * 20.0, 0)
    set3(L_FOOT_IDX - 1, 0, ss(p) * 8.0, 0)
    set3(L_TOEBASE_IDX - 1, 0, 0, 0)

    set3(RHIPJOINT_IDX - 1, 0, 0, 0)
    set3(R_UPLEG_IDX - 1, -ss(p) * 2.0, ss(p + math.pi) * 15.0, -2.0)
    set3(R_LEG_IDX - 1, 0, max(0, ss(p + math.pi - 0.3)) * 20.0, 0)
    set3(R_FOOT_IDX - 1, 0, ss(p + math.pi) * 8.0, 0)
    set3(R_TOEBASE_IDX - 1, 0, 0, 0)

    set3(LOWERBACK_IDX - 1, ss(p) * 3.0, 0, -ss(p) * 5.0)
    set3(SPINE_IDX - 1, ss(p) * 2.0, 0, -ss(p) * 3.0)
    set3(SPINE1_IDX - 1, 0, 0, ss(p + 0.5) * 4.0)
    set3(NECK_IDX - 1, 0, 0, ss(p) * 2.0)
    set3(NECK1_IDX - 1, 0, 0, 0)
    set3(HEAD_IDX - 1, 0, ss(p * 2) * 3.0, ss(p) * 4.0)

    # Left arm: raised and flowing
    set3(L_SHOULDER_IDX - 1, 5.0, 0, 0)
    set3(L_ARM_IDX - 1, 15.0 + ss(p) * 20.0, 30.0 + ss(p) * 15.0, 0)
    set3(L_FOREARM_IDX - 1, 0, -15.0 + ss(p * 2) * 20.0, 0)
    set3(L_HAND_IDX - 1, ss(p * 2) * 10.0, 0, 0)

    # Right arm: opposite flourish
    set3(R_SHOULDER_IDX - 1, -5.0, 0, 0)
    set3(R_ARM_IDX - 1, -15.0 + ss(p + 1.5) * 20.0, 30.0 + ss(p + 1.5) * 15.0, 0)
    set3(R_FOREARM_IDX - 1, 0, -15.0 + ss(p * 2 + 1.0) * 20.0, 0)
    set3(R_HAND_IDX - 1, -ss(p * 2) * 10.0, 0, 0)

    return v


def write_bvh(filename, num_frames, frame_time, make_frame_fn, cycle):
    lines = [CMU_HIERARCHY.strip()]
    lines.append(f"MOTION")
    lines.append(f"Frames: {num_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")
    for f in range(num_frames):
        t = f * frame_time
        values = make_frame_fn(t, cycle)
        lines.append(" ".join(f"{x:.4f}" for x in values))
    # Add duplicate last frame (CMU convention)
    values = make_frame_fn(0.0, cycle)
    lines.append(" ".join(f"{x:.4f}" for x in values))
    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Written: {filename} ({num_frames} frames)")


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    frame_time = 1.0 / 30.0  # 30fps

    clips = [
        ("walk.bvh",   60, make_frame_walk,   1.0),
        ("run.bvh",    45, make_frame_run,    0.6),
        ("idle.bvh",   90, make_frame_idle,   3.0),
        ("jump.bvh",   36, make_frame_jump,   1.2),
        ("attack.bvh", 30, make_frame_attack, 1.0),
        ("dance.bvh",  80, make_frame_dance,  1.6),
    ]

    for fname, nframes, fn, cycle in clips:
        write_bvh(os.path.join(out, fname), nframes, frame_time, fn, cycle)
