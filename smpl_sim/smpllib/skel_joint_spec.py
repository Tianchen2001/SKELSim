"""Static SKEL metadata used by the SMPLSim adapter.

This file intentionally contains only stable model metadata:
- the 24-body order,
- the kinematic parents,
- the 46 pose-parameter order,
- conservative joint limits,
- and default MuJoCo control gains.

Joint axes are *not* hard-coded here.  SKEL applies shape-dependent T-pose
and A-pose frame transformations around the raw OpenSim axes.  The adapter
therefore infers the effective zero-pose axes numerically from the official
SKEL forward pass.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np


SKEL_BODY_NAMES: Final[list[str]] = [
    "pelvis",
    "femur_r",
    "tibia_r",
    "talus_r",
    "calcn_r",
    "toes_r",
    "femur_l",
    "tibia_l",
    "talus_l",
    "calcn_l",
    "toes_l",
    "lumbar_body",
    "thorax",
    "head",
    "scapula_r",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "scapula_l",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
]

SKEL_PARENT_INDICES: Final[list[int]] = [
    -1,
    0, 1, 2, 3, 4,
    0, 6, 7, 8, 9,
    0, 11, 12,
    12, 14, 15, 16, 17,
    12, 19, 20, 21, 22,
]

SKEL_POSE_NAMES: Final[list[str]] = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_bending",
    "lumbar_extension",
    "lumbar_twist",
    "thorax_bending",
    "thorax_extension",
    "thorax_twist",
    "head_bending",
    "head_extension",
    "head_twist",
    "scapula_abduction_r",
    "scapula_elevation_r",
    "scapula_upward_rot_r",
    "shoulder_r_x",
    "shoulder_r_y",
    "shoulder_r_z",
    "elbow_flexion_r",
    "pro_sup_r",
    "wrist_flexion_r",
    "wrist_deviation_r",
    "scapula_abduction_l",
    "scapula_elevation_l",
    "scapula_upward_rot_l",
    "shoulder_l_x",
    "shoulder_l_y",
    "shoulder_l_z",
    "elbow_flexion_l",
    "pro_sup_l",
    "wrist_flexion_l",
    "wrist_deviation_l",
]

# Pose indices owned by each SKEL body.  The pelvis' three parameters become
# the orientation of MuJoCo's free root; they are not emitted as hinge joints.
SKEL_BODY_POSE_INDICES: Final[dict[str, list[int]]] = {
    "pelvis": [0, 1, 2],
    "femur_r": [3, 4, 5],
    "tibia_r": [6],
    "talus_r": [7],
    "calcn_r": [8],
    "toes_r": [9],
    "femur_l": [10, 11, 12],
    "tibia_l": [13],
    "talus_l": [14],
    "calcn_l": [15],
    "toes_l": [16],
    "lumbar_body": [17, 18, 19],
    "thorax": [20, 21, 22],
    "head": [23, 24, 25],
    "scapula_r": [26, 27, 28],
    "humerus_r": [29, 30, 31],
    "ulna_r": [32],
    "radius_r": [33],
    "hand_r": [34, 35],
    "scapula_l": [36, 37, 38],
    "humerus_l": [39, 40, 41],
    "ulna_l": [42],
    "radius_l": [43],
    "hand_l": [44, 45],
}

# Limits published with the SKEL implementation.  Parameters absent from this
# table use [-pi, pi] for the kinematic MVP.  These limits should be revisited
# before torque-driven simulation.
_PUBLISHED_LIMITS: Final[dict[str, tuple[float, float]]] = {
    "scapula_abduction_r": (-0.628, 0.628),
    "scapula_elevation_r": (-0.4, -0.1),
    "scapula_upward_rot_r": (-0.190, 0.319),
    "scapula_abduction_l": (-0.628, 0.628),
    "scapula_elevation_l": (-0.4, -0.1),
    "scapula_upward_rot_l": (-0.210, 0.219),
    "elbow_flexion_r": (0.0, 0.75 * math.pi),
    "elbow_flexion_l": (0.0, 0.75 * math.pi),
    "pro_sup_r": (-0.375 * math.pi, 0.375 * math.pi),
    "pro_sup_l": (-0.5 * math.pi, 0.5 * math.pi),
    "wrist_flexion_r": (-0.5 * math.pi, 0.5 * math.pi),
    "wrist_flexion_l": (-0.5 * math.pi, 0.5 * math.pi),
    "wrist_deviation_r": (-0.25 * math.pi, 0.25 * math.pi),
    "wrist_deviation_l": (-0.25 * math.pi, 0.25 * math.pi),
    "lumbar_bending": (-math.pi / 6.0, math.pi / 6.0),
    "lumbar_extension": (-math.pi / 4.0, math.pi / 4.0),
    "lumbar_twist": (-math.pi / 4.0, math.pi / 4.0),
    "thorax_bending": (-math.pi / 4.0, math.pi / 4.0),
    "thorax_extension": (-math.pi / 4.0, math.pi / 4.0),
    "thorax_twist": (-math.pi / 4.0, math.pi / 4.0),
    "head_bending": (-math.pi / 4.0, math.pi / 4.0),
    "head_extension": (-math.pi / 4.0, math.pi / 4.0),
    "head_twist": (-math.pi / 4.0, math.pi / 4.0),
    "ankle_angle_r": (-math.pi / 4.0, math.pi / 4.0),
    "ankle_angle_l": (-math.pi / 4.0, math.pi / 4.0),
    "subtalar_angle_r": (-math.pi / 4.0, math.pi / 4.0),
    "subtalar_angle_l": (-math.pi / 4.0, math.pi / 4.0),
    "mtp_angle_r": (-math.pi / 4.0, math.pi / 4.0),
    "mtp_angle_l": (-math.pi / 4.0, math.pi / 4.0),
    "knee_angle_r": (0.0, 0.75 * math.pi),
    "knee_angle_l": (0.0, 0.75 * math.pi),
}


def pose_limit(pose_name: str) -> np.ndarray:
    """Return a conservative [lower, upper] range in radians."""
    lower, upper = _PUBLISHED_LIMITS.get(pose_name, (-math.pi, math.pi))
    if lower > upper:
        lower, upper = upper, lower
    return np.asarray([lower, upper], dtype=np.float64)


# SMPLSim stores four user parameters on each MuJoCo joint.  They are used by
# downstream controllers as [kp, kd, gear-like scale, maximum torque].
# The current kinematic playback does not use these values, but the XML writer
# requires every body name to exist in its gain dictionary.
SKEL_GAINS_MJ: Final[dict[str, list[float]]] = {
    "pelvis": [1000, 100, 1, 500],
    "femur_r": [800, 80, 1, 500],
    "tibia_r": [800, 80, 1, 500],
    "talus_r": [500, 50, 1, 300],
    "calcn_r": [500, 50, 1, 300],
    "toes_r": [300, 30, 1, 150],
    "femur_l": [800, 80, 1, 500],
    "tibia_l": [800, 80, 1, 500],
    "talus_l": [500, 50, 1, 300],
    "calcn_l": [500, 50, 1, 300],
    "toes_l": [300, 30, 1, 150],
    "lumbar_body": [1000, 100, 1, 500],
    "thorax": [1000, 100, 1, 500],
    "head": [500, 50, 1, 250],
    "scapula_r": [500, 50, 1, 250],
    "humerus_r": [500, 50, 1, 250],
    "ulna_r": [400, 40, 1, 150],
    "radius_r": [300, 30, 1, 150],
    "hand_r": [300, 30, 1, 100],
    "scapula_l": [500, 50, 1, 250],
    "humerus_l": [500, 50, 1, 250],
    "ulna_l": [400, 40, 1, 150],
    "radius_l": [300, 30, 1, 150],
    "hand_l": [300, 30, 1, 100],
}


def validate_static_spec() -> None:
    """Fail early if the static metadata is internally inconsistent."""
    if len(SKEL_BODY_NAMES) != 24:
        raise RuntimeError(f"Expected 24 SKEL bodies, got {len(SKEL_BODY_NAMES)}.")
    if len(SKEL_PARENT_INDICES) != 24:
        raise RuntimeError("Parent table must have one entry per body.")
    if len(SKEL_POSE_NAMES) != 46:
        raise RuntimeError(f"Expected 46 pose parameters, got {len(SKEL_POSE_NAMES)}.")

    flattened = [
        index
        for body_name in SKEL_BODY_NAMES
        for index in SKEL_BODY_POSE_INDICES[body_name]
    ]
    if flattened != list(range(46)):
        raise RuntimeError(
            "SKEL_BODY_POSE_INDICES must cover pose indices 0..45 exactly once."
        )

    for body_index, parent_index in enumerate(SKEL_PARENT_INDICES):
        if body_index == 0 and parent_index != -1:
            raise RuntimeError("The pelvis must be the root.")
        if body_index > 0 and not 0 <= parent_index < body_index:
            raise RuntimeError(
                f"Invalid parent {parent_index} for body index {body_index}."
            )


validate_static_spec()
