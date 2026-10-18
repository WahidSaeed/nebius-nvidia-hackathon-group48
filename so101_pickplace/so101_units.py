"""Joint naming and unit conventions shared by the collector, converter and eval client.

The dataset stores joints in the LeRobot SO-101 convention (``use_degrees=False``):
arm joints are mapped linearly from their motion range to ``[-100, 100]`` and the gripper
from closed..open to ``[0, 100]``. GR00T normalises with dataset statistics anyway, so the
unit choice only matters for consistency between the dataset and the eval client -- and for
making a sim-trained checkpoint drop-in compatible with the real-robot client in
``~/Desktop/nvidia/so101_demo/eval_so101_demo.py`` (same feature names, same ranges).

The sim ranges below are the joint limits of the ``so101_new_calib`` USD (verified in sim by
``scripts/view_env.py``). They are deliberately constants rather than read from the asset at
runtime so the mapping can never drift between data collection and evaluation.
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS: list[str] = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS: list[str] = ARM_JOINTS + ["gripper"]
FEATURE_NAMES: list[str] = [f"{j}.pos" for j in JOINTS]

# Gripper jaw joint targets [rad]. Matches the contrib stack task constants.
GRIPPER_OPEN_RAD = 1.745
GRIPPER_CLOSE_RAD = -0.1745  # USD lower joint limit

# Arm joint ranges [rad] used for the [-100, 100] mapping (lower, upper).
# These are the USD joint limits; see scripts/view_env.py output ("joint limits").
ARM_RANGES_RAD: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69000, 1.69000),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.74385),
}

_LO = np.array([ARM_RANGES_RAD[j][0] for j in ARM_JOINTS] + [GRIPPER_CLOSE_RAD], dtype=np.float64)
_HI = np.array([ARM_RANGES_RAD[j][1] for j in ARM_JOINTS] + [GRIPPER_OPEN_RAD], dtype=np.float64)
_SCALE = np.array([200.0] * 5 + [100.0])
_OFFSET = np.array([-100.0] * 5 + [0.0])


def rad_to_norm(q_rad: np.ndarray) -> np.ndarray:
    """Map joint positions [rad], ordered as :data:`JOINTS`, to LeRobot normalised units."""
    q = np.asarray(q_rad, dtype=np.float64)
    return ((q - _LO) / (_HI - _LO) * _SCALE + _OFFSET).astype(np.float32)


def norm_to_rad(q_norm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rad_to_norm`; clips to the valid normalised range first."""
    q = np.asarray(q_norm, dtype=np.float64)
    q = np.clip(q, _OFFSET, _OFFSET + _SCALE)
    return ((q - _OFFSET) / _SCALE * (_HI - _LO) + _LO).astype(np.float32)
