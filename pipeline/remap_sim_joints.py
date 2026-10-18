#!/usr/bin/env python3
"""Map sim joint values (so101_units.py convention) into the station's real-arm calibration convention.

Fitted from data: first-frame poses + time-normalised mean trajectories of 180 sim vs 25 real demos
(shoulder_lift / elbow_flex start poses are mirror images: -34.4 vs +49.7, +47.2 vs -46.7;
pan / wrist_roll trajectories anti-correlate at -0.70 / -0.77).  real = SIGN * sim + OFFSET.

  python pipeline/remap_sim_joints.py <v3 dataset root> [<root> ...]     # in place, state + action
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

#                pan    lift   elbow  wflex  wroll  gripper
SIGN = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
OFFSET = np.array([6.7, 15.3, 0.5, 3.3, 11.1, 0.0], dtype=np.float32)


def remap(root: str):
    for f in sorted(glob.glob(f"{root}/data/*/*.parquet")):
        df = pd.read_parquet(f)
        for col in ("observation.state", "action"):
            v = np.stack(df[col].to_numpy()).astype(np.float32) * SIGN + OFFSET
            df[col] = list(np.clip(v, -100, 100))
        df.to_parquet(f)
    info_p = f"{root}/meta/info.json"
    info = json.load(open(info_p))
    info["sim_to_real_joint_remap"] = {"sign": SIGN.tolist(), "offset": OFFSET.tolist()}
    json.dump(info, open(info_p, "w"), indent=4)
    print("remapped", root)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        remap(r)
