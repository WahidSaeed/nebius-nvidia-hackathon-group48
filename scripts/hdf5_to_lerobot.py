"""Convert Isaac Lab ``record_demos.py`` HDF5 recordings (leader-arm teleop) to LeRobot v2.1.

    python scripts/hdf5_to_lerobot.py --hdf5 datasets/teleop.hdf5 --dataset_root <dir> --task_text "..."

Expects the ``Isaac-PickPlace-Cube-SO101-Joint-Teleop-v0`` layout: ``obs/joint_pos`` (T,6 rad),
``obs/joint_pos_target`` (T,6 rad), ``obs/front`` / ``obs/wrist`` (T,H,W,3 uint8). Actions are the
commanded joint targets (what the leader arm sent), converted to LeRobot normalised units.
Runs in the Isaac venv but does not start the simulator.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from so101_pickplace.lerobot_writer import LeRobotV21Writer  # noqa: E402
from so101_pickplace.so101_units import FEATURE_NAMES, rad_to_norm  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hdf5", required=True)
parser.add_argument("--dataset_root", required=True)
parser.add_argument("--task_text", default="pick up the red block and place it on the green pad")
parser.add_argument("--state_key", default="obs/joint_pos")
parser.add_argument("--action_key", default="obs/joint_pos_target", help="Fallback: 'actions' (raw action tensor).")
parser.add_argument("--front_key", default="obs/front")
parser.add_argument("--wrist_key", default="obs/wrist")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--only_success", action="store_true", default=True)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

with h5py.File(args.hdf5, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    first = f["data"][demos[0]]
    H, W = first[args.front_key].shape[1:3]
    writer = LeRobotV21Writer(
        args.dataset_root, task=args.task_text, fps=args.fps, video_keys=("front", "wrist"),
        image_hw=(H, W), feature_names=FEATURE_NAMES, resume=args.resume,
    )
    kept = 0
    for d in demos:
        g = f["data"][d]
        if args.only_success and not bool(g.attrs.get("success", True)):
            print(f"{d}: skipped (success=False)")
            continue
        state = np.asarray(g[args.state_key])
        action = np.asarray(g[args.action_key]) if args.action_key in g else np.asarray(g["actions"])
        front = np.asarray(g[args.front_key])
        wrist = np.asarray(g[args.wrist_key])
        n = min(len(state), len(action), len(front), len(wrist))
        writer.start_episode()
        for t in range(n):
            writer.add_frame(rad_to_norm(state[t]), rad_to_norm(action[t]), {"front": front[t][..., :3], "wrist": wrist[t][..., :3]})
        ep = writer.end_episode()
        kept += 1
        print(f"{d}: {n} frames -> episode {ep}")
    writer.finalize()
    print(f"done: {kept} episodes written to {args.dataset_root}")
