#!/usr/bin/env python3
"""Run an ENSEMBLE of ACT checkpoints on the real SO-101 follower (front camera only).

Every control step each policy predicts the next joint target from the same observation (through its
own pre/post-processors, so each uses its own normalisation stats); the targets are averaged with the
given weights and sent to the arm. With one checkpoint this is a plain ACT rollout.

  python pipeline/ensemble_rollout.py --ckpt ckpts/act_main ckpts/act_main_real --weights 0.5 0.5 \
      --port /dev/tty.usbmodem5C821065561 --camera 0 --seconds 115
"""
import argparse
import os
import time

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.robots.utils import make_robot_from_config

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = "pick up the red block and place it on the green pad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="pretrained_model dirs (or their parent run dirs)")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--port", default="/dev/tty.usbmodem5C821065561")
    ap.add_argument("--robot_id", default="station_follower")
    ap.add_argument("--camera", default=0, help="front camera index")
    ap.add_argument("--side_camera", default=1, help="side camera index (used only if a checkpoint expects it)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=115.0, help="judges cap each trial at 120 s")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max_step", type=float, default=8.0, help="max change per step per joint (normalised units)")
    ap.add_argument("--n_action_steps", type=int, default=None,
                    help="re-plan every N steps instead of the trained 100 (e.g. 20) for more closed-loop control")
    ap.add_argument("--temporal_ensemble", type=float, default=None,
                    help="ACT temporal ensembling coeff (e.g. 0.01): predict every step and average overlapping chunks")
    a = ap.parse_args()

    w = np.array(a.weights if a.weights else [1.0] * len(a.ckpt), dtype=np.float64)
    assert len(w) == len(a.ckpt), "one weight per checkpoint"
    w = w / w.sum()
    device = torch.device(a.device)

    policies = []
    for path in a.ckpt:
        path = path.rstrip("/")
        if not os.path.isfile(os.path.join(path, "config.json")):
            path = f"{path}/checkpoints/last/pretrained_model"
        cfg = PreTrainedConfig.from_pretrained(path)
        if a.temporal_ensemble is not None:
            cfg.temporal_ensemble_coeff, cfg.n_action_steps = a.temporal_ensemble, 1
        elif a.n_action_steps is not None:
            cfg.n_action_steps = a.n_action_steps
        policy = ACTPolicy.from_pretrained(path, config=cfg).to(device).eval()
        print(f"  n_action_steps={policy.config.n_action_steps} temporal_ensemble={policy.config.temporal_ensemble_coeff}")
        pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                             preprocessor_overrides={"device_processor": {"device": a.device}})
        policies.append((policy, pre, post))
        print(f"loaded {path}")

    idx = lambda c: int(c) if str(c).isdigit() else c
    wanted = {k.removeprefix("observation.images.") for p, _, _ in policies for k in p.config.input_features
              if k.startswith("observation.images.")}
    index = {"front": idx(a.camera), "side": idx(a.side_camera)}
    cams = {name: OpenCVCameraConfig(index_or_path=index[name], width=640, height=480, fps=30) for name in sorted(wanted)}
    print("cameras:", {n: index[n] for n in cams})
    robot = make_robot_from_config(SOFollowerRobotConfig(
        port=a.port, id=a.robot_id, use_degrees=False, max_relative_target=a.max_step, cameras=cams))
    robot.connect()
    try:
        for policy, _, _ in policies:
            policy.reset()
        t_end = time.perf_counter() + a.seconds
        step = 0
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            frame = {"observation.state": np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)}
            frame |= {f"observation.images.{n}": obs[n] for n in cams}
            acts = []
            for p, pre, post in policies:  # each policy only sees the cameras it was trained on
                f = {k: v for k, v in frame.items() if k == "observation.state" or k in p.config.input_features}
                acts.append(predict_action(f, p, device, pre, post, use_amp=False, task=TASK, robot_type="so_follower")
                            .cpu().numpy().reshape(-1))
            act = np.tensordot(w, np.stack(acts), axes=1)
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, act)})
            if step % (a.fps * 2) == 0:
                spread = np.abs(np.stack(acts) - act).max() if len(acts) > 1 else 0.0
                print(f"t={step / a.fps:5.1f}s action={np.round(act, 1)} ensemble_spread={spread:.1f}", flush=True)
            step += 1
            time.sleep(max(0.0, 1.0 / a.fps - (time.perf_counter() - t0)))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
