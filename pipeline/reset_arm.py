#!/usr/bin/env python3
"""Move the follower slowly (3 s) to the folded rest pose the real demos start from, then release it.

  .venv/bin/python starter/pipeline/reset_arm.py
"""
import argparse
import time

import numpy as np

from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.robots.utils import make_robot_from_config

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REST = np.array([5.0, -98.0, 99.5, 67.4, 17.1, 3.5])  # median start pose of the third real recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/tty.usbmodem5C821065561")
    ap.add_argument("--robot_id", default="station_follower")
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()

    robot = make_robot_from_config(SOFollowerRobotConfig(port=a.port, id=a.robot_id, use_degrees=False, cameras={}))
    robot.connect()
    try:
        obs = robot.get_observation()
        cur = np.array([obs[f"{j}.pos"] for j in JOINTS])
        print("from:", np.round(cur, 1))
        for t in np.linspace(0, 1, int(a.seconds * 30)):
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, cur + t * (REST - cur))})
            time.sleep(1 / 30)
        time.sleep(0.8)
        obs = robot.get_observation()
        print("rest:", np.round([obs[f"{j}.pos"] for j in JOINTS], 1))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
