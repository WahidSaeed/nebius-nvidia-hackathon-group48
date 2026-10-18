#!/usr/bin/env python3
"""Live side-by-side view of several cameras.

  .venv/bin/python starter/pipeline/camera_view.py            # cameras 0 and 1 (station USB cameras)
  .venv/bin/python starter/pipeline/camera_view.py --cams 0 1
Keys:  s = save a snapshot of every camera to outputs/   q = quit
Display only: camera 0 is flipped vertically, camera 1 rotated 90 deg counter-clockwise (recordings are unaffected).
"""
import argparse
import os
import time

import cv2
import numpy as np

# Per-camera display transform.
TRANSFORM = {
    0: lambda f: cv2.flip(f, 0),                               # flip vertically
    1: lambda f: cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE),  # rotate 90 deg counter-clockwise
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()

    h = a.width * 9 // 16
    caps = {i: cv2.VideoCapture(i) for i in a.cams}
    while True:
        tiles = []
        for i, cap in caps.items():
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((h, a.width, 3), np.uint8)
                cv2.putText(frame, f"camera {i}: no frame", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif i in TRANSFORM:
                frame = TRANSFORM[i](frame)
            frame = cv2.resize(frame, (frame.shape[1] * h // frame.shape[0], h))  # same height, keep aspect
            cv2.putText(frame, f"cam {i}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(frame)
        cv2.imshow("cameras  [s] snapshot  [q] quit", cv2.hconcat(tiles))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            os.makedirs("outputs", exist_ok=True)
            stamp = time.strftime("%H%M%S")
            for i, t in zip(caps, tiles):
                cv2.imwrite(f"outputs/cam{i}_{stamp}.png", t)
            print(f"saved outputs/cam*_{stamp}.png")
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
