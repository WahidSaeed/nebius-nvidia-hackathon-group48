#!/usr/bin/env python3
"""Live camera preview with the sim front-camera view overlaid, to place the real camera so it matches.

  python starter/pipeline/camera_align.py --camera 1
Keys:  n = next camera index   o = overlay on/off   +/- = overlay strength   s = save snapshot   q = quit
"""
import argparse
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--ref", default=os.path.join(HERE, "..", "docs", "front_cam.png"))
    a = ap.parse_args()

    ref = cv2.resize(cv2.imread(a.ref), (640, 480))
    idx, alpha, overlay = a.camera, 0.4, True
    cap = cv2.VideoCapture(idx)
    while True:
        ok, frame = cap.read()
        if not ok:
            frame = ref * 0
            cv2.putText(frame, f"camera {idx}: no frame", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame = cv2.resize(frame, (640, 480))
        view = cv2.addWeighted(frame, 1 - alpha, ref, alpha, 0) if overlay else frame.copy()
        # grid thirds + centre cross to help line things up
        for x in (213, 427):
            cv2.line(view, (x, 0), (x, 480), (0, 255, 255), 1)
        for y in (160, 320):
            cv2.line(view, (0, y), (640, y), (0, 255, 255), 1)
        cv2.drawMarker(view, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
        cv2.putText(view, f"cam {idx}  overlay {'ON' if overlay else 'off'} {alpha:.1f}  [n]ext [o]verlay [+/-] [s]ave [q]uit",
                    (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(view, f"cam {idx}  overlay {'ON' if overlay else 'off'} {alpha:.1f}  [n]ext [o]verlay [+/-] [s]ave [q]uit",
                    (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        side = cv2.hconcat([view, cv2.resize(ref, (640, 480))])
        cv2.imshow("camera align  (left: live + sim ghost | right: sim target)", side)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("n"):
            cap.release()
            idx = (idx + 1) % 4
            cap = cv2.VideoCapture(idx)
        if k == ord("o"):
            overlay = not overlay
        if k in (ord("+"), ord("=")):
            alpha = min(0.9, alpha + 0.1)
        if k == ord("-"):
            alpha = max(0.1, alpha - 0.1)
        if k == ord("s"):
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(f"outputs/camera_{idx}_snapshot.png", frame)
            print(f"saved outputs/camera_{idx}_snapshot.png")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
