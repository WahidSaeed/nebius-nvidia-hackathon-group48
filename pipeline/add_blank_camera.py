#!/usr/bin/env python3
"""Add a blank (black) camera stream to a LeRobot v3.0 dataset, so sim/Cosmos episodes (front camera only) can be
merged with real two-camera demos. Frame counts and timestamps are copied from an existing camera, so every
episode's video index stays valid; ACT then learns "black side view => rely on the front view".

  python pipeline/add_blank_camera.py <dataset_root> [--src observation.images.front] [--dst observation.images.side]
"""
import argparse
import glob
import json
import os
import subprocess

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--src", default="observation.images.front")
    ap.add_argument("--dst", default="observation.images.side")
    a = ap.parse_args()

    for src in sorted(glob.glob(f"{a.root}/videos/{a.src}/*/*.mp4")):
        dst = src.replace(f"/videos/{a.src}/", f"/videos/{a.dst}/")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", "drawbox=color=black:t=fill",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-g", "2", "-r", "30", dst], check=True)

    info_p = f"{a.root}/meta/info.json"
    info = json.load(open(info_p))
    info["features"][a.dst] = json.loads(json.dumps(info["features"][a.src]))
    json.dump(info, open(info_p, "w"), indent=4)

    for f in sorted(glob.glob(f"{a.root}/meta/episodes/*/*.parquet")):
        df = pd.read_parquet(f)
        for col in [c for c in df.columns if c.startswith(f"videos/{a.src}/")]:
            df[col.replace(f"videos/{a.src}/", f"videos/{a.dst}/")] = df[col]
        stats_cols = [c for c in df.columns if c.startswith(f"stats/{a.src}/")]
        for col in stats_cols:  # black frames: zero stats (recompute_stats refreshes them anyway)
            new = col.replace(f"stats/{a.src}/", f"stats/{a.dst}/")
            df[new] = df[col].apply(lambda v: v * 0 if hasattr(v, "__mul__") else v)
        df.to_parquet(f)
    print("added", a.dst, "to", a.root)


if __name__ == "__main__":
    main()
