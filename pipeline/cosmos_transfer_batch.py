#!/usr/bin/env python3
"""Cosmos Transfer 2.5 (edge, distilled) on sim episodes -> a "cosmos" copy of the sim LeRobot v2.1 dataset.

Actions/state are untouched; only the camera videos are re-rendered. Edge control keeps geometry and
motion fixed, so the recorded actions stay valid.

The distilled edge model takes exactly 93 frames, so each ~300-frame episode is resampled to 93 frames,
transferred, and the result is mapped back onto the original frame count (nearest frame, <=1.6 frames
of temporal jitter at 30 fps).

  # 1. on the H100 VM: make 93-frame inputs + spec JSONs (one shard per GPU)
  python pipeline/cosmos_transfer_batch.py prep --dataset_root $SIM --work cosmos_work --num_episodes 120 \
      --num_shards 8 --style_image station.jpg
  # 2. run each shard (model loads once per shard)
  for g in 0..7: CUDA_VISIBLE_DEVICES=$g python $CT/examples/inference.py -i cosmos_work/specs/shard$g/*.json \
      -o cosmos_work/out/shard$g --model=edge/distilled &
  # 3. build the cosmos dataset from whatever finished
  python pipeline/cosmos_transfer_batch.py assemble --dataset_root $SIM --work cosmos_work --out_root $COSMOS
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess

N_FRAMES = 93

# Appearance variants toward the real station (white booth + white table, white arm, blue cube, dark cloth pad).
# Round-robin over episodes so the policy sees several real-looking appearances, not one.
PROMPTS = [
    "A real photograph-quality video of a white 3D-printed SO-101 robot arm with black servo motors on a plain matte "
    "white tabletop inside a booth with white walls, bright even overhead lighting with soft shadows. The arm picks up "
    "a small solid blue plastic cube and places it on a square dark charcoal-grey cloth towel lying flat on the table. "
    "Realistic webcam noise, natural colors.",
    "Realistic webcam footage of a white desktop robot arm on a clean white table enclosed by white panel walls, a "
    "dark carpeted room visible behind the table. The gripper grasps a bright blue cube and sets it down on a folded "
    "black fabric square. Slight motion blur, faint shadows on the white surface.",
    "A real-world robotics booth: white tabletop with faint grey scuff marks, white side walls, white 3D-printed robot "
    "arm, cool overhead office light. The arm lifts a royal-blue cube and places it on a dark grey microfiber cloth "
    "pad. Photorealistic, slightly overexposed white surfaces.",
    "Documentary-style real video of a hobby robot arm in a white hackathon booth, a person's legs and cables visible "
    "on the dark floor behind the table. The robot picks a blue 3D-printed cube and places it gently on a black "
    "square cloth target. Realistic textures and lighting.",
]


def ffprobe_frames(mp4: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", mp4], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def episode_videos(root: str, camera: str) -> dict[int, str]:
    paths = glob.glob(os.path.join(root, "videos", "chunk-*", f"observation.images.{camera}", "episode_*.mp4"))
    return {int(re.search(r"episode_(\d+)", p).group(1)): p for p in paths}


def prep(a):
    vids = episode_videos(a.dataset_root, a.camera)
    eps = sorted(vids)
    random.Random(0).shuffle(eps)
    eps = sorted(eps[: a.num_episodes])
    os.makedirs(os.path.join(a.work, "inputs"), exist_ok=True)
    for i, ep in enumerate(eps):
        src = vids[ep]
        n = ffprobe_frames(src)
        dst = os.path.abspath(os.path.join(a.work, "inputs", f"ep{ep:06d}_{a.camera}.mp4"))
        # pick 93 evenly spaced frames, written at 16 fps (Transfer's native rate)
        idx = "+".join(f"eq(n\\,{round(k * (n - 1) / (N_FRAMES - 1))})" for k in range(N_FRAMES))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", f"select='{idx}',setpts=N/16/TB",
                        "-r", "16", "-frames:v", str(N_FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
        spec = {
            "name": f"ep{ep:06d}_{a.camera}",
            "prompt": PROMPTS[i % len(PROMPTS)],
            "video_path": dst,
            "guidance": 3,
            "num_steps": 4,
            "seed": 1000 + ep,
            "resolution": a.resolution,
            "edge": {"control_weight": a.edge_weight},
        }
        if a.style_image:
            spec["image_context_path"] = os.path.abspath(a.style_image)
        d = os.path.join(a.work, "specs", f"shard{i % a.num_shards}")
        os.makedirs(d, exist_ok=True)
        json.dump(spec, open(os.path.join(d, spec["name"] + ".json"), "w"), indent=2)
        json.dump({"episode": ep, "orig_frames": n, "prompt_id": i % len(PROMPTS)},
                  open(os.path.join(a.work, "inputs", spec["name"] + ".meta.json"), "w"))
    print(f"prepared {len(eps)} episodes in {a.num_shards} shards under {a.work}/specs")


def assemble(a):
    outs = {}
    for p in glob.glob(os.path.join(a.work, "out", "**", "*.mp4"), recursive=True):
        m = re.match(r"ep(\d+)_(\w+?)\.mp4$", os.path.basename(p))
        if m and m.group(2) == a.camera:
            outs[int(m.group(1))] = p
    if not outs:
        raise SystemExit("no transferred videos found")
    if os.path.exists(a.out_root):
        shutil.rmtree(a.out_root)
    shutil.copytree(a.dataset_root, a.out_root)
    dst_vids = episode_videos(a.out_root, a.camera)
    for ep, p in sorted(outs.items()):
        meta = json.load(open(os.path.join(a.work, "inputs", f"ep{ep:06d}_{a.camera}.meta.json")))
        n = meta["orig_frames"]
        # map frame k of the original episode to transferred frame round(k*(93-1)/(n-1)): stretch the 93-frame
        # clip to n frames at 30 fps, then scale back to the dataset resolution
        dur = n / 30.0
        tmp = dst_vids[ep] + ".tmp.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-vf",
                        f"setpts=N*{dur / (N_FRAMES - 1):.6f}/TB*({n - 1}/{n}),fps=30,scale={a.width}:{a.height}",
                        "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", tmp], check=True)
        got = ffprobe_frames(tmp)
        if got < n:  # pad by repeating the last frame
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-vf", f"tpad=stop_mode=clone:stop={n - got}",
                            "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", tmp + "2.mp4"], check=True)
            os.replace(tmp + "2.mp4", tmp)
        os.replace(tmp, dst_vids[ep])
    keep = sorted(outs)
    drop = sorted(set(dst_vids) - set(keep))
    used = {}  # prompt actually used per episode (read back from the specs, PROMPTS may have changed since prep)
    for f in glob.glob(os.path.join(a.work, "specs", "*", "*.json")):
        s = json.load(open(f))
        m = re.search(r"ep(\d+)_", s["name"])
        if m and int(m.group(1)) in outs:
            used[int(m.group(1))] = s["prompt"]
    json.dump({"transferred_episodes": keep, "not_transferred": drop, "camera": a.camera,
               "prompts_by_episode": {str(k): used[k] for k in sorted(used)}},
              open(os.path.join(a.out_root, "cosmos_transfer_manifest.json"), "w"), indent=2)
    print(f"cosmos dataset at {a.out_root}: {len(keep)} transferred, {len(drop)} to delete after v3 conversion")
    print("delete list:", drop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "assemble"])
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--num_episodes", type=int, default=120)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--resolution", default="480")
    ap.add_argument("--edge_weight", type=float, default=1.0)
    ap.add_argument("--style_image", default=None, help="photo of the real station (Transfer image_context_path)")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    prep(a) if a.mode == "prep" else assemble(a)
