#!/usr/bin/env python3
"""Score episode videos with Cosmos Reason (+ optional second VLM judge) and decide which to keep.

Works on any OpenAI-compatible endpoint:
  * Cosmos-Reason1-7B self-hosted with vLLM    (--base_url http://localhost:8000/v1 --model nvidia/Cosmos-Reason1-7B)
  * Nebius Token Factory VLM as 2nd judge    (--judge2_base_url https://api.tokenfactory.nebius.com/v1
                                               --judge2_model nvidia/Nemotron-3.5-Super-VL)

Input: a LeRobot dataset root (v2.1 per-episode mp4s, or v3.0 concatenated mp4s + meta/episodes).
Output: <out>.json with per-episode scores and the curation decision (keep / drop + reason).

  python pipeline/cosmos_reason_curate.py --dataset_root datasets/local/so101_sim --camera front \
      --out curation/sim_curation.json --source sim
"""
import argparse
import base64
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import tempfile

from openai import OpenAI

PROMPT = """You are a strict robotics data curator. The video shows an SO-101 robot arm attempting to
pick up a small red 30 mm cube and place it on a flat green pad.
Judge ONLY what is visible. Answer with a single JSON object, no prose, with these keys:
  "grasp_success": true/false   - the gripper closed on the cube and lifted it off the table
  "placed_on_pad": true/false   - at the end the cube rests on the green pad and the gripper has released it
  "motion_clean": true/false    - smooth motion, no collisions, no dropping, no long hesitation or jitter
  "num_regrasps": integer       - how many extra grasp attempts after the first
  "quality": integer 0-10       - overall usefulness as a training demonstration
  "notes": short string
"""


def episode_clips(root: str, camera: str):
    """Yield (episode_index, mp4_path, start_s, end_s) for one camera."""
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    key = f"observation.images.{camera}"
    if str(info.get("codebase_version", "")).startswith("v2"):
        for p in sorted(glob.glob(os.path.join(root, "videos", "chunk-*", key, "episode_*.mp4"))):
            yield int(re.search(r"episode_(\d+)", p).group(1)), p, None, None
        return
    import pandas as pd  # v3.0: concatenated videos, per-episode offsets in meta/episodes

    eps = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet"))))
    for _, r in eps.iterrows():
        p = os.path.join(root, info["video_path"].format(
            video_key=key, chunk_index=int(r[f"videos/{key}/chunk_index"]), file_index=int(r[f"videos/{key}/file_index"])))
        yield int(r["episode_index"]), p, float(r[f"videos/{key}/from_timestamp"]), float(r[f"videos/{key}/to_timestamp"])


def make_clip(mp4, start, end, fps=4, width=640) -> bytes:
    """Re-encode the episode at 4 fps (Cosmos Reason recommendation) -> mp4 bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        cmd = ["ffmpeg", "-loglevel", "error", "-y"]
        if start is not None:
            cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
        cmd += ["-i", mp4, "-vf", f"fps={fps},scale={width}:-2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", f.name]
        subprocess.run(cmd, check=True)
        return open(f.name, "rb").read()


def frames_jpeg(clip_mp4: bytes, n=8) -> list[bytes]:
    """Fallback for endpoints without video input: n evenly spaced JPEG frames."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "c.mp4")
        open(src, "wb").write(clip_mp4)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src, "-vf", "fps=4", os.path.join(d, "f%04d.jpg")], check=True)
        fs = sorted(glob.glob(os.path.join(d, "f*.jpg")))
        idx = [round(i * (len(fs) - 1) / max(n - 1, 1)) for i in range(n)] if fs else []
        return [open(fs[i], "rb").read() for i in idx]


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.findall(r"\{.*\}", text, flags=re.S)
    return json.loads(m[-1]) if m else {"parse_error": text[:300]}


def ask(client, model, clip: bytes, video: bool) -> dict:
    if video:
        media = [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(clip).decode()}}]
    else:
        media = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
                 for f in frames_jpeg(clip)]
    r = client.chat.completions.create(
        model=model, max_tokens=4096, temperature=0.2,
        messages=[{"role": "user", "content": media + [{"type": "text", "text": PROMPT}]}])
    content = r.choices[0].message.content or ""
    if isinstance(content, list):  # vLLM-Omni returns a list of content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return parse_json(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--source", default="sim", help="label written into the JSON: sim | cosmos | real")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_url", default=os.environ.get("COSMOS_REASON_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("COSMOS_REASON_MODEL", "nvidia/Cosmos-Reason1-7B"))
    ap.add_argument("--api_key", default=os.environ.get("COSMOS_REASON_KEY", "EMPTY"))
    ap.add_argument("--judge2_base_url", default=None)
    ap.add_argument("--judge2_model", default=None)
    ap.add_argument("--min_quality", type=int, default=6)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    judges = [("cosmos_reason", OpenAI(base_url=args.base_url, api_key=args.api_key), args.model, True)]
    if args.judge2_model:
        judges.append(("judge2:" + args.judge2_model,
                       OpenAI(base_url=args.judge2_base_url, api_key=os.environ["NEBIUS_API_KEY"]), args.judge2_model, False))

    clips = list(episode_clips(args.dataset_root, args.camera))
    if args.max_episodes:
        clips = clips[: args.max_episodes]

    def score(item):
        ep, mp4, s, e = item
        clip = make_clip(mp4, s, e)
        out = {}
        for name, client, model, video in judges:
            try:
                out[name] = ask(client, model, clip, video)
            except Exception as ex:  # keep going; an unscored episode is kept by default
                out[name] = {"error": str(ex)[:300]}
        return ep, out

    results = {}
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for ep, out in pool.map(score, clips):
            results[ep] = out
            print(ep, json.dumps(out)[:200], flush=True)

    # Curation decision: drop an episode if the Cosmos reasoner says the task failed or quality is low;
    # a second judge can only veto when it also says the task failed (ensemble = Cosmos primary + VLM veto).
    episodes, drop = {}, []
    for ep in sorted(results):
        c = results[ep].get("cosmos_reason", {})
        reasons = []
        if "error" not in c and "parse_error" not in c:
            if c.get("placed_on_pad") is False:
                reasons.append("cosmos: not placed on pad")
            if int(c.get("quality", 10)) < args.min_quality:
                reasons.append(f"cosmos: quality {c.get('quality')} < {args.min_quality}")
        for name, v in results[ep].items():
            if name.startswith("judge2") and v.get("placed_on_pad") is False and v.get("grasp_success") is False:
                reasons.append(f"{name}: grasp and place both failed")
        keep = not reasons
        episodes[ep] = {"keep": keep, "drop_reasons": reasons, "scores": results[ep]}
        if not keep:
            drop.append(ep)

    report = {
        "source": args.source,
        "dataset_root": os.path.abspath(args.dataset_root),
        "camera": args.camera,
        "judges": [{"name": n, "model": m} for n, _, m, _ in judges],
        "decision_rule": f"drop if Cosmos Reason placed_on_pad==false or quality<{args.min_quality}; "
                         "2nd judge may veto when it sees both grasp and place fail",
        "num_scored": len(episodes),
        "num_kept": len(episodes) - len(drop),
        "dropped_episodes": drop,
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"kept {report['num_kept']}/{report['num_scored']}  dropped={drop}  -> {args.out}")


if __name__ == "__main__":
    main()
