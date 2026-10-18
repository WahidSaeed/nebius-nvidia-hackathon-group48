#!/usr/bin/env python3
"""Build the training datasets (LeRobot v3.0, front camera only) + the per-source manifest.

Each source (v2.1 dataset from collect.sh / cosmos_transfer_batch.py assemble / lerobot-record) is
copied, converted to v3.0, filtered with its Cosmos Reason curation JSON (dropped episodes removed),
stripped of the wrist camera (the judging station has one fixed camera), then merged:

  main     = sim + cosmos (+ real)   -> ACT main checkpoint
  ablation = sim          (+ real)   -> ACT ablation (no Cosmos)

  python pipeline/build_datasets.py --work ~/work/ds \
      --sim ~/work/datasets/local/sim_s0:curation/sim_s0_curation.json ... \
      --cosmos ~/work/datasets/local/cosmos_s0 --real ~/work/datasets/real:curation/real_curation.json \
      --max_total 300
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

BIN = os.path.dirname(sys.executable)
CONVERT = [sys.executable, "-m", "lerobot.scripts.convert_dataset_v21_to_v30"]
EDIT = [os.path.join(BIN, "lerobot-edit-dataset")]
DROP_FEATURES = ["observation.images.wrist"]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def info(root):
    return json.load(open(os.path.join(root, "meta", "info.json")))


def prepare(spec: str, work: str, name: str) -> tuple[str, int, list[int]]:
    """spec = <dataset_root>[:<curation.json>] -> (clean v3.0 root, episodes kept, dropped indices)."""
    src, _, cur = spec.partition(":")
    dst = os.path.join(work, name)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(dst + "_old", ignore_errors=True)
    shutil.copytree(src, dst)
    if info(dst)["codebase_version"].startswith("v2"):
        run(CONVERT + [f"--repo-id=local/{name}", f"--root={dst}", "--push-to-hub=false"])
        shutil.rmtree(dst + "_old", ignore_errors=True)
    dropped = set()
    if cur:
        dropped |= {int(e) for e in json.load(open(cur)).get("dropped_episodes", [])}
    transfer_manifest = os.path.join(src, "cosmos_transfer_manifest.json")
    if os.path.exists(transfer_manifest):  # cosmos copy: episodes Transfer did not re-render are still raw sim
        dropped |= set(json.load(open(transfer_manifest))["not_transferred"])
    dropped = sorted(e for e in dropped if e < info(dst)["total_episodes"])
    if dropped:
        run(EDIT + [f"--repo_id=local/{name}", f"--root={dst}", f"--new_repo_id=local/{name}_f",
                    f"--new_root={dst}_f", "--operation.type=delete_episodes",
                    f"--operation.episode_indices={json.dumps(dropped)}"])
        shutil.rmtree(dst)
        shutil.move(dst + "_f", dst)
    feats = [f for f in DROP_FEATURES if f in info(dst)["features"]]
    if feats:
        run(EDIT + [f"--repo_id=local/{name}", f"--root={dst}", f"--new_repo_id=local/{name}_c",
                    f"--new_root={dst}_c", "--operation.type=remove_feature",
                    f"--operation.feature_names={json.dumps(feats)}"])
        shutil.rmtree(dst)
        shutil.move(dst + "_c", dst)
    return dst, info(dst)["total_episodes"], dropped


def merge(parts: list[str], out: str):
    shutil.rmtree(out, ignore_errors=True)
    names = [f"local/{os.path.basename(p)}" for p in parts]
    run(EDIT + [f"--new_repo_id=local/{os.path.basename(out)}", f"--new_root={out}", "--operation.type=merge",
                f"--operation.repo_ids={json.dumps(names)}", f"--operation.roots={json.dumps(parts)}",
                "--operation.concatenate_videos=false"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--sim", nargs="+", required=True)
    ap.add_argument("--cosmos", nargs="*", default=[])
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--max_total", type=int, default=300)
    ap.add_argument("--max_real", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    built = {"sim": [], "cosmos": [], "real": []}
    for src, specs in (("sim", a.sim), ("cosmos", a.cosmos), ("real", a.real)):
        for i, spec in enumerate(specs):
            root, n, dropped = prepare(spec, a.work, f"{src}_{i}")
            built[src].append({"root": root, "source_spec": spec, "episodes": n, "dropped_by_cosmos_reason": dropped})

    count = {k: sum(p["episodes"] for p in v) for k, v in built.items()}
    assert count["real"] <= a.max_real, f"real episodes {count['real']} > {a.max_real}"
    total_main = count["sim"] + count["cosmos"] + count["real"]
    assert total_main <= a.max_total, f"main dataset has {total_main} episodes > {a.max_total}; pass fewer sim shards"

    roots = {k: [p["root"] for p in v] for k, v in built.items()}
    merge(roots["sim"] + roots["cosmos"] + roots["real"], os.path.join(a.work, "main"))
    merge(roots["sim"] + roots["real"], os.path.join(a.work, "ablation"))

    manifest = {
        "camera": "observation.images.front only (wrist removed: judging station has one fixed camera)",
        "sources": built,
        "main": {"path": os.path.join(a.work, "main"), "episodes_per_source": count, "total": total_main},
        "ablation": {"path": os.path.join(a.work, "ablation"),
                     "episodes_per_source": {"sim": count["sim"], "cosmos": 0, "real": count["real"]},
                     "total": count["sim"] + count["real"]},
    }
    json.dump(manifest, open(os.path.join(a.work, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: manifest[k]["episodes_per_source"] for k in ("main", "ablation")}, indent=2))


if __name__ == "__main__":
    main()
