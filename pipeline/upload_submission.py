#!/usr/bin/env python3
"""Assemble the submission folder on the VM and upload it to Nebius Object Storage (ENHANCED_THROUGHPUT).

  set -a; source ~/work/.env; set +a
  python pipeline/upload_submission.py --prefix group48
"""
import argparse
import glob
import os
import shutil

import boto3

W = os.path.expanduser("~/work")
P = os.path.join(W, "so101_sim_pickplace")


def stage(out: str):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    def cp(src, dst):
        for s in glob.glob(src):
            d = os.path.join(out, dst)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(d, os.path.basename(s.rstrip("/"))) if dst.endswith("/") else d,
                                dirs_exist_ok=True)
            else:
                os.makedirs(d if dst.endswith("/") else os.path.dirname(d), exist_ok=True)
                shutil.copy(s, d)

    for name in ("act_main_v4_ft", "act_main_v4", "act_main_v3", "act_main_v2", "act_main", "act_ablation", "act_main_real", "act_real_ft"):
        cp(f"{W}/ckpt/{name}/checkpoints/last/pretrained_model", f"checkpoints/{name}")
    cp(f"{W}/ds_main/manifest.json", "manifests/main_manifest.json")
    cp(f"{W}/ds_abl/manifest.json", "manifests/ablation_manifest.json")
    cp(f"{W}/ds_real/manifest.json", "manifests/main_real_manifest.json")
    cp(f"{W}/ds_v2/manifest.json", "manifests/main_v2_manifest.json")
    cp(f"{W}/ds_v3/manifest.json", "manifests/main_v3_manifest.json")
    cp(f"{W}/ds_v4/manifest.json", "manifests/main_v4_manifest.json")
    cp(f"{P}/curation/*_curation.json", "curation/")
    for d in ("cosmos_s0", "cosmos_s2"):
        cp(f"{W}/datasets/local/{d}/cosmos_transfer_manifest.json", f"cosmos_transfer/{d}_manifest.json")
    for tag, work in (("generic_look", "cosmos_work"), ("station_look", "cosmos_work_b")):
        outs = sorted(glob.glob(f"{W}/{work}/out/shard0/ep*_front.mp4"))[:3]
        for o in outs:
            name = os.path.basename(o)
            cp(o, f"cosmos_transfer/samples/{tag}/{name.replace('.mp4', '_cosmos.mp4')}")
            cp(o.replace(".mp4", "_control_edge.mp4"), f"cosmos_transfer/samples/{tag}/{name.replace('.mp4', '_edge.mp4')}")
            cp(f"{W}/{work}/inputs/{name}", f"cosmos_transfer/samples/{tag}/{name.replace('.mp4', '_sim_input.mp4')}")
    for f in ("cosmos_transfer_batch.py", "cosmos_reason_curate.py", "build_datasets.py", "ensemble_rollout.py",
              "camera_align.py", "camera_view.py", "reset_arm.py", "add_blank_camera.py", "remap_sim_joints.py", "upload_submission.py"):
        cp(f"{P}/pipeline/{f}", f"code/{f}")
    cp(f"{P}/so101_pickplace/pickplace_env_cfg.py", "code/pickplace_env_cfg.py")
    cp(f"{P}/nebius/setup_vm_lean.sh", "code/setup_vm_lean.sh")
    cp(f"{P}/pipeline/SUBMISSION_README.md", "README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="group48")
    ap.add_argument("--stage_dir", default=os.path.join(W, "submission"))
    a = ap.parse_args()
    stage(a.stage_dir)
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], region_name="uk-south2")
    bucket = os.environ["S3_BUCKET"]
    n = size = 0
    for root, _, files in os.walk(a.stage_dir):
        for f in files:
            path = os.path.join(root, f)
            key = f"{a.prefix}/" + os.path.relpath(path, a.stage_dir)
            s3.upload_file(path, bucket, key, ExtraArgs={"StorageClass": "ENHANCED_THROUGHPUT"})
            n += 1
            size += os.path.getsize(path)
    print(f"uploaded {n} files, {size / 1e6:.0f} MB to s3://{bucket}/{a.prefix}/")


if __name__ == "__main__":
    main()
