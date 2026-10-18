#!/bin/bash
# Optional: record demos with the physical SO-101 leader arm through Isaac Lab's record_demos.py
# (needs the `isaacteleop` package + the so101_leader device; see README). Output: HDF5, then
# convert with scripts/hdf5_to_lerobot.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT="${1:-$DATASETS_DIR/hdf5/so101_sim_pickplace_teleop.hdf5}"
mkdir -p "$(dirname "$OUT")"
cd "$ISAACLAB_DIR"
exec "$ISAACLAB_PY" scripts/tools/record_demos.py \
    --task Isaac-PickPlace-Cube-SO101-Joint-Teleop-v0 \
    --external_callback so101_pickplace.register_tasks \
    --dataset_file "$OUT" --step_hz 30 --viz kit "${@:2}"
