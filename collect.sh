#!/bin/bash
# Collect scripted demonstrations in Isaac Sim -> LeRobot v2.1 dataset at $DATASET_ROOT.
#   ./collect.sh                       # $NUM_EPISODES successful episodes, headless
#   ./collect.sh --viz kit             # watch it in the Isaac Sim window
#   ./collect.sh --resume              # append to an existing dataset
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
exec "$ISAACLAB_PY" scripts/collect_demos.py \
    --num_episodes "$NUM_EPISODES" \
    --dataset_root "$DATASET_ROOT" \
    --task_text "$TASK" \
    "$@" 2>&1 | tee -a "$PROJECT_DIR/logs/collect.log"
