#!/bin/bash
# Terminal 2: closed-loop evaluation of the served checkpoint in the Isaac Lab scene.
#   ./eval.sh --num_episodes 20 [--viz kit] [--record_dir logs/eval]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_DIR"
exec "$ISAACLAB_PY" scripts/eval_in_sim.py --task_text "$TASK" --policy_host "$POLICY_HOST" --policy_port "$POLICY_PORT" "$@"
