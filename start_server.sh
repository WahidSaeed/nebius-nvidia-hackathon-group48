#!/bin/bash
# Terminal 1: GR00T N1.7 policy server for the fine-tuned checkpoint.
#   CHECKPOINT=$FINETUNE_OUTPUT/checkpoint-4000 ./start_server.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -d "$FINETUNE_OUTPUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
fi
[ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT/config.json" ] || { echo "set CHECKPOINT=<finetuned checkpoint dir>" >&2; exit 1; }
gr00t_activate
echo "Starting GR00T server: model=$CHECKPOINT tag=$EMBODIMENT_TAG port=$POLICY_PORT"
exec python gr00t/eval/run_gr00t_server.py \
    --model-path "$CHECKPOINT" \
    --modality-config-path "$MODALITY_CONFIG" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --device cuda --host 0.0.0.0 --port "$POLICY_PORT" "$@"
