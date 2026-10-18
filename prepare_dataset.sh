#!/bin/bash
# Make the collected dataset GR00T-ready: modality.json + stats.json/relative_stats.json + loader check.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DS="${1:-$DATASET_ROOT}"
[ -f "$DS/meta/info.json" ] || { echo "no dataset at $DS - run collect.sh first" >&2; exit 1; }
cp "$PROJECT_DIR/gr00t/modality.json" "$DS/meta/modality.json"
echo "modality.json installed in $DS/meta"
gr00t_activate
echo "Generating GR00T stats ..."
python gr00t/data/stats.py --dataset-path "$DS" --embodiment-tag "$EMBODIMENT_TAG" --modality-config-path "$MODALITY_CONFIG"
echo "Sanity check with GR00T's loader ..."
python "$PROJECT_DIR/gr00t/check_dataset.py" "$DS" "$MODALITY_CONFIG"
