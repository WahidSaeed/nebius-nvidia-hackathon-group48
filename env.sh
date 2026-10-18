#!/bin/bash
# Shared settings for the Isaac Lab SO-101 block pick-and-place -> GR00T N1.7 post-training pipeline.
#   source ~/Desktop/nvidia/so101_sim_pickplace/env.sh

export PROJECT_DIR=/home/solotech007/Desktop/nvidia/so101_sim_pickplace
export GR00T_REPO=/home/solotech007/Desktop/nvidia/Isaac-GR00T
export ISAACLAB_DIR=/home/solotech007/Desktop/isaac/IsaacLab
export ISAACLAB_PY=/home/solotech007/Desktop/isaac/env_isaaclab3/bin/python

# Isaac Sim needs these (same as ~/Desktop/isaac/launch_isaacsim.sh)
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Task + dataset
export TASK="${TASK:-pick up the red block and place it on the green pad}"
export DATASETS_DIR=$HOME/Desktop/nvidia/datasets
export DATASET_NAME="${DATASET_NAME:-so101_sim_pickplace}"
export DATASET_ROOT=$DATASETS_DIR/local/$DATASET_NAME
export NUM_EPISODES="${NUM_EPISODES:-50}"

# GR00T
export BASE_MODEL=$GR00T_REPO/checkpoints/GR00T-N1.7-3B
export MODALITY_CONFIG=$PROJECT_DIR/gr00t/so101_sim_config.py
export EMBODIMENT_TAG=NEW_EMBODIMENT
export FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-$HOME/Desktop/nvidia/finetune/$DATASET_NAME}"
export CHECKPOINT="${CHECKPOINT:-}"           # set to $FINETUNE_OUTPUT/checkpoint-<N> for start_server.sh
export POLICY_HOST=localhost
export POLICY_PORT=5555

# N1.7 loads the gated nvidia/Cosmos-Reason2-2B backbone from the Hub (do not set HF_HUB_OFFLINE)
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false

gr00t_activate() {
    cd "$GR00T_REPO"
    source .venv/bin/activate
    source scripts/activate_spark.sh >/dev/null
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
}
