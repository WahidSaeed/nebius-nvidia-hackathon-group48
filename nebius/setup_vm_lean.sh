#!/bin/bash
# Lean setup for the hackathon: Isaac Sim 6.1 + Isaac Lab 3.0 (sim collection) and LeRobot (ACT
# training). Skips GR00T (no time to fine-tune a 3B model). Run as the VM user:  bash setup_vm_lean.sh
set -euo pipefail
WORK="${WORK:-$HOME/work}"
REPO_URL="${REPO_URL:-https://github.com/ddiddi/so101-sim-pickplace-gr00t.git}"
mkdir -p "$WORK" && cd "$WORK"

sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg python3.12-venv python3.12-dev build-essential cmake jq \
  libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libvulkan1 vulkan-tools
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"

# --- LeRobot (ACT training) in its own venv, installed first so training can start early ---------
uv venv -p 3.12 env_lerobot
uv pip install --python env_lerobot/bin/python "lerobot[feetech,dataset,training]" boto3 openai

# --- Isaac Sim 6.1 (pip) + Isaac Lab 3.0 -------------------------------------------------------
uv venv -p 3.12 env_isaaclab --seed
source env_isaaclab/bin/activate
uv pip install "isaacsim[all,extscache]==6.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
export OMNI_KIT_ACCEPT_EULA=YES
[ -d IsaacLab ] || git clone https://github.com/isaac-sim/IsaacLab.git
(cd IsaacLab && git checkout v3.0.0-EA && ./isaaclab.sh --install)
uv pip install pyzmq msgpack msgpack-numpy h5py imageio pyarrow
deactivate

[ -d so101_sim_pickplace ] || git clone "$REPO_URL" so101_sim_pickplace
cd so101_sim_pickplace
sed -i "s|^export PROJECT_DIR=.*|export PROJECT_DIR=$WORK/so101_sim_pickplace|;
        s|^export ISAACLAB_DIR=.*|export ISAACLAB_DIR=$WORK/IsaacLab|;
        s|^export ISAACLAB_PY=.*|export ISAACLAB_PY=$WORK/env_isaaclab/bin/python|;
        s|^export DATASETS_DIR=.*|export DATASETS_DIR=$WORK/datasets|" env.sh
echo "SETUP DONE"
