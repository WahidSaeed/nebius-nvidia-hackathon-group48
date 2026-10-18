#!/bin/bash
# One-shot environment setup on a fresh Nebius GPU VM (Ubuntu 24.04 + CUDA image, x86_64, L40S or
# RTX PRO 6000). Installs Isaac Sim 6.1 + Isaac Lab 3.0 (pip, headless-capable) and Isaac-GR00T N1.7,
# then clones this repo. Run as the VM user:  bash setup_vm.sh
# Versions match the machine this project was developed on (Isaac Sim 6.1.0, Isaac Lab 3.0.0, Python 3.12).
set -euo pipefail
WORK="${WORK:-$HOME/work}"
REPO_URL="${REPO_URL:-https://github.com/ddiddi/so101-sim-pickplace-gr00t.git}"
mkdir -p "$WORK" && cd "$WORK"

sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg python3.12-venv python3.12-dev build-essential cmake jq \
  libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libvulkan1 vulkan-tools
git lfs install

# --- Isaac Sim 6.1 (pip) + Isaac Lab 3.0 -------------------------------------------------------
python3.12 -m venv env_isaaclab && source env_isaaclab/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout v3.0.0 && ./isaaclab.sh --install && cd ..
pip install pyzmq msgpack msgpack-numpy h5py imageio pyarrow
deactivate

# --- Isaac-GR00T N1.7 --------------------------------------------------------------------------
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T
uv sync --python 3.12
uv run huggingface-cli login      # Cosmos-Reason2-2B backbone is gated: accept its terms on the Hub first
uv run hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
cd ..

# --- this project --------------------------------------------------------------------------------
git clone "$REPO_URL" so101_sim_pickplace
cat <<EOT

Done. Edit so101_sim_pickplace/env.sh so that
  ISAACLAB_PY=$WORK/env_isaaclab/bin/python
  ISAACLAB_DIR=$WORK/IsaacLab
  GR00T_REPO=$WORK/Isaac-GR00T
and remove the DGX-Spark-only lines in gr00t_activate (activate_spark.sh / nvpl) -- see README "Nebius".
EOT
