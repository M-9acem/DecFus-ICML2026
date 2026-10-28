#!/bin/bash
#SBATCH --job-name=decfus_official_bridge_n40
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPUArch_Ampere
#SBATCH --mem=16G
#SBATCH --time=144:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source /etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/decfus"

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

pip install --quiet swanlab

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Same bridge-topology experiment as run_official_decfus_bridge.sh (job 1057295,
# 65.16% final at N=10), scaled up to N=40 clients -- the topology's own default
# size in our reproduction (decfus/topology/registry.py's bridge_topology(n=40)),
# since the paper itself never specifies a client count for this topology (we
# added it, not the authors). build_bridge_topology() in Training_DecFus.py was
# generalized from the hardcoded N=10 dict to support this (verified to
# reproduce the exact same N=10 graph before this change, and to match our own
# reproduction's bridge_topology(40) node-for-node: 2 hubs degree 20 -- 19
# same-cluster leaves each + 1 bridge edge -- 38 leaves degree 1).
python main_fed.py \
  --algorithm DecFus \
  --dataset cifar10 \
  --model resnet50 \
  --epochs 1000 \
  --local_bs 64 \
  --iid 0 \
  --data_beta 0.5 \
  --seed 0 \
  --gpu 0 \
  --topology bridge \
  --num_users 40
