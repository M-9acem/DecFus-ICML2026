#!/bin/bash
#SBATCH --job-name=decfus_official
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPUArch_Ampere
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source /etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/decfus"

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

# swanlab isn't installed in the shared decfus env yet, and installing it on the
# Hades login node gets killed (resource limits there) -- install it here instead,
# on the compute node, where it has room. Harmless no-op on a future resubmit
# once it's actually present in the env.
pip install --quiet swanlab

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Official DecFus-ICML2026 code, run unmodified except one line in main_fed.py
# (swanlab.init(..., mode="local") -- the compute nodes have no swanlab cloud
# login, so this just redirects its logging to a local ./swanlog folder instead
# of trying to sync online; no effect on the actual training/aggregation logic).
#
# Settings chosen to match our own Cell 1 baseline as closely as their CLI
# allows, for direct comparison: ResNet-50, CIFAR-10, Topology 1 (their fixed
# 10-client graph), DecFus, Dirichlet alpha=0.5, seed=0, 1000 rounds.
python main_fed.py \
  --algorithm DecFus \
  --dataset cifar10 \
  --model resnet50 \
  --epochs 1000 \
  --local_bs 64 \
  --iid 0 \
  --data_beta 0.5 \
  --seed 0 \
  --gpu 0
