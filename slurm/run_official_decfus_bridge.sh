#!/bin/bash
#SBATCH --job-name=decfus_official_bridge
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

pip install --quiet swanlab

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Same run as run_official_decfus.sh (job 1056953) -- ResNet-50, CIFAR-10, DecFus,
# Dirichlet alpha=0.5, seed=0, 1000 rounds -- but with --topology bridge instead of
# the repo's original 'fixed' graph: the two-cluster/double-star topology we built
# in our own reproduction (MG-DFL-style), added to Training_DecFus.py specifically
# to test whether its severe accuracy collapse (23.93% under our 'module' layer
# granularity, vs. 86.97% on the original topology at the same alpha/seed) shows up
# against the authors' own implementation too, or was an artifact of our own code.
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
  --topology bridge
