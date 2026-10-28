#!/bin/bash
#SBATCH --job-name=decfus_official_bridge_n80_r18
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPUArch_Ampere
#SBATCH --mem=16G
#SBATCH --time=240:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source /etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/decfus"

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

pip install --quiet swanlab

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# New experiment: ResNet-18 (added -- this repo only had VGG16/ResNet50;
# models/Nets.py now has a BasicBlock + ResNet18(), reusing ResNet4's
# stem/bn_final/dropout/fc exactly as-is, same house style as their own
# ResNet50 -- only the block type (plain 3x3x2, expansion=1) and depth
# ([2,2,2,2]) differ), CIFAR-10, alpha=0.1 (strong heterogeneity), bridge
# topology scaled to N=80 (39 leaves/hub, hub degree 40).
python main_fed.py \
  --algorithm DecFus \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 1000 \
  --local_bs 64 \
  --iid 0 \
  --data_beta 0.1 \
  --seed 0 \
  --gpu 0 \
  --topology bridge \
  --num_users 80
