#!/bin/bash
#SBATCH --job-name=OFFLOG_C_resnet8_first_1
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

# Re-run of an already-measured configuration, this time with --comm_log.
# Accuracy is therefore a regression check: the recorder makes no RNG calls and
# was verified not to perturb the aggregation, so the same seed must reproduce
# the previously recorded final accuracy. A mismatch means the instrumentation
# changed behaviour and the logs should not be trusted.
python main_fed.py \
  --algorithm DecFus \
  --dataset cifar10 \
  --model resnet8 \
  --epochs 1000 \
  --local_bs 64 \
  --iid 0 \
  --data_beta 0.5 \
  --seed 0 \
  --gpu 0 \
  --topology two_cluster \
  --num_users 20 \
  --exchange_prob 0.4 \
  --comm_log \
  --comm_log_dir ./output/comm \
  --run_id OFFLOG_C_resnet8_first_1 \
  --layer_policy first_1
