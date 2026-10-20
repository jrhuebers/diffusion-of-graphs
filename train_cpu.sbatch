#!/bin/bash
#SBATCH --job-name=graph-ddpm-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Guarantee CPU-only execution and install the CPU PyTorch wheel in job-local env.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

python graph_diffusion.py --train \
  --nodes 16 \
  --steps 100 \
  --dataset-size 1024 \
  --batch-size 64 \
  --diffusion-steps 50 \
  --checkpoint graph_ddpm_cpu.pt
