#!/bin/bash
#SBATCH --job-name=cls_bench
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/userspace/vvn/HMTL_with_calibration/logs/slurm-%j.out
#SBATCH --error=/userspace/vvn/HMTL_with_calibration/logs/slurm-%j.err

set -euo pipefail

USER_DIR=/userspace/vvn
PROJECT_DIR=$USER_DIR/HMTL_with_calibration
cd "$PROJECT_DIR"

export XDG_CACHE_HOME=$USER_DIR/cache
export PIP_CACHE_DIR=$USER_DIR/cache/pip
export HF_HOME=$USER_DIR/cache/huggingface
export OPENML_CACHE_DIR=$USER_DIR/cache/openml
export TRANSFORMERS_CACHE=$USER_DIR/cache/transformers
export MPLCONFIGDIR=$PROJECT_DIR/.cache/matplotlib
export OPENML_DOWNLOAD_RETRIES=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$PROJECT_DIR/logs"

.venv/bin/python scripts/run_classification_benchmark.py \
  --sizes 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --n-seeds 4 \
  --baselines catboost \
  --output "$PROJECT_DIR/experiments/classification_benchmark" \
  --high-level-progress-only \
  --max-dataset-workers 1 \
  --reverse-dataset-order
