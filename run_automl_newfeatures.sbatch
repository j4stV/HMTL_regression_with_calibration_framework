#!/bin/bash
#SBATCH --job-name=automl_newft
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=140:00:00
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

.venv/bin/python scripts/run_automlbenchmark_experiment.py \
  --output "$PROJECT_DIR/experiments/automl_newfeatures_v2" \
  --n-seeds 2 \
  --baselines catboost single_mlp \
  --max-dataset-workers 6 \
  --high-level-progress-only
