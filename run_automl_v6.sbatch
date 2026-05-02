#!/bin/bash
#SBATCH --job-name=automl_v6_resoff
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=140:00:00
#SBATCH --output=/userspace/vvn/HMTL_v6_residual_off/logs/slurm-%j.out
#SBATCH --error=/userspace/vvn/HMTL_v6_residual_off/logs/slurm-%j.err

set -euo pipefail

USER_DIR=/userspace/vvn
PROJECT_DIR=$USER_DIR/HMTL_v6_residual_off
cd "$PROJECT_DIR"

# Share caches with the main repo to avoid re-download of OpenML datasets, pip etc.
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

# V6 differs from V4 only by configs/model_snn.yaml :: encoder.residual = false
# All other settings (contrastive aux, n_bins=5, 20 models, RAdam+Lookahead) identical.
.venv/bin/python scripts/run_automlbenchmark_experiment.py \
  --output "$PROJECT_DIR/experiments/automl_v6_residual_off" \
  --n-seeds 2 \
  --baselines catboost single_mlp \
  --max-dataset-workers 6 \
  --high-level-progress-only
