#!/bin/bash
#SBATCH --job-name=optsplit-data
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=optsplit-data-%j.out
# From the code dir: `sbatch slurm/prepare_data.sbatch shakespeare_char tinystories cifar10`.
set -e
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
for dataset in "$@"; do
    echo "preparing $dataset"
    python "data/prepare_$dataset.py"
done
