#!/bin/bash
#SBATCH --job-name=optsplit-cmd
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=optsplit-cmd-%j.out
# Run one command on a GPU node (offline function metrics, mode connectivity). From the code dir:
#   sbatch slurm/cmd.sbatch python -m optsplit.metrics.run 'ts10m_main/*' 'ts10m_forks/*' --steps all
# Partition and time can be overridden on the command line (sbatch -p mit_preemptable -t 12:00:00 ...).
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
"$@"
