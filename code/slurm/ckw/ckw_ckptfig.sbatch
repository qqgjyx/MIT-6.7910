#!/bin/bash
#SBATCH --job-name=ckw-fig
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
# Checkpoint-only Figures 1 and 2 with AdamW and Muon only (the report keeps its SGD versions).
source $HOME/optsplit-ckw/code/slurm/env.sh
export OPTSPLIT_ROOT=/orcd/pool/014/conchell/optsplit
cd $OPTSPLIT_CODE
R=$OPTSPLIT_ROOT
F=$R/analysis/ts10m_ckw/figures
set -x
python -m optsplit.analysis.plots pairs --runs 'ts10m_main/adamw_d*' 'ts10m_main/muon_d*' --max-loss 4 --log-offset 1 \
    --out $F --name divergence_pairs_adamw_muon.pdf
python -m optsplit.analysis.plots divergence --runs-a 'ts10m_main/adamw_d*' --runs-b 'ts10m_main/muon_d*' \
    --split $R/analysis/ts10m/split_adamw_muon.json --traj $R/analysis/ts10m/thin0.28/trajectories_densmap.json \
    --traj-steps 500 1000 2000 3000 5000 10000 --traj-label-steps 250 1000 2000 --traj-name-step 500 \
    --mark-step 250 --linear-steps --max-loss 4 --val-runs 'ts10m_main/adamw_d*' 'ts10m_main/muon_d*' \
    --out $F --name divergence_umap_adamw_muon.pdf
echo FIGS DONE
