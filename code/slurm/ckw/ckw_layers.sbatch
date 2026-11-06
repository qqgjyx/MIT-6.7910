#!/bin/bash
#SBATCH --job-name=ckw-lay
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
# Layer figures with transformer blocks numbered from 1 (checkpoint-wording).
source $HOME/optsplit-ckw/code/slurm/env.sh
export OPTSPLIT_ROOT=/orcd/pool/014/conchell/optsplit
cd $OPTSPLIT_CODE
A=$OPTSPLIT_ROOT/analysis
F=$A/ts10m_ckw/figures
set -x
python -m optsplit.analysis.lmc_layers --plot \
    "Shakespeare=$A/shakespeare/pairs_cross/lmc_layers.json,$A/shakespeare/pairs_adamw/lmc_layers.json,$A/shakespeare/pairs_muon/lmc_layers.json,$A/shakespeare/pairs_sgd/lmc_layers.json" \
    "TinyStories-10M=$A/ts10m_sgdtuned/pairs_cross_nosgd.json,$A/ts10m_sgdtuned/pairs_cross_sgd/lmc_layers.json,$A/ts10m/pairs_adamw/lmc_layers.json,$A/ts10m/pairs_muon/lmc_layers.json,$A/ts10m_sgdtuned/pairs_sgd/lmc_layers.json" \
    "TinyStories-30M=$A/ts30m/pairs_cross/lmc_layers.json,$A/ts30m/pairs_adamw/lmc_layers.json,$A/ts30m/pairs_muon/lmc_layers.json,$A/ts30m/pairs_sgd/lmc_layers.json" \
    --out $F
python -m optsplit.analysis.anatomy_layers --plot Shakespeare=$A/shakespeare/anatomy_layers.json \
    TinyStories-10M=$A/ts10m_sgdtuned/anatomy/anatomy_layers.json TinyStories-30M=$A/ts30m/anatomy_layers.json --out $F
echo LAYERS DONE
