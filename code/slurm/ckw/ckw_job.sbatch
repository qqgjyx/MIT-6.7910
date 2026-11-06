#!/bin/bash
#SBATCH --job-name=ckw-ana
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
# Report fixes (checkpoint-wording): TinyStories-10M figures with SGD at 0.06, learning-rate checks re-centred on
# 0.06, balanced tables at matched steps, and numbers. Code snapshot in ~/optsplit-ckw/code.
source $HOME/optsplit-ckw/code/slurm/env.sh
export OPTSPLIT_ROOT=/orcd/pool/014/conchell/optsplit
cd $OPTSPLIT_CODE
R=$OPTSPLIT_ROOT
OUT=$R/analysis/ts10m_ckw
F=$OUT/figures
mkdir -p $F
set -x
python -m optsplit.analysis.table 'ts10m_main_sgd0.06/*' 'ts10m_parent_sgd0.06/*' 'ts10m_ckpt_sgd0.06/*' 'ts10m_lr/*' \
    'ts10m_sgdtune/*' 'ts10m_main/sgd_d*' 'ts10m_lrsweep/sgd*'
python -m optsplit.analysis.plots losses --runs 'ts10m_main/adamw_d*' 'ts10m_main/muon_d*' 'ts10m_main_sgd0.06/sgd_d*' \
    --out $F --name losses
for a in adamw muon; do
    python -m optsplit.analysis.plots divergence --runs-a "ts10m_main/${a}_d*" --runs-b 'ts10m_main_sgd0.06/sgd_d*' \
        --split $R/analysis/ts10m_sgdtuned/split_${a}_sgd.json --out $F --name divergence_${a}_sgd
done
python -m optsplit.analysis.ngram 'ts10m_main/adamw_d0' 'ts10m_main/muon_d0' 'ts10m_main_sgd0.06/sgd_d0' --out $OUT
python -m optsplit.analysis.lr_robust --sweep 'ts10m_lrsweep/*' \
    --main 'ts10m_main/adamw_d*' 'ts10m_main/muon_d*' 'ts10m_main_sgd0.06/sgd_d*' --below 3.8 \
    --title TinyStories-10M --out $OUT
python $HOME/optsplit-ckw/ckw_numbers.py stoch lam30 bal around
python -m optsplit.layerlr.balanced plot \
    "v1 SGD, TinyStories-10M=$R/analysis/balanced/ts10m_sgd.json" \
    "v1 AdamW, TinyStories-10M=$R/analysis/balanced/ts10m_adamw.json" \
    "v1 Muon, TinyStories-10M=$R/analysis/balanced/ts10m_muon.json" \
    "v1 SGD, TinyStories-30M=$R/analysis/balanced/ts30m_sgd.json" \
    "v1 AdamW, TinyStories-30M=$R/analysis/balanced/ts30m_adamw.json" \
    "v1 Muon, TinyStories-30M (A cancelled at 6500)=$R/analysis/balanced/ts30m_muon.json" \
    "v1 SGD, Shakespeare=$R/analysis/balanced/shakespeare_sgd.json" \
    "v1 AdamW, Shakespeare=$R/analysis/balanced/shakespeare_adamw.json" \
    "v1 Muon, Shakespeare=$R/analysis/balanced/shakespeare_muon.json" \
    "v2 AdamW, TinyStories-10M (R: whole Hessian; Ro: own block)=$R/analysis/balanced/ts10m_v2_adamw.json" \
    "v2 Muon, TinyStories-10M (R: whole Hessian; Ro: own block)=$R/analysis/balanced/ts10m_v2_muon.json" \
    --out $F/balanced.pdf
python $HOME/optsplit-ckw/ckw_lrtraj.py $OUT
echo CKW DONE
