#!/bin/bash
# Analysis of one setting (shakespeare, ts10m or ts30m) on CPU; add `swap` once its forks are done.
#   sbatch -p mit_normal -c 8 --mem 64G -t 04:00:00 -o analyze-%j.out scripts/analyze.sh ts10m [swap]
# JSON and text go to $OPTSPLIT_ROOT/analysis/<setting>/, figures to .../<setting>/figures/.
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"
S=$1
M=${S}_main
OUT=$OPTSPLIT_ROOT/analysis/$S
F=$OUT/figures
mkdir -p "$F"
python -m optsplit.analysis.table "$M/*" | tee "$OUT/table.txt"
python -m optsplit.analysis.plots losses --runs "$M/*" --out "$F"
for pair in "adamw muon" "adamw sgd" "muon sgd"; do
    read -r a b <<< "$pair"
    python -m optsplit.analysis.split --runs-a "$M/${a}_d*" --runs-b "$M/${b}_d*" --out "$OUT/split_${a}_${b}.json"
    python -m optsplit.analysis.plots divergence --runs-a "$M/${a}_d*" --runs-b "$M/${b}_d*" \
        --split "$OUT/split_${a}_${b}.json" --out "$F" --name "divergence_${a}_${b}"
done 2>&1 | tee "$OUT/split.txt"
python -m optsplit.analysis.eos "$M/*" --split "$OUT"/split_*.json | tee "$OUT/eos.txt"
python -m optsplit.analysis.plots eos --runs "$M/*_d0" --split "$OUT/split_adamw_sgd.json" --out "$F"
python -m optsplit.analysis.plots sharpness --runs "$M/*_d0" --split "$OUT/split_adamw_muon.json" --out "$F"
python -m optsplit.analysis.plots anatomy --runs "$M/*_d0" --out "$F"
python -m optsplit.analysis.plots anatomy --runs "$M/*_d0" --group layer --out "$F" --name anatomy_layer
python -m optsplit.analysis.plots anatomy --runs "$M/adamw_d0" --precond --out "$F" --name anatomy_precond
python -m optsplit.analysis.ngram "$M/*_d0" --out "$OUT" | tee "$OUT/ngram.txt"
python -m optsplit.analysis.decompose "$M/*" --out "$OUT" | tee "$OUT/decompose.txt"
if [ "$2" = swap ]; then
    python -m optsplit.analysis.swap --forks "${S}_forks/*" --pure "$M/*" --out "$OUT/swap" > "$OUT/swap.txt"
    python -m optsplit.analysis.plots swap --swap-json "$OUT/swap.json" --split "$OUT/split_adamw_muon.json" --out "$F"
fi
