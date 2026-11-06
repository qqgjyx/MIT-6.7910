# Environment for every job and interactive shell on ORCD: `source slurm/env.sh`.
module load miniforge
eval "$(conda shell.bash hook)"
# default: optsplit/ in your ORCD pool storage (/orcd/pool/<NNN>/$USER; the ~/orcd/pool link can be broken)
export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-$(ls -d /orcd/pool/*/"$USER" 2>/dev/null | head -1)/optsplit}
export OPTSPLIT_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
conda activate "$OPTSPLIT_ROOT/env"
export PYTHONPATH="$OPTSPLIT_CODE${PYTHONPATH:+:$PYTHONPATH}"
