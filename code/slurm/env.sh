# Environment for every job and interactive shell: `source slurm/env.sh`.
# OPTSPLIT_CLUSTER selects the cluster: orcd (default) or fasrc (FAS RC Cannon).
export OPTSPLIT_CLUSTER=${OPTSPLIT_CLUSTER:-orcd}
export OPTSPLIT_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "$OPTSPLIT_CLUSTER" = fasrc ]; then
    # set OPTSPLIT_ROOT to lab storage (not $HOME, 100 GB); the env is a uv venv at $OPTSPLIT_ROOT/env
    : "${OPTSPLIT_ROOT:?set OPTSPLIT_ROOT to lab storage, e.g. on /n/netscratch}"
    source "$OPTSPLIT_ROOT/env/bin/activate"
    # gpu_test gives 20 GB MIG slices; AdamW's batch sharpness (double backward on 128 x 256 tokens, fp32)
    # needs about 17 GB and otherwise fails on fragmentation after the preconditioned Lanczos
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
else
    module load miniforge
    eval "$(conda shell.bash hook)"
    # default: optsplit/ in your ORCD pool storage (/orcd/pool/<NNN>/$USER; the ~/orcd/pool link can be broken)
    export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-$(ls -d /orcd/pool/*/"$USER" 2>/dev/null | head -1)/optsplit}
    conda activate "$OPTSPLIT_ROOT/env"
fi
export PYTHONPATH="$OPTSPLIT_CODE${PYTHONPATH:+:$PYTHONPATH}"
