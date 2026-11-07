#!/bin/bash
# One-time setup on the login node: `bash slurm/setup_env.sh`.
# ORCD (default): conda env at $OPTSPLIT_ROOT/env (default: optsplit/env in your ORCD pool storage).
# FAS RC (OPTSPLIT_CLUSTER=fasrc, OPTSPLIT_ROOT set): uv venv at $OPTSPLIT_ROOT/env.
set -eo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ "${OPTSPLIT_CLUSTER:-orcd}" = fasrc ]; then
    : "${OPTSPLIT_ROOT:?set OPTSPLIT_ROOT to lab storage, e.g. on /n/netscratch}"
    mkdir -p "$OPTSPLIT_ROOT"
    # copy mode: the uv cache stays in $HOME, the env in lab storage (cache locks can hang on lab NFS)
    export UV_LINK_MODE=copy VIRTUAL_ENV="$OPTSPLIT_ROOT/env"
    uv venv -p 3.11 "$VIRTUAL_ENV"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install -r "$CODE/requirements.txt"
    PY="$VIRTUAL_ENV/bin/python"
else
    # default: optsplit/ in your ORCD pool storage (/orcd/pool/<NNN>/$USER; the ~/orcd/pool link can be broken)
    export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-$(ls -d /orcd/pool/*/"$USER" 2>/dev/null | head -1)/optsplit}
    module load miniforge
    eval "$(conda shell.bash hook)"
    mkdir -p "$OPTSPLIT_ROOT"
    conda create -y -p "$OPTSPLIT_ROOT/env" python=3.11
    conda activate "$OPTSPLIT_ROOT/env"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r "$CODE/requirements.txt"
    PY=python
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
