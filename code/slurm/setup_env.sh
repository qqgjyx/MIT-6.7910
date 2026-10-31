#!/bin/bash
# One-time setup on the ORCD login node: `bash slurm/setup_env.sh`.
# Creates the conda env at $OPTSPLIT_ROOT/env (default: optsplit/env in your ORCD pool storage).
set -eo pipefail
# default: optsplit/ in your ORCD pool storage (/orcd/pool/<NNN>/$USER; the ~/orcd/pool link can be broken)
export OPTSPLIT_ROOT=${OPTSPLIT_ROOT:-$(ls -d /orcd/pool/*/"$USER" 2>/dev/null | head -1)/optsplit}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

module load miniforge
eval "$(conda shell.bash hook)"
mkdir -p "$OPTSPLIT_ROOT"
conda create -y -p "$OPTSPLIT_ROOT/env" python=3.11
conda activate "$OPTSPLIT_ROOT/env"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r "$CODE/requirements.txt"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
