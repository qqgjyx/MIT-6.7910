#!/bin/bash
#SBATCH --job-name=optsplit-test
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=optsplit-test-%j.out
# GPU sanity check. From the code dir: `sbatch slurm/test_gpu.sbatch`.
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
nvidia-smi
python - <<'EOF'
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(), 'bf16', torch.cuda.is_bf16_supported())

a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
a @ a
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
print(f'bf16 matmul: {20 * 2 * 8192**3 / (time.time() - t) / 1e12:.0f} TFLOP/s')

q, k, v = (torch.randn(2, 4, 64, 32, device='cuda', requires_grad=True) for _ in range(3))
with sdpa_kernel(SDPBackend.MATH):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    g, = torch.autograd.grad(y.square().sum(), q, create_graph=True)
    h, = torch.autograd.grad(g.sum(), q)
print('SDPA math double backward ok:', bool(torch.isfinite(h).all()))
EOF
