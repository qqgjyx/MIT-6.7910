import torch

from ..models import double_backward_ok, param_blocks
from .gradcov import gradcov
from .hessian import HVP, adam_preconditioner, anatomy, batch_sharpness, lanczos


def measure_step(model, optimizer, cfg, data, step):
    """Sharpness measurements at `step` (see DESIGN.md). Leaves the model and its grads untouched."""
    m = cfg.measure
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names, params = [n for n, _ in named], [p for _, p in named]
    dim, device = sum(p.numel() for p in params), params[0].device
    blocks = param_blocks(model)
    out = {}
    with double_backward_ok(), torch.autocast(device.type, enabled=False):
        if m.hessian:
            x, y = data.fixed_batch('train', m.hessian_batch, seed=1)
            hvp = HVP(model, x, y, params)
            evals, vecs = lanczos(hvp, dim, device, m.lanczos_iters, m.n_eigs)
            out['lambda_max'], out['eigs'] = evals[0], evals
            if m.anatomy:
                out['anatomy'] = anatomy(vecs[:, 0], hvp, names, blocks, params)
            P = adam_preconditioner(optimizer, params) if cfg.optimizer == 'adamw' and m.preconditioned else None
            if P is not None:
                s = P.rsqrt()

                def op(u):
                    return s * hvp(s * u)

                evals, vecs = lanczos(op, dim, device, m.lanczos_iters, 1)
                out['precond_lambda_max'] = evals[0]
                if m.anatomy:
                    out['anatomy_precond'] = anatomy(vecs[:, 0], op, names, blocks, params)
            del hvp, vecs
        if m.batch_sharpness_batches:
            batches = data.random_batches(m.batch_sharpness_batches, cfg.train.batch_size, seed=10**6 + step)
            out['batch_sharpness'] = batch_sharpness(model, params, batches)
        if m.gradcov:
            batches = data.random_batches(m.gradcov_L, m.gradcov_batch or cfg.train.batch_size,
                                          seed=2 * 10**6 + step)
            out['gradcov'] = gradcov(model, params, batches)
    return out
