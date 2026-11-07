"""Muon, vendored from https://github.com/KellerJordan/Muon (muon.py at f98f1ca, MIT license).

Only the single-device optimizer with auxiliary AdamW is kept. One change: `muon_update` no longer
overwrites `grad` in place (upstream does `grad.lerp_`), so a shadow optimizer stepping after Muon on
the same gradients sees them unchanged. Added: a group may set `orthogonalize=False` (ablation): the Nesterov
momentum is then only rescaled to the Frobenius norm of an orthogonal update of the same shape. And
`sv_power=p` (ablation, replaces Newton-Schulz): the update is U S^p V^T from the SVD U S V^T of the Nesterov
momentum, rescaled to that same Frobenius norm; p=0 is exact orthogonalization, p=1 is `orthogonalize=False`.
"""

import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Approximate orthogonalization U S' V^T of G by a quintic Newton-Schulz iteration (see upstream)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, orthogonalize=True, sv_power=None):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    if sv_power is not None:
        U, S, Vh = torch.linalg.svd(update.float(), full_matrices=False)
        update = (U * S.pow(sv_power)) @ Vh
        update = update * (min(update.shape[-2:]) ** 0.5 / (update.norm() + 1e-7))
    elif orthogonalize:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    else:  # same Frobenius norm as an orthogonal update: sqrt(min(n_out, n_in))
        update = update * (min(update.shape[-2:]) ** 0.5 / (update.norm() + 1e-7))
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for the param groups with `use_muon=True`, AdamW for the others."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["orthogonalize"] = group.get("orthogonalize", True)
                group["sv_power"] = group.get("sv_power", None)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon",
                                                 "orthogonalize", "sv_power"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"],
                                         orthogonalize=group["orthogonalize"], sv_power=group["sv_power"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
