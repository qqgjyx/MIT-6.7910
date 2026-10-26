"""Accuracy and robustness: Gaussian input noise for images, token-substitution sensitivity
(Hahn et al., 2021) for the language model."""

import torch

from ..models import GPT


@torch.no_grad()
def _predict(model, x, chunk=64):
    return torch.cat([model(x[i:i + chunk]).argmax(-1) for i in range(0, len(x), chunk)])


@torch.no_grad()
def cnn_robustness(model, x, y, sigmas=(0.05, 0.1, 0.2), seed=0):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    out = {'acc': (_predict(model, x) == y).float().mean().item()}
    for s in sigmas:
        noisy = x + s * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        out[f'acc_noise_{s:g}'] = (_predict(model, noisy) == y).float().mean().item()
    return out


@torch.no_grad()
def hahn_sensitivity(model, x, vocab_size, seed=0):
    """For each sequence, replace each position in turn by a different random token and count how
    often the argmax prediction at the last position changes."""
    gen = torch.Generator(device=x.device).manual_seed(seed)
    counts = []
    for seq in x:
        T = len(seq)
        variants = seq.repeat(T, 1)
        shift = torch.randint(1, vocab_size, (T,), generator=gen, device=x.device)
        idx = torch.arange(T, device=x.device)
        variants[idx, idx] = (seq + shift) % vocab_size
        base = model(seq[None])[0, -1].argmax()
        counts.append((model(variants)[:, -1].argmax(-1) != base).sum().item())
    counts = torch.tensor(counts, dtype=torch.float64)
    return {'sensitivity': (counts / x.shape[1]).mean().item(), 'sensitivity_count': counts.mean().item()}


@torch.no_grad()
def gpt_robustness(model, x, y, n_sens=32, context=64, seed=0):
    out = {'acc': (_predict(model, x) == y).float().mean().item()}
    out.update(hahn_sensitivity(model, x[:n_sens, :context], model.config.vocab_size, seed))
    return out


def robustness(model, x, y, context=64):
    if isinstance(model, GPT):
        return gpt_robustness(model, x, y, context=context)
    return cnn_robustness(model, x, y)
