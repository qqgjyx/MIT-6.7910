import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call

from optsplit.layerlr.layer_lanczos import block_lanczos, index, top_eigs
from optsplit.config import load_config
from optsplit.measure.hessian import flat, split
from optsplit.models import build_model, double_backward_ok


@pytest.mark.parametrize('coupling', [0.0, 1.0])
def test_block_lanczos_returns_each_blocks_top_eigenvalue(coupling):
    """On a quadratic whose Hessian is block-diagonal (and with coupling between the blocks, which must not count),
    each block's value is the top eigenvalue of its own block, and arange(dim) gives the whole operator's."""
    g = torch.Generator().manual_seed(0)
    sizes = [3, 5, 7]
    D = torch.block_diag(*[(lambda m: m + m.T)(torch.randn(n, n, generator=g)) for n in sizes])
    C = torch.randn(15, 15, generator=g)
    A = D + coupling * (C + C.T) * (D == 0)
    blocks = list(torch.arange(15).split(sizes)) + [torch.arange(15)]
    got = block_lanczos(lambda V: [A @ v for v in V], 15, blocks, 'cpu', iters=20)
    expect = [torch.linalg.eigvalsh(A[b][:, b])[-1].item() for b in blocks]
    assert got == pytest.approx(expect, rel=1e-5, abs=1e-5)
    if coupling == 0:
        assert got[-1] == pytest.approx(max(got[:-1]), rel=1e-5)


@pytest.mark.parametrize('precond', [False, True])
def test_top_eigs_matches_the_exact_hessian_blocks(precond):
    """On a small GPT, per layer and sublayer, with the Hessian averaged over two chunks of data (and with
    S H S, S diagonal, for AdamW's preconditioner)."""
    torch.manual_seed(0)
    cfg = load_config(None, ['task=gpt', 'model.n_layer=2', 'model.n_embd=8', 'model.n_head=2',
                             'model.block_size=6'])
    model = build_model(cfg, vocab_size=11).double()
    named = list(model.named_parameters())
    params = [p for _, p in named]
    tok = torch.randint(0, 11, (8, 7), generator=torch.Generator().manual_seed(1))
    x, y = tok[:, :-1], tok[:, 1:]

    def loss(theta):
        logits = functional_call(model, {n: t for (n, _), t in zip(named, split(theta, params))}, (x,))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    with double_backward_ok():
        H = torch.autograd.functional.hessian(loss, flat([p.detach() for p in params]))
    s = torch.rand(len(H), generator=torch.Generator().manual_seed(2), dtype=torch.float64) + 0.5 \
        if precond else None
    A = H if s is None else s[:, None] * H * s[None, :]
    groups = {'all': torch.arange(len(H)), **index(named, 'layer', 'cpu'), **index(named, 'sublayer', 'cpu')}
    assert {'emb', '1', '2', 'head', 'attn1', 'mlp2'} <= set(groups)
    torch.set_default_dtype(torch.float64)
    try:
        with double_backward_ok():
            got = top_eigs(model, params, [(x[:4], y[:4]), (x[4:], y[4:])], groups, iters=len(H), s=s)
    finally:
        torch.set_default_dtype(torch.float32)
    for gname, b in groups.items():
        assert got[gname] == pytest.approx(torch.linalg.eigvalsh(A[b][:, b])[-1].item(), rel=1e-4), gname
