from contextlib import contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel

from .cnn import SimpleCNN
from .gpt import GPT, GPTConfig


def build_model(cfg, vocab_size=None, num_classes=None):
    m = cfg.model
    if cfg.task == 'gpt':
        return GPT(GPTConfig(vocab_size=vocab_size, block_size=m.block_size, n_layer=m.n_layer,
                             n_head=m.n_head, n_embd=m.n_embd, bias=m.bias,
                             tie_weights=m.tie_weights))
    if cfg.task == 'cnn':
        return SimpleCNN(width=m.cnn_width, hidden=m.cnn_hidden, num_classes=num_classes,
                         activation=m.activation)
    raise ValueError(cfg.task)


@contextmanager
def double_backward_ok():
    """Flash and memory-efficient attention kernels have no double backward; the math one does."""
    with sdpa_kernel(SDPBackend.MATH):
        yield


def param_blocks(model):
    """Partition of all parameter names into the blocks of the Hessian anatomy."""
    blocks = {}
    for name, _ in model.named_parameters():
        if isinstance(model, GPT):
            parts = name.split('.')
            if name.startswith(('transformer.wte', 'transformer.wpe')):
                b = 'emb'
            elif name.startswith('lm_head'):
                b = 'head'
            elif 'ln_' in name:
                b = 'norms'
            else:  # transformer.h.<i>.<attn|mlp>.<proj>.<weight|bias>
                i, module, proj = parts[2], parts[3], parts[4]
                if module == 'mlp':
                    b = f'L{i}.mlp'
                else:
                    b = f'L{i}.qk' if proj in ('q', 'k') else f'L{i}.ov'
        else:
            b = name.split('.')[0]
        blocks.setdefault(b, []).append(name)
    return blocks


def muon_params(model):
    """(hidden, other): parameter names updated by Muon and by its auxiliary AdamW."""
    hidden, other = [], []
    for name, p in model.named_parameters():
        if isinstance(model, GPT):
            is_hidden = name.startswith('transformer.h.') and p.ndim == 2
        else:
            is_hidden = p.ndim >= 2 and not name.startswith('fc2')
        (hidden if is_hidden else other).append(name)
    return hidden, other
