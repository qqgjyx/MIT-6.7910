"""Distance of each run's predictions from the unigram and bigram statistics of the training data, on the
validation probe: whether training passes through the n-gram solutions, and whether optimizers meet there.
python -m optsplit.analysis.ngram 'shakespeare_main/*_d0' [--out analysis/shakespeare]"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.distance import js_divergence, loss_points
from optsplit.analysis.io import load_runs
from optsplit.config import root_dir
from optsplit.data import build_data
from optsplit.probe import Probe


def ngram_logp(cfg, alpha=1.0, chunk=50_000_000):
    """Log-probabilities of the unigram and bigram models at the probe positions. The bigram is smoothed
    towards the unigram: p(b | a) = (n(a, b) + alpha p(b)) / (n(a) + alpha)."""
    data = build_data(cfg, 'cpu')
    probe = Probe(cfg, data)
    V, train = data.vocab_size, data.splits['train']
    uni = torch.zeros(V, dtype=torch.float64)
    bi = torch.zeros(V * V, dtype=torch.float64)
    for i in range(0, len(train) - 1, chunk):
        t = torch.from_numpy(np.asarray(train[i:i + chunk + 1], dtype=np.int64))
        uni += torch.bincount(t[:-1], minlength=V)
        bi += torch.bincount(t[:-1] * V + t[1:], minlength=V * V)
    p_uni = (uni + 1) / (uni + 1).sum()
    bi = bi.view(V, V)
    p_bi = (bi + alpha * p_uni) / (bi.sum(1, keepdim=True) + alpha)
    cur = probe.val[0][:, probe.pos].reshape(-1)
    return p_uni.log().expand(len(cur), V), p_bi[cur].log()


def curves(runs):
    lu, lb = ngram_logp(runs[0].cfg)
    out = {}
    for run in runs:
        steps, loss = loss_points(run)
        out[run.id] = dict(optimizer=run.optimizer, steps=steps.tolist(), loss=loss.tolist(),
                           js_unigram=[js_divergence(run.logp(s), lu) for s in steps],
                           js_bigram=[js_divergence(run.logp(s), lb) for s in steps])
    return out


def plot(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    opts = sorted({r['optimizer'] for r in res.values()})
    for rid, r in res.items():
        for ax, key in zip(axes, ('js_unigram', 'js_bigram')):
            ax.plot(r['loss'], r[key], color=f"C{opts.index(r['optimizer'])}", label=rid)
    for ax, name in zip(axes, ('unigram', 'bigram')):
        ax.set(xlabel='training loss', title=f'distance from the {name} model')
        ax.invert_xaxis()
    axes[0].set(ylabel='JS divergence (bits)')
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', help='output directory')
    args = p.parse_args()
    runs = load_runs(args.runs)
    out = Path(args.out or root_dir() / 'analysis' / runs[0].id.split('/')[0])
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    res = curves(runs)
    (out / 'ngram.json').write_text(json.dumps(res))
    for rid, r in res.items():
        i, j = int(np.argmin(r['js_unigram'])), int(np.argmin(r['js_bigram']))
        print(f"{rid}: closest to unigram {r['js_unigram'][i]:.3f} at loss {r['loss'][i]:.2f}, "
              f"to bigram {r['js_bigram'][j]:.3f} at loss {r['loss'][j]:.2f}")
    plot(res, out / 'figures' / 'ngram.png')


if __name__ == '__main__':
    main()
