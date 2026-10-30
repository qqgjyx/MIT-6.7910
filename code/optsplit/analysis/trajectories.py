"""Prediction trajectories of the pure runs in two dimensions, as in Fig. 1 of Jastrzebski et al. (2020) but from
the distance used everywhere else: UMAP of the Jensen-Shannon distance (the square root of the divergence, a
metric) between the predictions of every pair of (run, step) on `--rows` rows of the validation probe. Markers
show where each run first reaches the training losses `--losses`, so that equal positions along the paths can be
compared at equal loss.
python -m optsplit.analysis.trajectories 'ts10m_main/*' --losses 6 4 2.5 [--method umap densmap mds] --out analysis/ts10m
python -m optsplit.analysis.trajectories --plot TITLE=analysis/ts10m/trajectories.json ... --out DIR"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.transforms import Bbox, ScaledTranslation

from optsplit.analysis.distance import loss_points
from optsplit.analysis.io import load_runs

MARKERS = 'osD^v'
STEP_MARKERS = 'sD^vPX'
HALO = [patheffects.withStroke(linewidth=1.5, foreground='w')]


def entropy(p):
    """Mean over rows of the entropy in bits; p is (..., rows, V)."""
    return -(p * torch.log2(p.clamp_min(1e-30))).sum(-1).mean(-1)


@torch.no_grad()
def js_matrix(P, batch=32):
    """Pairwise mean-over-rows JS divergence in bits between the probabilities P (n, rows, V):
    JS(p, q) = H((p + q) / 2) - (H(p) + H(q)) / 2, the same quantity as `distance.js_divergence`."""
    h = entropy(P)
    D = torch.zeros(len(P), len(P), device=P.device)
    for i in range(len(P)):
        for j in range(i + 1, len(P), batch):
            D[i, j:j + batch] = entropy((P[i] + P[j:j + batch]) / 2) - (h[i] + h[j:j + batch]) / 2
    return (D + D.T).clamp_min(0)


def embed(D, method, seed=0):
    """2-D coordinates from the distance matrix D: UMAP keeps each point's nearest neighbours, densMAP (UMAP that
    also keeps how spread out each neighbourhood is) and metric MDS (which fits all distances at once) keep more of
    the distances."""
    if method == 'mds':
        from sklearn.manifold import MDS
        return MDS(dissimilarity='precomputed', random_state=seed).fit_transform(D)
    import umap
    return umap.UMAP(metric='precomputed', densmap=method == 'densmap', random_state=seed).fit_transform(D)


def parse(spec, losses=()):
    """'JSON[@L1,L2,...]' -> (trajectories, losses to mark)."""
    f, _, ls = spec.partition('@')
    return json.loads(Path(f).read_text()), [float(x) for x in ls.split(',')] if ls else list(losses)


def draw(ax, d, losses, title=None, steps=(), label_steps=()):
    """One panel; each run's end is labelled with its data order. Markers where each run first reaches `losses`
    and at its logged step nearest each of `steps`, with the run's training loss written next to those (and its
    first point) at `label_steps`. A star marks the shared initialization or, if the runs start later, a dot each
    run's first point. Each label sits 5 points from its point, in its preferred direction or the nearest turn of it
    clear of the markers and of the labels placed before."""
    opts = sorted({r['optimizer'] for r in d.values()})
    marks = {n: [0] + [int(np.abs(np.array(r['steps']) - s).argmin()) for s in steps] for n, r in d.items()}
    at = {n: np.array(d[n]['xy'])[ix] for n, ix in marks.items()}  # each run's first point and step markers
    todo = []  # labels (text, point, preferred direction, colour), placed once every path is drawn
    for name, r in d.items():
        c = f'C{opts.index(r["optimizer"])}'
        xy = np.array(r['xy'])
        ax.plot(*xy.T, '-', lw=0.6, alpha=0.6, color=c)
        todo.append((name.rsplit('_', 1)[-1], xy[-1], xy[-1] - xy[(len(xy) - 1) // 2], c))  # beyond the run's end
        if r['steps'][0]:
            ax.plot(*xy[0], 'o', ms=2.4, mfc='k', mec=c, mew=0.8)
        for m, L in zip(MARKERS, losses):
            i = next((i for i, l in enumerate(r['loss']) if l <= L), None)
            if i is not None:
                ax.plot(*xy[i], m, ms=4, mfc=c, mec='k', mew=0.4)
        idx = marks[name]
        for k, i in enumerate(idx):
            if k:
                ax.plot(*xy[i], STEP_MARKERS[k - 1], ms=4, mfc=c, mec='k', mew=0.4)
            if r['steps'][i] in label_steps:
                # beside the path from the marker before to the one after, on the side away from the nearest marker
                # of another run at the same step
                v = xy[idx[min(k + 1, len(idx) - 1)]] - xy[idx[max(k - 1, 0)]]
                v = np.array([-v[1], v[0]])
                near = min((at[o][k] for o in d if o != name), key=lambda p: np.hypot(*(p - xy[i])), default=xy[i])
                todo.append((f'{r["loss"][i]:#.3g}', xy[i], v if v @ (xy[i] - near) > 0 else -v, c))
    first = next(iter(d.values()))
    if not first['steps'][0]:
        ax.plot(*np.array(first['xy'][0]), 'k*', ms=9)
    ax.get_xlim()  # settles the limits, so that display positions are known
    ren = ax.figure.canvas.get_renderer()
    pt = ren.points_to_pixels(1)
    taken = [Bbox.from_bounds(*(ax.transData.transform(p) - 2.5 * pt), 5 * pt, 5 * pt)  # the markers
             for ps in at.values() for p in ps]
    side = lambda t, lo, hi: lo if t < -2.5 else hi if t > 2.5 else 'center'
    for s, xy, v, c in todo:
        v = ax.transData.transform(xy + v) - ax.transData.transform(xy)
        turns = np.arctan2(v[1], v[0]) + np.radians([0, 45, -45, 90, -90, 135, -135, 180])
        for a in [*turns, turns[0]]:  # the preferred direction again if no turn is clear
            dx, dy = 5 * np.cos(a), 5 * np.sin(a)
            t = ax.annotate(s, xy, xytext=(dx, dy), textcoords='offset points', fontsize=5, color=c,
                            path_effects=HALO, ha=side(dx, 'right', 'left'), va=side(dy, 'top', 'bottom'))
            b = t.get_window_extent(ren)
            if not any(b.overlaps(o) for o in taken):
                break
            t.remove()
        taken.append(b)
    for o in opts:
        ax.plot([], [], '-', color=f'C{opts.index(o)}', label=o)
    if first['steps'][0] or steps:
        ax.plot([], [], ' ', label='step')  # a group heading
    if first['steps'][0]:
        ax.plot([], [], 'o', ms=2.4, mfc='k', mec='0.5', mew=0.8, label=f'{first["steps"][0]}')
    for m, s in zip(STEP_MARKERS, steps):
        ax.plot([], [], m, ms=4, color='w', mec='k', label=f'{s}')
    if losses:
        ax.plot([], [], ' ', label='loss')
    for m, L in zip(MARKERS, losses):
        ax.plot([], [], m, ms=4, color='w', mec='k', label=f'{L:g}')
    ax.set(xticks=[], yticks=[], title=title)
    leg = ax.legend(fontsize=7, labelspacing=0.2, handlelength=1.5, handletextpad=0.4, borderpad=0.3)
    for t in leg.get_texts():
        if t.get_text() in ('step', 'loss'):  # the group headings start at the legend's left edge, above the markers
            t.set_transform(t.get_transform() + ScaledTranslation(-(1.5 + 0.4) * 7 / 72, 0, ax.figure.dpi_scale_trans))


def plot(panels, path):
    """panels: {title: (trajectories, losses to mark)}."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), squeeze=False)
    for ax, (title, (d, losses)) in zip(axes[0], panels.items()):
        draw(ax, d, losses, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--plot', nargs='+', metavar='TITLE=JSON[@L1,L2,...]',
                   help='only plot earlier trajectories.json files, side by side, each with its own losses')
    p.add_argument('--losses', type=float, nargs='+', default=[], help='training losses to mark on each path')
    p.add_argument('--rows', type=int, default=512, help='probe rows used for the distance')
    p.add_argument('--method', nargs='+', default=['umap'], choices=['umap', 'densmap', 'mds'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--name', default='trajectories.png', help='file name of the --plot figure (.pdf for vector)')
    args = p.parse_args()
    if args.plot:
        panels = {t: parse(f, args.losses) for t, f in (a.split('=', 1) for a in args.plot)}
        return plot(panels, Path(args.out) / args.name)

    runs = [r for r in load_runs(args.runs) if not r.cfg.fork.parent and r.done]
    n = runs[0].logp(runs[0].pred_steps[0]).shape[0]
    rows = torch.linspace(0, n - 1, min(args.rows, n)).long()
    index, P = [], []
    for r in runs:
        steps, loss = loss_points(r)
        index += [(r, s, l) for s, l in zip(steps, loss)]
        P += [r.logp(int(s))[rows].float().exp().to(args.device) for s in steps]
    D = np.sqrt(js_matrix(torch.stack(P)).cpu().numpy())
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / 'trajectories_dist.npz', dist=D, run=[r.id for r, _, _ in index],
                        step=[s for _, s, _ in index])
    for method in args.method:
        xy = embed(D, method, args.seed)
        out = {}
        for (r, s, l), z in zip(index, xy):
            d = out.setdefault(r.id, dict(optimizer=r.optimizer, steps=[], loss=[], xy=[]))
            d['steps'].append(int(s))
            d['loss'].append(float(l))
            d['xy'].append([float(v) for v in z])
        name = 'trajectories' if method == 'umap' else f'trajectories_{method}'
        (path / f'{name}.json').write_text(json.dumps(out))
        plot({f'{path.name}, {method}': (out, args.losses)}, path / 'figures' / f'{name}.png')


if __name__ == '__main__':
    main()
