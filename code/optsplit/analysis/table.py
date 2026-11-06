"""Status and final losses of runs, e.g. to pick the best learning rate of a sweep:
python -m optsplit.analysis.table 'ts10m_lrsweep/*' [--sort val_loss]"""

import argparse

from .io import load_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='glob patterns of run ids')
    ap.add_argument('--sort', default=None, help='column to sort by (e.g. val_loss)')
    args = ap.parse_args()
    rows = []
    for run in load_runs(args.runs):
        ev = run.records_of('eval')
        last = ev[-1] if ev else {}
        rows.append(dict(run=run.id, status='done' if run.done else 'running', step=last.get('step', 0),
                         lr=run.lr, train_loss=last.get('train_loss', float('nan')),
                         val_loss=last.get('val_loss', float('nan')), val_acc=last.get('val_acc', float('nan'))))
    if args.sort:
        rows.sort(key=lambda r: r[args.sort])
    print(f'{"run":<48} {"status":<8} {"step":>7} {"lr":>9} {"train":>7} {"val":>7} {"acc":>6}')
    for r in rows:
        print(f'{r["run"]:<48} {r["status"]:<8} {r["step"]:>7} {r["lr"]:>9.3g} {r["train_loss"]:>7.4f} '
              f'{r["val_loss"]:>7.4f} {r["val_acc"]:>6.3f}')


if __name__ == '__main__':
    main()
