"""Train the value network V(s) -> P(observer wins) on generated positions.

    python oracle/train_value.py --shards 'positions/round1b_shard*.npz' 'positions/round2_shard*.npz' --out checkpoints/value_r2 \
        --features all --hidden 512 --depth 2 --dropout 0.3 --wd 1e-3 --epochs 3
Held-out split is by game seed (10% of games). Reports log-loss, Brier, AUC overall, by turn and for terminal
states; saves model weights, input scaler and a JSON report. Needs torch (GPU optional: V r2 trained in 3.3 min on 32 CPU threads).
"""
from __future__ import annotations

import argparse, glob, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ValueMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 1024, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers, d = [], dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc(y_true, y_score):
    order = np.argsort(y_score)
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    pos = y_true > 0.5; n1 = pos.sum(); n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p):
    eps = 1e-6; p = np.clip(p, eps, 1 - eps)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {'n': int(len(y)), 'logloss': round(ll, 4), 'brier': round(float(np.mean((p - y) ** 2)), 4),
            'auc': round(auc(y[y != 0.5], p[y != 0.5]), 4) if (y != 0.5).sum() else float('nan'),
            'base_rate': round(float(y.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--holdout-mod', type=int, default=10)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--features', default='all', choices=['all', 'board'],
                    help="board = countries + global only (first 2421 dims); drops the card multi-hots, which identify a game")
    ap.add_argument('--no-aux', action='store_true', help='drop the aux block even if the shards carry it')
    ap.add_argument('--keep-terminal', action='store_true', help='keep terminal states (the encoding cannot tell who caused DEFCON 1, so they are noise)')
    ap.add_argument('--wd', type=float, default=1e-4)
    args = ap.parse_args()
    files = sorted(sum([glob.glob(s) for s in args.shards], []))
    Xs, ys, Ms = [], [], []
    for f in files:
        z = np.load(f); Xs.append(z['X']); ys.append(z['y']); Ms.append(z['meta']); print('loaded', f, len(z['y']), flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys); M = np.concatenate(Ms)
    if not args.keep_terminal:
        keep = M[:, 7] == 0; X, y, M = X[keep], y[keep], M[keep]
    sys.path.insert(0, str(Path(__file__).parent))
    from features import AUX_DIM
    has_aux = X.shape[1] == 3081 + AUX_DIM
    base_idx = list(range(2421 if args.features == 'board' else 3081))
    aux_idx = list(range(3081, 3081 + AUX_DIM)) if (has_aux and not args.no_aux) else []
    feat_idx = base_idx + aux_idx
    X = X[:, feat_idx]
    feat_dim = len(feat_idx)
    print(f'features: {args.features}, aux block {"used" if aux_idx else "absent/dropped"}, dim {feat_dim}', flush=True)
    seeds = M[:, 0]; held = (seeds % args.holdout_mod) == 0
    print(f'{len(y)} positions, {held.sum()} held out ({len(np.unique(seeds))} games), dim {X.shape[1]}', flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(X[~held].astype(np.float32)); ytr = torch.tensor(y[~held])
    Xte = torch.tensor(X[held].astype(np.float32)).to(dev); yte = y[held]; Mte = M[held]
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-3
    Xtr = ((Xtr - mean) / std); Xte = ((Xte - mean.to(dev)) / std.to(dev))
    model = ValueMLP(X.shape[1], args.hidden, args.depth, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(ytr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    report = {'files': files, 'positions': int(len(y)), 'held_out': int(held.sum()), 'epochs': []}
    t0 = time.perf_counter()
    best = (float('inf'), None, None)  # held-out logloss, state_dict copy, p
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(ytr)); tot = 0.0
        for i in range(0, len(ytr), args.bs):
            idx = perm[i:i + args.bs]
            xb = Xtr[idx].to(dev, non_blocking=True); yb = ytr[idx].to(dev)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(model(Xte[i:i + 8192])) for i in range(0, len(Xte), 8192)]).cpu().numpy()
        m = metrics(yte, p); m['train_logloss'] = round(tot / len(ytr), 4); m['epoch'] = ep + 1
        report['epochs'].append(m); print(m, flush=True)
        if m['logloss'] < best[0]:
            best = (m['logloss'], {k: v.detach().clone() for k, v in model.state_dict().items()}, p)
    model.load_state_dict(best[1]); p = best[2]; report['best_logloss'] = best[0]
    by = {}
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 10)):
        sel = (Mte[:, 2] >= lo) & (Mte[:, 2] <= hi) & (Mte[:, 7] == 0)
        if sel.sum():
            by[f'turn{lo}-{hi}'] = metrics(yte[sel], p[sel])
    by['terminal'] = metrics(yte[Mte[:, 7] == 1], p[Mte[:, 7] == 1])
    by['non_terminal'] = metrics(yte[Mte[:, 7] == 0], p[Mte[:, 7] == 0])
    # calibration (non-terminal)
    sel = Mte[:, 7] == 0; bins = np.linspace(0, 1, 11); cal = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = sel & (p >= a) & (p < b)
        if s.sum() > 50:
            cal.append((round(float(a), 1), int(s.sum()), round(float(p[s].mean()), 3), round(float(yte[s].mean()), 3)))
    report['by'] = by; report['calibration_nonterminal'] = cal; report['minutes'] = round((time.perf_counter() - t0) / 60, 1)
    report['model'] = {'dim': int(X.shape[1]), 'features': args.features, 'aux_used': bool(aux_idx), 'terminal_kept': args.keep_terminal, 'hidden': args.hidden, 'depth': args.depth, 'dropout': args.dropout, 'wd': args.wd, 'games': int(len(np.unique(seeds)))}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'dim': X.shape[1], 'hidden': args.hidden, 'depth': args.depth, 'feat_dim': feat_dim, 'feat_idx': feat_idx, 'aux': has_aux}, out / 'value.pt')
    json.dump(report, open(out / 'report.json', 'w'), indent=1)
    print(json.dumps(by, indent=1)); print('calibration (bin, n, mean p, mean y):', cal); print('saved', out)


if __name__ == '__main__':
    main()
