# Shot-to-shot charge jitter from the instruments' own jitter: ridge vs RNN vs GRU (the fast-model step of
# docs/charge_models_explained.md; still same-shot, no forecasting). Causal slow/fast split of every signal within its
# clean segment:
#   slow = mean of the previous SLOW readings (of the logs; linear for step, mean_x, tilt), fast = reading - slow
# Target: the BCM's fast part, log Q[n] - mean log Q of the previous SLOW good shots: the jitter against past BCM
# readings. The level is that trailing mean, so the total charge = exp(level + predicted jitter).
# Inputs: the fast parts of same_shot_regression's 'reduced set + clex, scope' (0 where missing, plus a missing flag
# per instrument) of the current shot and the L-1 before it in the segment; optionally the past BCM jitter (each window
# step carries the previous good shot's target, never the current one). One model per regime:
#   ridge      current shot only, or + the last LAGS targets (RidgeCV, contiguous folds, squared loss)
#   RNN, GRU   torch.nn.RNN / GRU on windows of L shots, as gru_forecast.py: training-std scaling clipped at CLIP,
#              zero-initialised output (starts at 'no jitter' and may stay there), L1 loss, early stop on the last 10 %
#              of each segment's training part, SEEDS averaged
#   ridge + GRU  the GRU learns the ridge's residual, so the ridge keeps extrapolating where the GRU saturates
# Split and test regions as same_shot_regression.py. Scores per test segment: jitter explained = 1 - mean square of
# (target - prediction) / mean square of the target; MAE of the total charge in % of the level and in pC.
# Usage: python -I jitter_sequence.py <tables dir> <out dir> <run> [<run> ...]
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from same_shot_regression import ALPHAS, CV, GROUP, LINEAR, REGIMES, SETS, stitched

SLOW = 50               # shots in the trailing level (1 min)
LAGS = 16               # past BCM targets in the ridge
# (cell, window L, past BCM input, ridge whose residual it learns or None)
NETS = [(c, L, False, None) for L in (1, 16, 64) for c in ('RNN', 'GRU')] + [
    ('GRU', 16, True, None), ('GRU', 1, False, 'ridge, current shot'), ('GRU', 16, False, 'ridge, current shot'),
    ('GRU', 16, True, 'ridge, + past BCM'), ('GRU', 64, True, 'ridge, + past BCM')]
H = 32
SEEDS = (0, 1, 2)
BATCH, EPOCHS, PATIENCE, LR, CLIP = 256, 40, 5, 1e-3, 5
NAMES = SETS['reduced set + clex, scope']


def trailing(v, seg):
    # mean of the previous SLOW readings within the segment (gaps skipped); NaN until 5 are there
    return v.groupby(seg).transform(lambda c: c.shift(1).rolling(SLOW, min_periods=5).mean())


def features(F):
    Xr = pd.DataFrame({k: F[k] if k in LINEAR else np.log(F[k].where(F[k] > 0)) for k in NAMES})
    X = (Xr - trailing(Xr, F.seg)).fillna(0)
    for g in sorted({GROUP[k] for k in NAMES}):
        X[f'{g} missing'] = Xr[[k for k in NAMES if GROUP[k] == g]].isna().any(axis=1).astype(float)
    return X


class Seq(torch.nn.Module):
    def __init__(self, cell, scale):
        super().__init__()
        self.register_buffer('scale', scale)
        self.rnn, self.out = getattr(torch.nn, cell)(len(scale), H, batch_first=True), torch.nn.Linear(H, 1)

    def forward(self, X):
        return self.out(self.rnn((X / self.scale).clamp(-CLIP, CLIP))[0][:, -1]).squeeze(-1)


def predict(model, A, I):
    with torch.no_grad():
        return torch.cat([model(A[b]) for b in torch.from_numpy(I).split(4096)]).numpy()


def fit(cell, A, I, y, tr, va, seed):
    # A: rows x channels (+ a zero pad row), I: rows x L indices into A; tr, va: row numbers
    torch.manual_seed(seed)
    model = Seq(cell, A[tr].std(0).clamp(min=1e-6))
    torch.nn.init.zeros_(model.out.weight); torch.nn.init.zeros_(model.out.bias)
    opt = torch.optim.Adam(model.parameters(), LR)
    It, yt = torch.from_numpy(I[tr]), torch.from_numpy(y[tr])
    best, state, wait = np.mean(np.abs(y[va])), {k: t.clone() for k, t in model.state_dict().items()}, 0
    for epoch in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(It)).split(BATCH):
            loss = (model(A[It[b]]) - yt[b]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v = np.mean(np.abs(predict(model, A, I[va]) - y[va]))
        if v < best:
            best, state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return model, epoch + 1


def main():
    tables, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    torch.set_num_threads(4)
    F = stitched(tables, runs)
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    X = features(F)
    ok = level.notna()
    fit_rows = F.train & ok
    k = fit_rows.groupby(F.seg).cumsum()
    va = fit_rows & (k > 0.9 * k.groupby(F.seg).transform('max'))            # last 10 % of each training part
    pos = F.groupby('seg').cumcount().values
    past = pd.concat({f'BCM lag {j}': yf.groupby(F.seg).shift(j) for j in range(1, LAGS + 1)}, axis=1).fillna(0)
    ridge_X = {'ridge, current shot': X, 'ridge, + past BCM': pd.concat([X, past], axis=1)}
    A = torch.from_numpy(np.vstack([np.c_[X.values, past['BCM lag 1'].values], np.zeros((1, X.shape[1] + 1))]).astype(np.float32))
    y = yf.fillna(0).values.astype(np.float32)

    def windows(L):                                  # rows x L indices, the pad row before the segment starts
        lag = np.arange(L - 1, -1, -1)
        return np.where(pos[:, None] >= lag, np.arange(len(F))[:, None] - lag, len(F))
    I1 = windows(1)
    assert np.array_equal(A[I1[:, 0], :-1].numpy(), X.values.astype(np.float32))
    j = np.flatnonzero(ok & (pos > 1))[100]
    assert A[j, -1] == np.float32(yf.iloc[j - 1])                            # the window's BCM is the previous shot's

    P = {'trailing level only': pd.Series(0.0, F.index)}
    for rname in REGIMES:
        R = (F.regime == rname) & ok
        base = {}                                    # ridge prediction on every row (0 outside the regime)
        for name, XR in ridge_X.items():
            m = RidgeCV(alphas=ALPHAS, cv=CV).fit(XR[fit_rows & R], yf[fit_rows & R])
            base[name] = np.where(R, m.predict(XR), 0).astype(np.float32)
            P.setdefault(name, pd.Series(np.nan, F.index))[R] = base[name][R]
            print(f'{rname}, {name}: alpha {m.alpha_:.3g}', flush=True)
        tr, v = np.flatnonzero(fit_rows & R & ~va), np.flatnonzero(va & R)
        print(f'{rname}: {len(tr)} training, {len(v)} early-stopping shots', flush=True)
        for cell, L, use_past, on in NETS:
            I, Ac = windows(L), A if use_past else A[:, :-1]
            b = base[on] if on else np.zeros_like(y)
            t0, preds, eps = time.time(), [], []
            for seed in SEEDS:
                model, ep = fit(cell, Ac, I, y - b, tr, v, seed)
                preds.append(predict(model, Ac, I[R.values])); eps.append(ep)
            name = f'{on} + ' * bool(on) + f'{cell}, L={L}' + ', past BCM' * use_past
            P.setdefault(name, pd.Series(np.nan, F.index))[R] = b[R] + np.mean(preds, 0)
            print(f'  {name}: epochs {eps}, {time.time() - t0:.0f} s', flush=True)

    test = ~F.train & ok
    rows = []
    for name, p in P.items():
        for s, g in F[test].groupby('seg'):
            i = g.index
            rows.append({'model': name, 'seg': s,
                         'jitter explained %': 100 * (1 - ((yf[i] - p[i]) ** 2).mean() / (yf[i] ** 2).mean()),
                         'rel %': 100 * (np.exp(level[i] + p[i]) - F.Q[i]).abs().mean() / F.Q[i].mean(),
                         'MAE pC': 1000 * (np.exp(level[i] + p[i]) - F.Q[i]).abs().mean()})
    D = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    info = F[test].groupby('seg').agg(run=('run', lambda r: r.iloc[0][5:16]), regime=('regime', 'first'), n=('Q', 'size'),
                                      level=('Q', 'mean'))
    info['target rms %'] = 100 * np.sqrt((yf[test] ** 2).groupby(F.seg[test]).mean())
    print(f'\ntest regions (last 10 % of each clean segment); target = log(Q / geometric mean of the previous {SLOW} good BCM readings)')
    print(info.round(3).to_string())
    for col in ('jitter explained %', 'rel %', 'MAE pC'):
        print(f'\n{col}' + (' (MAE of exp(level + predicted jitter) in % of the level)' if col == 'rel %' else ''))
        print(D.pivot_table(index='model', columns='seg', values=col).reindex(P).round(2 if col == 'rel %' else 1).to_string())

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best = max((n for n in P if 'L=' in n), key=lambda n: D[D.model == n]['jitter explained %'].mean())
    segs = sorted(set(F.seg[test]))
    fig, axes = plt.subplots(len(segs), 1, figsize=(15, 2.4 * len(segs)))
    for a, s in zip(axes, segs):
        i = F.index[test & (F.seg == s)]
        a.plot(i, 100 * yf[i], '.-', color='k', ms=2, lw=0.4, label='BCM jitter')
        for n, c in (('ridge, current shot', 'C0'), (best, 'C3')):
            a.plot(i, 100 * P[n][i], '-', color=c, lw=0.8, label=n)
        a.set_title(f'segment {s} ({info.run[s]}, {info.regime[s]}): test region', fontsize=9)
        a.set_ylabel('% of level'); a.legend(loc='upper left', fontsize=7)
    axes[-1].set_xlabel('row (stitched good shots)')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'jitter_sequence.png'), dpi=100)
    print('->', os.path.join(out, 'jitter_sequence.png'))


if __name__ == '__main__':
    main()
