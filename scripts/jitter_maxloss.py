# Minimising the largest error instead of the mean: linear models of the jitter (jitter_sequence.py's target and
# current-shot inputs, standardised; one per regime) fitted with the mean of |error|^p for p = 1, 2, 4, 8, and the
# maximum |error| (p = inf, Chebyshev regression, a linear program). Larger p weighs the worst shots more. Training
# rows exclude 5-MAD outliers of the target: BCM-lost shots (BCM ~0) and spikes would otherwise set the maximum alone.
# Test scores per segment, in pC: MAE, 99th and 99.9th percentile and maximum of |error|, on all shots and without the
# BCM-lost ones (BCM < 0.5 x the trailing level); the best offline model (L1-trained GRU) for comparison.
# Usage: python -I jitter_maxloss.py <tables dir> <jitter_sequence out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from jitter_sequence import features, trailing
from same_shot_regression import REGIMES, stitched

PS = (1, 2, 4, 8, np.inf)
BEST = 'ridge, + past BCM + GRU, L=64, past BCM'


def fit(Z, y, p):
    # weights (last = intercept) minimising mean |y - Z w|^p, or max |y - Z w| for p = inf
    Z = np.c_[Z, np.ones(len(Z))]
    if p == np.inf:                   # min t  s.t.  -t <= y - Z w <= t
        n, k = Z.shape
        c = np.r_[np.zeros(k), 1.0]
        Aub = np.block([[Z, -np.ones((n, 1))], [-Z, -np.ones((n, 1))]])
        r = linprog(c, A_ub=Aub, b_ub=np.r_[y, -y], bounds=[(None, None)] * k + [(0, None)], method='highs')
        assert r.success, r.message
        return r.x[:k]
    w0 = np.linalg.lstsq(Z, y, rcond=None)[0]
    s = np.abs(y - Z @ w0).mean()     # scale so the loss stays O(1) for large p
    f = lambda w: (np.mean(np.abs((y - Z @ w) / s) ** p), -p / len(y) / s * Z.T @ (np.sign(y - Z @ w) * np.abs((y - Z @ w) / s) ** (p - 1)))
    return minimize(f, w0, jac=True, method='L-BFGS-B', options={'maxiter': 2000}).x


def main():
    tables, pdir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    F = stitched(tables, runs)
    P = pd.read_parquet(os.path.join(pdir, 'jitter_predictions.parquet'))
    assert (P.run.values == F.run.values).all() and (P.shot.values == F.shot.values).all()
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    X = features(F)
    ok = level.notna()
    # self-check: on noise-free data every loss recovers the line; with bounded noise L-inf has the smallest maximum
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 2)); yy = z @ [1, -2] + 0.5 + rng.uniform(-0.1, 0.1, 500)
    W = {p: fit(z, yy, p) for p in PS}
    assert all(np.allclose(w, [1, -2, 0.5], atol=0.03) for w in W.values())
    mx = {p: np.abs(yy - np.c_[z, np.ones(500)] @ w).max() for p, w in W.items()}
    assert mx[np.inf] <= min(mx.values()) + 1e-9

    out = {f'linear, p={p}': pd.Series(np.nan, F.index) for p in PS}
    for rname in REGIMES:
        R = (F.regime == rname) & ok
        tr = F.train & R
        dev = (yf[tr] - yf[tr].median()).abs()
        tr = tr & (yf - yf[tr].median()).abs().lt(5 * 1.4826 * dev.median())
        sd = X[tr].std().replace(0, 1)
        for p in PS:
            w = fit((X[tr] / sd).values, yf[tr].values, p)
            out[f'linear, p={p}'][R] = np.c_[(X[R] / sd).values, np.ones(R.sum())] @ w
            print(f'{rname}, p={p}: training max |error| {100 * np.abs(yf[tr] - out[f"linear, p={p}"][tr]).max():.2f} %', flush=True)
    out[BEST] = P[BEST]

    test = ~F.train & ok
    lost = yf < np.log(0.5)
    rows = []
    for name, p in out.items():
        for s, g in F[test].groupby('seg'):
            for sub, i in (('all', g.index), ('no BCM-lost', g.index[~lost[g.index]])):
                e = 1000 * (np.exp(level[i] + p[i]) - F.Q[i]).abs()
                rows.append({'model': name, 'seg': s, 'shots': sub, 'MAE': e.mean(), 'p99': e.quantile(0.99), 'p99.9': e.quantile(0.999), 'max': e.max()})
    D = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(f'\nBCM-lost test shots per segment: {lost[test].groupby(F.seg[test]).sum().to_dict()}')
    for sub in ('all', 'no BCM-lost'):
        for col in ('MAE', 'p99', 'p99.9', 'max'):
            print(f'\n|error| pC, {col}, test regions, {sub} shots')
            print(D[D.shots == sub].pivot_table(index='model', columns='seg', values=col).reindex(out).round(1).to_string())


if __name__ == '__main__':
    main()
