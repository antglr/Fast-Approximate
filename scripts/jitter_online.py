# Online correction of the jitter models: with the BCM known up to the previous shot (as the trailing level already
# assumes), a linear correction on the current shot's input jitters is refitted after every shot. Exponentially
# weighted ridge on the residual of an offline model of jitter_sequence.py:
#   w_n = argmin_w  sum_{m<n} LAM^(n-1-m) (r_m - w.z_m)^2 + RHO |w|^2,   r = target - offline prediction
# z = jitter_sequence.py's inputs, standardised by the regime's training std, plus a constant (the trailing level's
# lag). RHO pulls towards the offline model (w = 0); 1 / (1 - LAM) is the memory in shots. Each segment starts from the
# offline model and runs through its shots in time order: a shot's prediction uses only the shots before it.
# LAM and RHO (or no correction) are picked per regime and offline model on the early-stopping rows (last 10 % of each
# segment's training part), never on the test region. Scores as jitter_sequence.py.
# Usage: python -I jitter_online.py <tables dir> <jitter_sequence out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from jitter_sequence import features, trailing
from same_shot_regression import REGIMES, stitched

BASES = ['ridge, current shot', 'ridge, + past BCM + GRU, L=64, past BCM']
LAMS = (0.999, 0.9997, 0.9999)            # memory ~1000, 3300, 10000 shots
RHOS = (10, 100, 1000)                    # pull towards the offline model, in shots' worth of data


def online(z, r, seg, lam, rho):
    # the correction for each row from the rows before it in its segment
    out, I = np.zeros(len(r)), rho * np.eye(z.shape[1])
    for s in np.unique(seg):
        A, b = np.zeros_like(I), np.zeros(len(I))
        for i in np.flatnonzero(seg == s):
            out[i] = z[i] @ np.linalg.solve(A + I, b)
            A, b = lam * A + np.outer(z[i], z[i]), lam * b + z[i] * r[i]
    return out


def main():
    tables, pdir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    F = stitched(tables, runs)
    P = pd.read_parquet(os.path.join(pdir, 'jitter_predictions.parquet'))
    assert (P.run.values == F.run.values).all() and (P.shot.values == F.shot.values).all()
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    X = features(F)
    ok = level.notna()
    fit_rows = F.train & ok
    k = fit_rows.groupby(F.seg).cumsum()
    va = fit_rows & (k > 0.9 * k.groupby(F.seg).transform('max'))            # as jitter_sequence.py
    # self-check: a constant residual is learned within a few shots, and the first shot of a segment gets no correction
    c = online(np.ones((50, 1)), np.full(50, 0.1), np.zeros(50), 0.999, 1.0)
    assert c[0] == 0 and abs(c[-1] - 0.1) < 0.003

    out = {}
    for base in BASES:
        out[base] = P[base]
        pred = pd.Series(np.nan, F.index)
        for rname in REGIMES:
            R = (F.regime == rname) & ok
            z = (X[R] / X[fit_rows & R].std().replace(0, 1)).assign(const=1.0).values
            r, seg, v = (yf[R] - P[base][R]).values, F.seg[R].values, va[R].values
            corr = {None: np.zeros(len(r))}
            corr.update({(lam, rho): online(z, r, seg, lam, rho) for lam in LAMS for rho in RHOS})
            mse = {key: np.mean((r[v] - c[v]) ** 2) for key, c in corr.items()}
            best = min(mse, key=mse.get)
            print(f'{base}, {rname}: early-stopping-row mean square vs no correction, per (LAM, RHO):',
                  {key: round(float(m / mse[None]), 3) for key, m in mse.items() if key}, '-> use', best, flush=True)
            pred[R] = P[base][R] + corr[best]
        out[f'{base} + online'] = pred

    test = ~F.train & ok
    rows = []
    for name, p in out.items():
        for s, g in F[test].groupby('seg'):
            i = g.index
            e = np.exp(level[i] + p[i]) - F.Q[i]
            rows.append({'model': name, 'seg': s, 'jitter explained %': 100 * (1 - ((yf[i] - p[i]) ** 2).mean() / (yf[i] ** 2).mean()),
                         'rel %': 100 * e.abs().mean() / F.Q[i].mean(), 'MAE pC': 1000 * e.abs().mean()})
    D = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    for col in ('jitter explained %', 'rel %', 'MAE pC'):
        print(f'\n{col}, test regions')
        print(D.pivot_table(index='model', columns='seg', values=col).reindex(out).round(2 if col == 'rel %' else 1).to_string())


if __name__ == '__main__':
    main()
