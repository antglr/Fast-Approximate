# Do more per-shot readings help the jitter model? Screens two groups not in jitter_sequence.py's inputs against its
# current-shot ridge (same target, fast parts, split and regimes):
#   laser shape  the laser-scope pulse beyond its matched-filter amplitude (= extract_features.py's laser_amp, already
#                in as 'scope'): NPC principal components of the pulse / amplitude (fitted on each run's first 90 %),
#                centroid and width
#   RF           flat-top means before the beam (samples 300-630) and their slopes: gun incident amplitude and phase,
#                gun reflected, klystron phase and reflected
# Prints, per regime, the r of each feature with the ridge's training residual, and the test-region jitter explained of
# ridge / ridge + group / group only.
# Usage: python -I jitter_input_screen.py <tables dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from jitter_sequence import features, trailing
from pulses import align, pulse_ids, rf_shift, stamp_shift
from same_shot_regression import ALPHAS, CV, REGIMES, stitched

NPC = 5
RF = ['PGIA', 'PGRA', 'PGIP', 'PKI15P', 'PKR15A']


def laser_shape(tables, run):
    L = np.load(os.path.join(tables, run + '_laser.npz'))
    sc, t = L['scope'].astype(float), L['scope_times'][0]
    clean = sc - sc[:, t < 5].mean(1, keepdims=True)
    win = (t > 5.6) & (t < 39.8)
    tmpl = np.nanmean(clean, 0) * win
    amp = clean @ tmpl / (tmpl @ tmpl)
    bad = np.r_[False, np.all(sc[1:] == sc[:-1], 1)] | ~L['scope_acquired'].astype(bool) | L['scope_saturated'].astype(bool) | (amp <= 0)
    shape = clean[:, win] / np.where(bad, 1, amp)[:, None]
    fitrows = ~bad & (np.arange(len(sc)) < 0.9 * len(sc))
    mu = shape[fitrows].mean(0)
    pcs = (shape - mu) @ np.linalg.svd(shape[fitrows] - mu, full_matrices=False)[2][:NPC].T
    w = np.clip(clean[:, win], 0, None)
    cen = (w * t[win]).sum(1) / w.sum(1)
    M = np.c_[pcs, cen, np.sqrt((w * (t[win] - cen[:, None]) ** 2).sum(1) / w.sum(1))]
    M[bad] = np.nan
    names = [f'scope PC{k + 1}' for k in range(NPC)] + ['scope centroid', 'scope width']
    return pd.DataFrame(align(M, pulse_ids(L['t'], L['chunk']), stamp_shift(L['scope_off'])), columns=names).assign(shot=L['shot'])


def rf(tables, run):
    d = np.load(os.path.join(tables, run + '_rf.npz'))
    pid, cols = pulse_ids(d['t'], d['chunk']), {}
    for k in RF:
        a = align(d[k], pid, rf_shift(d[k + '_off']))
        cols[k], cols[k + ' slope'] = a[:, 300:630].mean(1), a[:, 560:630].mean(1) - a[:, 300:370].mean(1)
    return pd.DataFrame(cols).assign(shot=d['shot'])


def main():
    tables, runs = sys.argv[1], sys.argv[2:]
    F = stitched(tables, runs)
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    X = features(F)
    ok = level.notna()
    fit, test = F.train & ok, ~F.train & ok
    rows = []
    for gname, get in (('laser shape', laser_shape), ('RF', rf)):
        E = pd.concat([get(tables, run).assign(run=run) for run in sorted(runs)])
        G = F[['run', 'shot']].merge(E, on=['run', 'shot'], how='left').set_index(F.index).drop(columns=['run', 'shot'])
        G = G.loc[:, G.std() > 0]                                # PGIA is not recorded in some runs
        Xe = G - trailing(G, F.seg)
        Xe = Xe.fillna(0).assign(missing=Xe.isna().any(axis=1).astype(float))
        print(f'{gname}: {list(G.columns)}, missing on {Xe.missing[ok].mean():.1%} of shots')
        for rname in REGIMES:
            R = (F.regime == rname) & ok
            m = RidgeCV(alphas=ALPHAS, cv=CV).fit(X[fit & R], yf[fit & R])
            res = yf[fit & R] - m.predict(X[fit & R])
            r = {c: np.corrcoef(Xe.loc[fit & R, c], res)[0, 1] for c in G.columns if Xe.loc[fit & R, c].std() > 0}
            print(f'  {rname}: r with the ridge residual, largest |r| {max(r.values(), key=abs):+.3f} ({max(r, key=lambda c: abs(r[c]))})')
            for name, XX in (('ridge', X), (f'ridge + {gname}', pd.concat([X, Xe], axis=1)), (f'{gname} only', Xe)):
                p = pd.Series(RidgeCV(alphas=ALPHAS, cv=CV).fit(XX[fit & R], yf[fit & R]).predict(XX[R]), F.index[R])
                for s in sorted(set(F.seg[R & test])):
                    i = F.index[R & test & (F.seg == s)]
                    rows.append({'model': name, 'seg': s, 'jitter explained %': 100 * (1 - ((yf[i] - p[i]) ** 2).mean() / (yf[i] ** 2).mean())})
    D = pd.DataFrame(rows).drop_duplicates(['model', 'seg'])
    print('\njitter explained %, test regions')
    print(D.pivot(index='model', columns='seg', values='jitter explained %').reindex(dict.fromkeys(D.model)).round(1).to_string())


if __name__ == '__main__':
    main()
