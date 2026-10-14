# Same-shot charge from the reduced feature set (no BCM history, no forecasting): log(Q) = ridge on the shot's own
# readings, one model per charge regime. The reduced set is extract_features.py's FEATURES on our pulse grid, with the
# windowed beam-loading step in place of PGLA_drop; '+ clex, scope' adds the other laser monitors.
# Positive readings enter as logs (charge ~ laser energy x QE, screen light ~ charge), the rest linearly. A missing
# reading takes the segment's last one, plus a flag per instrument (lab, clex, step, scope, camera, RF).
# Split: the first 90 % of each clean segment trains, the last 10 % tests. Four ways to use a regime's segments:
#   absolute                 one intercept for the regime
#   per-segment calibration  each segment's own intercept (its training-part mean), shared slopes: fixed effects
#   new segment              slopes and intercept from the regime's other segments only (a segment never seen)
#   new segment, calibrated  slopes from the other segments, intercept from this segment's training part
# Scores per segment test region: MAE (nC), MAE in % of the level, and r of the jitter (51-shot centred median removed
# from prediction and BCM, analysis only).
# Usage: python -I same_shot_regression.py <tables dir> <out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from nbfuncs import TARGET, TEST_FRAC
from pulses import align, pulse_ids, rf_shift
from run_data import kept_chunks, load_run
from virtual_bcm import scope_amp

# clean segments in the stitched order of results/datasets/manifest.json (runs sorted, then chunks)
REGIMES = {'low charge': [0, 2, 3, 4], 'high, repetitive': [5], 'high, non-repetitive': [1, 6, 7]}
CAMERA = ['BTV_intensity', 'BTV_amplitude', 'BTV_mean_x', 'BTV_sigma_major', 'BTV_sigma_minor', 'BTV_tilt',
          'BTV_ellipticity', *[f'SVD_s{k}' for k in range(1, 7)]]
SETS = {'lab only': ['lab'], 'reduced set': ['lab', 'step', 'PKI15A', *CAMERA],
        'reduced set + clex, scope': ['lab', 'step', 'PKI15A', *CAMERA, 'clex', 'scope']}
LINEAR = {'step', 'BTV_mean_x', 'BTV_tilt'}          # can be <= 0: not logged
GROUP = {'lab': 'lab', 'clex': 'clex', 'step': 'step', 'scope': 'scope', 'PKI15A': 'rf', **{c: 'camera' for c in CAMERA}}
CV = KFold(5)                                        # contiguous blocks: neighbouring shots share the drift
ALPHAS = np.logspace(-3, 4, 30)


def load(tables, run):
    # one row per good shot of the clean chunks: raw features (NaN = no reading of this pulse), Q, local segment number
    df, x, bcm_bad, good = load_run(tables, run)
    full = pd.read_parquet(os.path.join(tables, run + '.parquet'))
    kept = kept_chunks(full, run)
    seg = (kept != kept.shift()).cumsum().loc[df.index]
    d = np.load(os.path.join(tables, run + '_rf.npz'))
    pos = np.searchsorted(d['shot'], df.index)
    F = pd.DataFrame({k: x[k].where(x[k + '_ok']) for k in ('lab', 'clex', 'step')}, df.index)
    F['scope'] = scope_amp(tables, run).reindex(df.index)
    F['PKI15A'] = align(d['PKI15A'], pulse_ids(d['t'], d['chunk']), rf_shift(d['PKI15A_off']))[pos, 300:640].mean(1)
    C = np.load(os.path.join(tables, run + '_camera.npz'))
    assert np.array_equal(C['shot'], df.index)
    for k in CAMERA:
        F[k] = C[k]
    F['Q'], F['seg'] = df[TARGET], seg
    return F[kept.loc[df.index] & good & (df[TARGET] > 0)]


def design(F, names, train):
    # transformed features + missing flags; gaps take the segment's last reading, the very first ones the training mean
    X = pd.DataFrame(index=F.index)
    for k in names:
        X[k] = F[k] if k in LINEAR else np.log(F[k].where(F[k] > 0))
    for g in sorted({GROUP[k] for k in names}):
        X[f'{g} missing'] = X[[k for k in names if GROUP[k] == g]].isna().any(axis=1).astype(float)
    X = X.groupby(F['seg']).ffill()
    return X.fillna(X[train].mean())


def stitched(tables, runs):
    # all runs' rows with stitched segment numbers, regime and the training flag (first 90 % of each segment)
    parts = [load(tables, run).assign(run=run).rename_axis('shot').reset_index() for run in sorted(runs)]
    off = 0
    for p in parts:                                  # stitched segment numbers
        p['seg'] = p['seg'].rank(method='dense').astype(int) - 1 + off
        off = p['seg'].max() + 1
    F = pd.concat(parts, ignore_index=True)
    regime = {s: r for r, ss in REGIMES.items() for s in ss}
    assert set(F.seg) == set(regime), f'segments {sorted(set(F.seg))} vs REGIMES'
    F['regime'] = F.seg.map(regime)
    F['train'] = F.groupby('seg').cumcount() < (F.groupby('seg').Q.transform('size') * (1 - TEST_FRAC)).astype(int)
    return F


def main():
    tables, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    F = stitched(tables, runs)
    y = np.log(F.Q)

    def tmean(v, by='seg'):                          # each segment's (regime's) training-part mean, on every row
        return v[F.train].groupby(F[by][F.train]).mean().reindex(F[by]).set_axis(F.index)

    P = {}                                           # (feature set, mode) -> log-charge prediction on every row
    coefs = {}
    for sname, names in SETS.items():
        X = design(F, names, F.train)
        mx, my = tmean(X), tmean(y)                  # fixed effects: centre each segment on its own training part
        for mode in ('absolute', 'per-segment calibration', 'new segment', 'new segment, calibrated'):
            pred = pd.Series(np.nan, F.index)
            for rname, segs in REGIMES.items():
                for s in segs:
                    src = [o for o in segs if o != s] if mode.startswith('new') else segs
                    if not src:
                        continue                     # a regime with one segment has no 'new segment' case
                    tr = F.train & F.seg.isin(src) if not mode.startswith('new') else F.seg.isin(src)
                    calib = 'calibrat' in mode
                    Xc, yc = (X - mx, y - my) if calib else (X, y)
                    m = RidgeCV(alphas=ALPHAS, cv=CV).fit(Xc[tr], yc[tr])
                    at = F.seg == s
                    pred[at] = m.predict(Xc[at]) + (my[at] if calib else 0)
                    if mode == 'per-segment calibration' and s == segs[0]:
                        sd = Xc[tr].std()
                        coefs[(sname, rname)] = pd.Series(m.coef_ * sd.values * 100, X.columns)
            P[(sname, mode)] = pred
    # constant baselines: the regime's / the segment's training mean
    P[('constant', 'absolute')] = tmean(y, 'regime')
    P[('constant', 'per-segment calibration')] = tmean(y)

    def det(v):
        return v - v.groupby(F.seg).transform(lambda c: c.rolling(51, center=True, min_periods=10).median())

    rows = []
    for (sname, mode), lp in P.items():
        q = np.exp(lp)
        e = (q - F.Q).abs()
        jq, jp = det(F.Q), det(q)
        for s, g in F[~F.train].groupby('seg'):
            i = g.index
            ok = jq[i].notna() & jp[i].notna()
            rows.append({'features': sname, 'mode': mode, 'seg': s, 'MAE': e[i].mean(), 'rel %': 100 * e[i].mean() / F.Q[i].mean(),
                         'jitter r': np.corrcoef(jq[i][ok], jp[i][ok])[0, 1] if ok.sum() > 20 and jp[i][ok].std() > 0 else np.nan})
    R = pd.DataFrame(rows)
    info = F[~F.train].groupby('seg').agg(run=('run', lambda r: r.iloc[0][5:16]), regime=('regime', 'first'),
                                            n=('Q', 'size'), level=('Q', 'mean'))
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('test regions (last 10 % of each clean segment)')
    print(info.round(3).to_string())
    order = [k for k in dict.fromkeys(zip(R.features, R['mode']))]
    for col, fmt in (('rel %', 2), ('jitter r', 3), ('MAE', 4)):
        T = R.pivot_table(index=['features', 'mode'], columns='seg', values=col).reindex(order)
        print(f'\n{col} ({"MAE in % of the level" if col == "rel %" else col}), per test segment')
        print(T.round(fmt).to_string())

    # where is the error: split each residual (log, in %) into a slow part (51-shot centred median, analysis only) and
    # the fast rest; the BCM's own fast part is the jitter there is to explain
    def split(v):
        slow = v.groupby(F.seg).transform(lambda c: c.rolling(51, center=True, min_periods=10).median())
        return slow, v - slow
    _, bcm_fast = split(100 * y)
    rows = []
    for key in [('constant', 'per-segment calibration'), ('lab only', 'per-segment calibration'),
                ('reduced set + clex, scope', 'per-segment calibration'), ('reduced set + clex, scope', 'new segment, calibrated'),
                ('reduced set + clex, scope', 'new segment')]:
        slow, fast = split(100 * (P[key] - y))
        for s, g in F[~F.train].groupby('seg'):
            i = g.index
            if fast[i].isna().all():
                continue
            ms_s, ms_f = (slow[i] ** 2).mean(), (fast[i] ** 2).mean()
            rows.append({'model': f'{key[0]}, {key[1]}', 'seg': s, 'slow rms %': np.sqrt(ms_s), 'fast rms %': np.sqrt(ms_f), 'slow share %': 100 * ms_s / (ms_s + ms_f),
                         'jitter explained %': 100 * (1 - ms_f / (bcm_fast[i] ** 2).mean())})
    D = pd.DataFrame(rows)
    print('\nerror split, test regions: residual log(pred / BCM) in %, slow = its 51-shot centred median, fast = the rest;'
          '\njitter explained = 1 - fast mean square / the BCM\'s own fast mean square')
    print('BCM fast rms % per segment:', (bcm_fast[~F.train].groupby(F.seg[~F.train]).apply(lambda v: np.sqrt((v ** 2).mean()))).round(2).to_dict())
    for col in ('slow rms %', 'fast rms %', 'slow share %', 'jitter explained %'):
        print(f'\n{col}')
        print(D.pivot_table(index='model', columns='seg', values=col).reindex(dict.fromkeys(D.model)).round(1).to_string())
    print('\nper-segment calibration, change of log Q per std of each feature (%), largest 10')
    for (sname, rname), c in coefs.items():
        if sname != 'lab only':
            c = c.reindex(c.abs().sort_values(ascending=False).index)[:10]
            print(f'  {sname}, {rname}: ' + '  '.join(f'{n} {v:+.2f}' for n, v in c.items()))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [('constant', 'per-segment calibration', 'C0'), ('reduced set + clex, scope', 'per-segment calibration', 'C3'),
            ('reduced set + clex, scope', 'new segment', 'C2')]
    segs = sorted(set(F.seg))
    fig, axes = plt.subplots(len(segs), 1, figsize=(15, 2.6 * len(segs)))
    for a, s in zip(axes, segs):
        i = F.index[(F.seg == s) & ~F.train]
        a.plot(i, F.Q[i], '.', color='k', ms=2, label='BCM_GUN')
        for sname, mode, c in show:
            q = np.exp(P[(sname, mode)][i])
            if q.notna().any():
                a.plot(i, q, '-', color=c, lw=0.8, label=f'{sname}, {mode}: {100 * (q - F.Q[i]).abs().mean() / F.Q[i].mean():.2f} %')
        a.set_title(f'segment {s} ({info.run[s]}, {info.regime[s]}): test region', fontsize=9)
        a.set_ylabel('nC'); a.legend(loc='upper left', fontsize=7)
    axes[-1].set_xlabel('row (stitched good shots)')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'same_shot_regression.png'), dpi=100)
    print('->', os.path.join(out, 'same_shot_regression.png'))


if __name__ == '__main__':
    main()
