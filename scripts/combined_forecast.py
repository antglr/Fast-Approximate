# Next-shot forecast = hybrid level x (1 + ridge correction), the correction applied once a segment is AGE_MIN shots old.
# The hybrid (forecasters.hybrid, charge mode, z=6, wmax=20: the setting that worked on both runs) restarts at level
# changes. The ridge predicts its relative error on the next shot from features computed only inside the current
# segment, so a level change resets them too (a long mean across a change would pull back to the old level).
# Everything is relative to the current level, so one model serves 1 to 20 nC. To try more inputs, add columns in
# features(). The correction is trained on stable shots of each run's first 90 %, on the pooled first 90 % of all runs, and on the other runs only;
# scored on each run's last 10 % with score_forecast (h=1).
# Usage: python -I combined_forecast.py <tables dir> <out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from forecasters import hybrid
from nbfuncs import TARGET, cut_shot, rolling_baseline, score_forecast, scored_shots
from run_data import level_labels, load_run

Z, WMAX = 6, 20
AGE_MIN = 20                   # apply the correction once the segment is this many shots old (stable beam)
ALPHAS = np.logspace(-3, 4, 30)


def seg_mean(v, seg, k):
    # trailing mean over the last k shots, never reaching back past the start of the current segment
    return v.groupby(seg).transform(lambda s: s.rolling(k, min_periods=1).mean())


def features(Q, x, p, seg, used):
    # relative deviations, known after shot i, for predicting shot i+1. Add new inputs here.
    q = Q.where(used)
    F = pd.DataFrame(index=Q.index)
    for k in (1, 3, 50, 200):
        F[f'q{k}'] = seg_mean(q, seg, k) / p - 1                      # p is the 20-shot capped segment mean
    for name in ('lab', 'clex', 'step'):
        level = seg_mean(x[name], seg, 200)
        # the step is already in nC and sits near 0 (with an offset) at low charge: scale it by the charge level instead
        scale = p if name == 'step' else level
        for k in (1, 9):
            F[f'{name}{k}'] = (seg_mean(x[name], seg, k) - level) / scale
    return F.replace([np.inf, -np.inf], np.nan).fillna(0)


def prepare(tables, run):
    df, x, bcm_bad, good = load_run(tables, run)
    Q = df[TARGET]
    q = Q.where(good)
    cut = cut_shot(df)
    train = df.index < cut
    med, jump, stable = level_labels(q, good)
    sig_q = (q / med - 1)[stable & train].std()
    sig_l = (x.lab / x.lab.rolling(41, center=True, min_periods=10).median() - 1)[stable & train].std()
    step_med = x.step.rolling(5, min_periods=1).median().values
    p, new, used = hybrid(Q.values, x.lab.values, step_med, 'charge', Z, WMAX, sig_q, sig_l)
    p = pd.Series(p, df.index)
    seg = pd.Series(np.cumsum(new), df.index)
    F = features(Q, x, p, seg, pd.Series(used, df.index))
    on = (seg.groupby(seg).cumcount() >= AGE_MIN).values
    y = (q.shift(-1) / p - 1).values                                  # relative error of the hybrid, next shot
    # train the correction on stable beam only: during scans the charge ramps without a restart, and learning to chase
    # the last shots there adds noise on stable beam. (The label looks ahead, but only selects training rows.)
    fit = train & (df.index + 1 < cut) & np.isfinite(y) & stable.shift(-1, fill_value=False).values & on
    return dict(df=df, p=p, F=F, y=y, fit=fit, on=on, jump=jump, stable=stable, train=train, q=q, med=med)


def score(r, pred_next):
    # pred_next: forecast made after shot i, for shot i+1
    return score_forecast(r['df'], pd.DataFrame({'shot': r['df'].index + 1, 'h': 1, 'pred': np.asarray(pred_next)})).MAE[1]


tables, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
R = {run: prepare(tables, run) for run in runs}
fits = {f'own run': {run: [run] for run in runs}, 'pooled runs': {run: runs for run in runs}}
if len(runs) > 1:
    fits['other runs only'] = {run: [o for o in runs if o != run] for run in runs}
models = {}
for kind, src in fits.items():
    for run, use in src.items():
        key = tuple(use)
        if key not in models:
            X = pd.concat([R[u]['F'][R[u]['fit']] for u in use])
            y = np.concatenate([R[u]['y'][R[u]['fit']] for u in use])
            models[key] = RidgeCV(alphas=ALPHAS).fit(X, y)

rows = []
for run, r in R.items():
    shots = scored_shots(r['df'])
    yq = r['df'].loc[shots, TARGET].values
    st, ch = (r['stable'] & ~r['train']).reindex(shots).values, r['jump'].reindex(shots).values
    cands = {'9-shot mean (baseline)': rolling_baseline(r['df']).values, f'hybrid (z={Z}, wmax={WMAX})': r['p'].values}
    for kind, src in fits.items():
        cands[f'combined, ridge on {kind}'] = r['p'].values * (1 + r['on'] * models[tuple(src[run])].predict(r['F']))
    for name, v in cands.items():
        e = np.abs(yq - pd.Series(v, r['df'].index).shift(1).reindex(shots).values)
        rows.append({'run': run, 'forecast': name, 'test MAE': score(r, v), 'stable': np.nanmean(e[st]),
                     'after change': np.nanmean(e[ch]) if ch.any() else np.nan, 'n stable': st.sum(), 'n change': ch.sum()})
    r['cands'] = cands
T = pd.DataFrame(rows)
T['dMAE %'] = T.groupby('run')['test MAE'].transform(lambda s: 100 * (1 - s / s.iloc[0]))
pd.set_option('display.width', 250)
print('next-shot MAE (nC) on each run\'s test region (last 10 %); stable / after change = the test shots in each regime')
print(T.round(4).to_string(index=False))
m = models[tuple(runs)]
print(f'\npooled ridge (alpha {m.alpha_:.3g}): relative correction per std of each feature, in % of the level')
X = pd.concat([R[u]['F'][R[u]['fit']] for u in runs])
print('  ' + '  '.join(f'{k} {100 * c * s:+.3f}' for k, c, s in zip(X.columns, m.coef_, X.std())))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
for run, r in R.items():
    shots = scored_shots(r['df'])
    yq = r['df'].loc[shots, TARGET].values
    show = {'9-shot mean (baseline)': 'C0', f'hybrid (z={Z}, wmax={WMAX})': 'C2', 'combined, ridge on pooled runs': 'C3'}
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 1]})
    ax[0].plot(shots, yq, '.', color='k', ms=2, label='actual BCM_GUN')
    for k, c in show.items():
        v = pd.Series(r['cands'][k], r['df'].index).shift(1).reindex(shots).values
        ax[0].plot(shots, v, '-', color=c, lw=0.8, label=f'{k}: MAE {np.nanmean(np.abs(yq - v)):.4f} nC')
        d = 100 * (yq - v) / r['med'].reindex(shots).values
        ax[1].hist(d[np.isfinite(d)], np.linspace(-20, 20, 81), histtype='step', color=c,
                   label=f'{k}: mean |error| {np.nanmean(np.abs(d)):.2f} %, median {np.nanmedian(np.abs(d)):.2f} %')
    ax[0].set_xlabel('shot'); ax[0].set_ylabel('charge (nC)'); ax[0].legend(loc='upper left', fontsize=8)
    ax[0].set_title(f'{run}: next-shot forecast, test region')
    ax[1].set_yscale('log'); ax[1].set_xlabel('(measured - predicted) / level (%)'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'combined_{run}.png'), dpi=110)
    print('->', os.path.join(out, f'combined_{run}.png'))
