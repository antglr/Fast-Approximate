# Next-shot hybrid forecast: the mean of the good shots since the last level change, capped at wmax shots. On stable
# beam that is a long mean (the jitter is nearly white); after a change it restarts at the new level. A change is a
# charge deviation > z * jitter from the current mean, and the variants differ in what confirms it:
#   'charge'          the deviation alone
#   'charge+laser'    the lab meter moved the same way by > z * its own jitter (a dropout or BCM glitch does not)
#   'charge x2'       two shots in a row deviate the same way
# Everything the forecast uses is causal. The official dropout labels (centred median) are used only for scoring;
# the forecast skips shots by its own rules: unconfirmed drops of > 70 %, and BCM readings far below the trailing
# beam-loading-step charge. z and wmax are tuned on the training part (first 90 %); scored on the test region with
# score_forecast, and on the stable shots of the whole run.
# Usage: python -I hybrid_forecast.py <tables dir> <run> [<plot.png>]
import itertools, os, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from nbfuncs import TARGET, cut_shot, rolling_baseline, score_forecast, scored_shots
from run_data import level_labels, load_run

MODES = ['charge', 'charge+laser', 'charge x2']
ZS = [3, 4, 5, 6, 8]
WMAXES = [9, 20, 30, 50, 100]
DROP = -0.7                    # an unconfirmed fall by more than this (relative) is a dropout, not a level change


def hybrid(Q, lab, step_med, mode, z, wmax, sig_q, sig_l):
    # -> forecast for shot i+1 made after shot i (same index as Q), and whether shot i started a new segment
    pred, new = np.full(len(Q), np.nan), np.zeros(len(Q), bool)
    buf, sq, sl, cand = deque(), 0.0, 0.0, []
    last = np.nan
    for i, (qi, li, si) in enumerate(zip(Q, lab, step_med)):
        ok = np.isfinite(qi) and not (si > 1 and qi < 0.5 * si)          # BCM stuck low while the beam is there
        if ok and buf:
            m, ml = sq / len(buf), sl / len(buf)
            rq, rl = qi / m - 1, li / ml - 1
            big = abs(rq) > z * sig_q
            if mode == 'charge':
                change = big
            elif mode == 'charge+laser':
                change = big and abs(rl) > z * sig_l and np.sign(rl) == np.sign(rq)
            else:                                                       # two in a row, same direction
                change = big and bool(cand) and np.sign(cand[-1] / m - 1) == np.sign(rq)
                cand = [qi] if big and not change else []
            if change:
                start = ([cand[-1]] if mode == 'charge x2' and cand else []) + [qi]
                buf, sq, sl = deque(zip(start, [li] * len(start))), sum(start), li * len(start)
                new[i] = True
                ok = False                                              # already in the new segment
            elif big and (mode == 'charge x2' or rq < DROP):
                ok = False                                              # held back: a candidate or a dropout
        if ok:
            buf.append((qi, li)); sq += qi; sl += li
            if len(buf) > wmax:
                a, b = buf.popleft(); sq -= a; sl -= b
        if buf:
            last = sq / len(buf)
        pred[i] = last
    return pred, new


assert np.allclose(hybrid(np.r_[np.ones(5), 3 * np.ones(3)], np.r_[np.ones(5), 3 * np.ones(3)], np.zeros(8),
                          'charge+laser', 4, 9, 0.03, 0.03)[0], [1, 1, 1, 1, 1, 3, 3, 3])

df, x, bcm_bad, good = load_run(sys.argv[1], sys.argv[2])
Q = df[TARGET]
q = Q.where(good)
cut = cut_shot(df)
med, jump, stable = level_labels(q, good)
train = df.index < cut
# jitter scales (relative to the level), from the stable shots of the training part only
lab_med = x.lab.rolling(41, center=True, min_periods=10).median()
sig_q = (q / med - 1)[stable & train].std()
sig_l = (x.lab / lab_med - 1)[stable & train].std()
step_med = x.step.rolling(5, min_periods=1).median().values    # trailing, short: must follow real level changes
print(f'{sys.argv[2]}: jitter (train, stable) charge {100 * sig_q:.2f} %, lab meter {100 * sig_l:.2f} %')


def as_pred(p):                     # forecast made after shot i is for shot i+1
    return pd.DataFrame({'shot': df.index + 1, 'h': 1, 'pred': p})


def train_mae(p):
    y = q.shift(-1).values                                               # target: the next shot, if good
    m = train & np.isfinite(y) & np.isfinite(p) & (df.index + 1 < cut)
    return np.mean(np.abs(y[m] - p[m]))


args = (Q.values, x.lab.values, step_med)
grid = {(mode, z, w): train_mae(hybrid(*args, mode, z, w, sig_q, sig_l)[0])
        for mode, z, w in itertools.product(MODES, ZS, WMAXES)}
shots = scored_shots(df)
y = Q.reindex(shots).values
ref = {'9-shot mean (baseline)': rolling_baseline(df).values, '3-shot mean': q.rolling(3, min_periods=1).mean().values,
       '20-shot mean': q.rolling(20, min_periods=1).mean().values}
rows, best_pred = [], {}
for name, p in ref.items():
    rows.append({'model': name, 'train MAE': train_mae(p), 'test MAE': score_forecast(df, as_pred(p)).MAE[1], 'p': p})
for mode in MODES:
    (_, z, w), _ = min(((k, v) for k, v in grid.items() if k[0] == mode), key=lambda kv: kv[1])
    p, new = hybrid(*args, mode, z, w, sig_q, sig_l)
    best_pred[mode] = (p, new)
    rows.append({'model': f'hybrid, {mode} (z={z}, wmax={w})', 'train MAE': grid[(mode, z, w)],
                 'test MAE': score_forecast(df, as_pred(p)).MAE[1], 'p': p, 'changes': int(new.sum()),
                 'test changes': int(new[~train].sum())})
R = pd.DataFrame(rows)
P = {r.model: pd.Series(r.p, df.index).shift(1) for r in R.itertuples()}         # indexed by the predicted shot
R['test dMAE %'] = 100 * (1 - R['test MAE'] / R['test MAE'][0])
st_test, ch_test = (stable & ~train).reindex(shots).values, jump.reindex(shots).values
R['test: stable'] = [np.nanmean(np.abs(y - P[k].reindex(shots).values)[st_test]) for k in R.model]
R['test: after change'] = [np.nanmean(np.abs(y - P[k].reindex(shots).values)[ch_test]) for k in R.model]
for lo, hi, name in ((2, 7, 'run stable ~5.5 nC'), (7, 14, 'run stable ~10 nC')):
    m = stable & (med > lo) & (med < hi)
    R[name] = [np.nanmean(np.abs(q - P[k])[m]) for k in R.model]
pd.set_option('display.width', 250)
print(f'\nnext-shot MAE (nC). test = last 10 % ({len(shots)} scored: {st_test.sum()} stable, {ch_test.sum()} within 20 shots '
      f'after a level change); "run stable" = stable shots of the whole run (train + test)')
print(R.drop(columns='p').round(4).fillna('').to_string(index=False))
print('\ntrain MAE over the grid (rows z, columns wmax):')
for mode in MODES:
    print(f'  {mode}\n' + pd.DataFrame([[grid[(mode, z, w)] for w in WMAXES] for z in ZS], ZS, WMAXES).round(4).to_string())

if len(sys.argv) > 3:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best_mode = R.iloc[3:].sort_values('train MAE').model.iloc[0]
    show = {'9-shot mean (baseline)': 'C0', best_mode: 'C3'}
    ch_shots = shots[np.r_[False, np.diff(ch_test.astype(int)) == 1]]
    fig = plt.figure(figsize=(15, 9))
    zooms = [(s - 15, s + 30) for s in ch_shots[:3]]
    axes = [fig.add_subplot(2, 1, 1)] + [fig.add_subplot(2, len(zooms), len(zooms) + i + 1) for i in range(len(zooms))]
    for a, (lo, hi) in zip(axes, [(shots[0], shots[-1])] + zooms):
        m = (shots >= lo) & (shots <= hi)
        a.plot(shots[m], y[m], '.', color='k', ms=3, label='actual BCM_GUN')
        for k, c in show.items():
            v = P[k].reindex(shots).values
            a.plot(shots[m], v[m], '-', color=c, lw=1, label=f'{k}: test MAE {np.nanmean(np.abs(y - v)):.3f} nC')
        a.set_xlabel('shot'); a.set_ylabel('charge (nC)')
    axes[0].legend(loc='upper left'); axes[0].set_title(f'{sys.argv[2]}: next-shot forecast, test region')
    for a in axes[1:]:
        a.set_title('zoom on a level change')
    fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110)
    print('->', sys.argv[3])

# sensitivity only (not used to choose): test MAE of the 'charge' variant over the same grid
print('\ntest MAE, charge variant, over the grid (rows z, columns wmax) - for sensitivity, not selection:')
print(pd.DataFrame([[score_forecast(df, as_pred(hybrid(*args, 'charge', z, w, sig_q, sig_l)[0])).MAE[1] for w in WMAXES]
                    for z in ZS], ZS, WMAXES).round(4).to_string())
