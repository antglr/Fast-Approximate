# Forecast h shots ahead on one contiguous run with the official split (train on the first 90 % of shots, test on
# the last 10 %), scored by the kickoff notebook's score_forecast against the mean of the last 9 good shots.
# Models: the rolling mean with its window tuned on the training part, and a ridge on causal features
# (past charge, laser meters, beam-loading step) predicting the change from the 9-shot mean.
# Usage: python -I forecast_run.py <tables dir> <run> [<next-shot plot.png> [<stable residuals plot.png>]]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from beam_loading_step import step
from nbfuncs import TARGET, W, cut_shot, dropouts, rolling_baseline, score_forecast, scored_shots
from pulses import align, meter_shift, pulse_ids, read_delay, rf_shift

H = [1, 2, 5, 10, 20, 50]
WINDOWS = [1, 3, 5, 9, 20, 50, 100, 200, 500]
PER_NC = 0.89

p = os.path.join(sys.argv[1], sys.argv[2])
df = pd.read_parquet(p + '.parquet')
d, L = np.load(p + '_rf.npz'), np.load(p + '_laser.npz')
assert np.array_equal(d['shot'], df.index) and np.array_equal(L['shot'], df.index)
t, ch = d['t'], d['chunk']
pid = pulse_ids(t, ch)
delay = read_delay(df['pc_time'], t)
x = pd.DataFrame({'step': step(align(d['PGLA'], pid, rf_shift(d['PGLA_off']))) / PER_NC,
                  'lab': align(L['lab'], pid, meter_shift(delay)) * 1e6,
                  'clex': align(L['clex'], pid, meter_shift(delay)) * 1e6}, index=df.index)
x = x.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction='both')

# shots where the BCM itself is wrong (stuck low while the beam-loading step sees full charge): never train on them
qstep = x.step.rolling(41, center=True, min_periods=10).median()
bcm_bad = (qstep > 1) & (df[TARGET] < 0.5 * qstep)
good = ~dropouts(df) & ~bcm_bad
q = df[TARGET].where(good)
cut = cut_shot(df)
train_origin = df.index < cut
print(f'{sys.argv[2]}: {len(df)} shots, test from shot {cut} ({(df.index >= cut).sum()} shots, '
      f'{len(scored_shots(df))} scored); '
      f'{dropouts(df).sum()} dropouts, {bcm_bad.sum()} BCM-wrong shots')

# --- 1. the rolling mean itself: which window? (chosen on train, also shown on test) ---------------------------
def roll_pred(w, h):
    m = q.rolling(w, min_periods=1).mean()
    return pd.DataFrame({'shot': df.index + h, 'h': h, 'pred': m.values})


def train_mae(pred):
    s = pred.set_index('shot')['pred']
    y = q.reindex(s.index)
    m = y.notna() & (s.index < cut) & (s.index - pred.h.iloc[0] >= df.index[0])
    return np.mean(np.abs(y[m] - s[m]))


print('\n1. rolling mean of the last W good shots: test MAE (nC) per horizon; * = window picked on the train part')
print('   W    ' + ''.join(f'h={h:<6d}' for h in H))
best = {h: min(WINDOWS, key=lambda w: train_mae(roll_pred(w, h))) for h in H}
for w in WINDOWS:
    sc = score_forecast(df, pd.concat([roll_pred(w, h) for h in H]))
    print(f'   {w:<4d} ' + ''.join(f'{sc.MAE[h]:.4f}{"*" if best[h] == w else " "} ' for h in H))

# --- 2. ridge on causal features -----------------------------------------------------------------------------------
base = rolling_baseline(df)
F = pd.DataFrame({f'q{w}': q.rolling(w, min_periods=1).mean() - base for w in (1, 3, 20, 50, 200)}, index=df.index)
F['trend'] = q.rolling(20, min_periods=5).apply(lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True)
for k in ('lab', 'clex', 'step'):
    for w in (1, 9, 50):
        F[f'{k}{w}'] = x[k].rolling(w, min_periods=1).mean() - x[k].rolling(200, min_periods=10).mean()
F = F.fillna(0)

preds, coefs = [], {}
for h in H:
    y = (q.shift(-h) - base).values                       # change from the 9-shot mean, h shots on
    m = train_origin & np.isfinite(y) & (df.index + h < cut)
    r = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(F[m], y[m])
    preds.append(pd.DataFrame({'shot': df.index + h, 'h': h, 'pred': base.values + r.predict(F)}))
    coefs[h] = pd.Series(r.coef_ * F[m].std().values, F.columns)
sc = score_forecast(df, pd.concat(preds))
sc['dMAE %'] = 100 * sc.dMAE / sc.baseline_MAE
best_roll = score_forecast(df, pd.concat([roll_pred(best[h], h) for h in H])).MAE
sc['tuned W'] = [best[h] for h in sc.index]
sc['tuned MAE'] = best_roll
print('\n2. ridge on past charge + laser + step vs the 9-shot mean (test region)')
print(sc.round(4).to_string())
print('\n   largest standardised ridge coefficients (nC per std):')
for h in (1, 10, 50):
    print(f'   h={h:<3d}', '  '.join(f'{k} {v:+.3f}' for k, v in coefs[h].abs().nlargest(4).index.map(lambda k: (k, coefs[h][k]))))

r = lambda a, b: pd.concat([a, b], axis=1).dropna().corr().iloc[0, 1]
dev = q - base
print(f'\nautocorrelation of (charge - 9-shot mean): lag1 {r(dev, dev.shift(1)):+.3f}, lag5 {r(dev, dev.shift(5)):+.3f}, '
      f'lag20 {r(dev, dev.shift(20)):+.3f}')

# --- 3. where do the errors come from? steady shots vs the 20 shots after a level change -------------------------
# (labelled with the future, for analysis only: a shot is 'after a change' if the 41-shot centred median of the
# charge moved by > 20 % within the previous 20 shots)
med = q.rolling(41, center=True, min_periods=10).median()
jump = (med / med.shift(20) - 1).abs().rolling(20, min_periods=1).max() > 0.2
shots = scored_shots(df)
print(f'\n3. test MAE (nC) split by regime: {(~jump[shots]).sum()} steady shots, {jump[shots].sum()} within 20 shots of a level change')
P = pd.concat(preds).set_index(['h', 'shot'])['pred']
for h in (1, 5, 10):
    y = df.loc[shots, TARGET]
    cands = {'9-shot mean': base.reindex(shots - h).values, f'mean W={best[h]}': q.rolling(best[h], min_periods=1).mean().reindex(shots - h).values,
             'ridge': P[h].reindex(shots).values}
    print(f'   h={h:<3d}' + ''.join(f'  {k}: steady {np.nanmean(np.abs(y - v)[~jump[shots].values]):.4f} change {np.nanmean(np.abs(y - v)[jump[shots].values]):.3f}' for k, v in cands.items()))

# --- 4. plot: predicted vs actual next shot (h=1) over the test region --------------------------------------------
if len(sys.argv) > 3:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y = df.loc[shots, TARGET]
    cands = {'9-shot mean (baseline)': (base.reindex(shots - 1).values, 'C0'),
             f'{best[1]}-shot mean': (q.rolling(best[1], min_periods=1).mean().reindex(shots - 1).values, 'C2'),
             'ridge (charge + laser + step)': (P[1].reindex(shots).values, 'C3')}
    first_jump = shots[jump[shots].values][0] if jump[shots].any() else shots[len(shots) // 2]
    fig = plt.figure(figsize=(15, 9))
    a0, a1 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 2, 3)
    zoom = (shots >= first_jump - 40) & (shots <= first_jump + 40)
    for a, m in ((a0, slice(None)), (a1, zoom)):
        a.plot(shots[m], y.values[m], '.', color='k', ms=3, label='actual BCM_GUN')
        for k, (v, c) in cands.items():
            a.plot(shots[m], v[m], '-', color=c, lw=1, label=f'{k}: MAE {np.nanmean(np.abs(y.values - v)):.3f} nC')
        a.set_xlabel('shot'); a.set_ylabel('charge (nC)')
    a0.legend(loc='upper left'); a0.set_title(f'{sys.argv[2]}: next-shot forecast over the test region (last 10 %)')
    a1.set_title('zoom on the first level change')
    a2 = fig.add_subplot(2, 2, 4)
    for k, (v, c) in cands.items():
        a2.plot(y.values, v, '.', color=c, ms=2, alpha=0.5, label=k)
    lim = [np.nanmin(y), np.nanmax(y)]
    a2.plot(lim, lim, 'k--', lw=0.8); a2.set_xlabel('actual next-shot charge (nC)'); a2.set_ylabel('predicted (nC)')
    a2.set_title('predicted vs actual'); a2.legend(markerscale=6)
    fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110)
    print('->', sys.argv[3])

# --- 5. next-shot residuals (measured - predicted) on the stable shots of the whole run ----------------------------
# Stable: a good shot, not within 20 shots of a level change, and the local level (41-shot median) moves < 10 % over
# ~100 shots, which removes the laser scan; and no bad shot in the baseline window. The means are fitted on nothing, so they are scored everywhere. The ridge
# is out-of-fold over 10 contiguous blocks, so no shot is predicted by a model trained on it.
if len(sys.argv) > 4:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats
    span = med.rolling(101, center=True, min_periods=20)
    # also no BCM-wrong shot or dropout among the last W+1 shots: the official baseline keeps averaging stuck readings
    recent_bad = (~good).astype(float).rolling(W + 1, min_periods=1).max().astype(bool)
    stable = good & ~jump & ~recent_bad & ((span.max() - span.min()) / med < 0.10) & q.notna()
    y1 = (q.shift(-1) - base).values
    block = np.arange(len(df)) * 10 // len(df)
    oof = np.full(len(df), np.nan)
    for b in range(10):
        tr = np.isfinite(y1) & (block != b)
        oof[block == b] = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(F[tr], y1[tr]).predict(F[block == b])
    pred = {'9-shot mean (baseline)': base.shift(1),                      # value at origin n, for shot n+1
            f'{best[1]}-shot mean': q.rolling(best[1], min_periods=1).mean().shift(1),
            'ridge (out-of-fold)': (base + oof).shift(1)}
    lvl = med[stable]
    D = pd.DataFrame({k: (q - v)[stable] for k, v in pred.items()}).dropna()
    lvl = lvl[D.index]
    print(f'\n5. next-shot residual = measured - predicted, {len(D)} stable shots of the whole run '
          f'(skew SE {np.sqrt(6 / len(D)):.2f}, excess-kurtosis SE {np.sqrt(24 / len(D)):.2f}; Gaussian: both 0)')
    bands = pd.cut(lvl, [0, 2, 7, 14, 99], labels=['~1 nC', '~5.5 nC', '~10 nC', '~19 nC'])
    rows = []
    for k in D:
        for name, sel in [('all, nC', slice(None))] + [(f'{b}', (bands == b).values) for b in bands.cat.categories]:
            d = D[k][sel]
            if len(d) > 30:
                rows.append({'model': k, 'shots': name, 'n': len(d), 'mean': d.mean(), 'std': d.std(), 'MAE': d.abs().mean(),
                             'rel std %': 100 * (d / lvl[d.index]).std(), 'skew': stats.skew(d), 'ex.kurt': stats.kurtosis(d)})
        r = D[k] / lvl
        rows.append({'model': k, 'shots': 'all, relative', 'n': len(r), 'mean': np.nan, 'std': np.nan, 'MAE': np.nan,
                     'rel std %': 100 * r.std(), 'skew': stats.skew(r), 'ex.kurt': stats.kurtosis(r)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3)
    show = ['9-shot mean (baseline)', 'ridge (out-of-fold)']
    for i, k in enumerate(show):
        a = fig.add_subplot(gs[i, :])
        sc_ = a.scatter(D.index, D[k], c=lvl, s=2, cmap='viridis', vmin=0, vmax=20)
        sd = D[k].rolling(101, center=True, min_periods=20).std().reindex(df.index)      # NaN in the gaps breaks the line
        a.plot(df.index, sd, color='r', lw=1, label='rolling std (101 stable shots)')
        a.plot(df.index, -sd, color='r', lw=1)
        a.axhline(0, color='k', lw=0.5); a.set_ylabel('measured - predicted (nC)')
        a.set_title(f'{k}: next-shot residual on stable shots (gaps = scan, level changes, BCM fault, dropouts)')
        a.legend(loc='upper left'); a.set_ylim(-3, 3)
        if i == 1:
            a.set_xlabel('shot')
    fig.colorbar(sc_, ax=fig.axes, label='local charge level (nC)', shrink=0.5, location='right')
    ah, aq, ar = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])
    bins = np.linspace(-3, 3, 121)
    for k, c in zip(D, ('C0', 'C2', 'C3')):
        d = D[k]
        ah.hist(d, bins, histtype='step', color=c, label=f'{k}: std {d.std():.3f}, skew {stats.skew(d):+.2f}, ex.kurt {stats.kurtosis(d):+.1f}')
        ah.plot(bins, len(d) * (bins[1] - bins[0]) * stats.norm.pdf(bins, d.mean(), d.std()), ':', color=c)
        (osm, osr), _ = stats.probplot((d - d.mean()) / d.std())
        aq.plot(osm, osr, '.', ms=2, color=c, label=k)
        r = 100 * d / lvl
        rb = np.linspace(-25, 25, 101)
        ar.hist(r, rb, histtype='step', color=c, label=f'std {r.std():.1f} %, skew {stats.skew(r):+.2f}, ex.kurt {stats.kurtosis(r):+.1f}')
        ar.plot(rb, len(r) * (rb[1] - rb[0]) * stats.norm.pdf(rb, r.mean(), r.std()), ':', color=c)
    ah.set_yscale('log'); ah.set_ylim(0.5, None); ah.set_xlabel('measured - predicted (nC)')
    ah.set_title('distribution, nC (dotted: Gaussian, same std)'); ah.legend(fontsize=7, loc='lower center')
    aq.plot([-5, 5], [-5, 5], 'k--', lw=0.8); aq.set_xlim(-4.5, 4.5); aq.set_ylim(-8, 8)
    aq.set_xlabel('Gaussian quantile'); aq.set_ylabel('standardised residual quantile'); aq.set_title('Q-Q vs Gaussian')
    aq.legend(markerscale=6, fontsize=7)
    ar.set_yscale('log'); ar.set_ylim(0.5, None); ar.set_xlabel('(measured - predicted) / charge level (%)')
    ar.set_title('relative to the charge level'); ar.legend(fontsize=7, loc='lower center')
    fig.savefig(sys.argv[4], dpi=110, bbox_inches='tight')
    print('->', sys.argv[4])
