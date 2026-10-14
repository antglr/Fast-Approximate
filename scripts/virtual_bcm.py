# Same-shot virtual BCM: the charge of shot n from that shot's own laser meters, laser scope and beam-loading step,
# plus the BCM history up to shot n-1 (the hybrid level and the combined forecaster's features). Not a forecast: it
# needs shot n's instruments, so it is a virtual diagnostic (e.g. while the BCM is stuck or reads wrong).
# Model: log(Q[n] / hybrid level after n-1) = ridge on
#   history   the combined forecaster's features (as in combined_forecast.py)
#   same shot log(reading[n] / its segment mean over the last 20 and 200 shots up to n-1) for lab, clex, scope
# Not the beam-loading step: its noise is ~fixed in nC, so relative to a 0.3 nC level it is 8x noisier than at 5 nC
# and a model trained on high-charge runs over-trusts it (run 3, other runs only: 0.0511 nC with it, 0.0472 without).
# Log ratios, because photoemission scales with the laser: a laser-driven level change is a large ratio, not an outlier.
# One ridge per charge regime (hybrid level above / below LEVEL_SPLIT, known before the shot): the laser->charge relation
# differs at 0.1-0.3 nC, and one pooled model over 5 runs made runs 1-2 worse (0.302 / 0.125 vs 0.295 / 0.119 nC split).
# Missing readings stay missing (feature 0): ~10 % of meter and 13-36 % of step readings have no pulse of their own.
# Trained on all good shots of the first 90 % (pooled runs, and the other runs only), scored on the last 10 % with
# score_forecast against the BCM, like the forecasts.
# Also: "BCM lost at the cut" - no BCM at all in the test region, calibration frozen at the end of the training part.
# Usage: python -I virtual_bcm.py <tables dir> <out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from combined_forecast import ALPHAS, WMAX, Z, prepare, score, seg_mean
from nbfuncs import TARGET, cut_shot, scored_shots
from pulses import align, pulse_ids, stamp_shift

LEVEL_SPLIT = 1.0        # nC
SAME_SHOT = {'lab only': ['lab'], 'lab, clex, scope': ['lab', 'clex', 'scope']}


def scope_amp(tables, run):
    # laser scope: matched filter on the mean pulse train (notebook 03), NaN where frozen, not acquired or saturated
    L = np.load(os.path.join(tables, run + '_laser.npz'))
    sc, t = L['scope'].astype(float), L['scope_times'][0]
    clean = sc - sc[:, t < 5].mean(1, keepdims=True)
    tmpl = np.nanmean(clean, 0) * ((t > 5.6) & (t < 39.8))
    amp = clean @ tmpl / (tmpl @ tmpl)
    bad = np.r_[False, np.all(sc[1:] == sc[:-1], 1)] | ~L['scope_acquired'].astype(bool) | L['scope_saturated'].astype(bool)
    amp[bad] = np.nan
    return align(amp, pulse_ids(L['t'], L['chunk']), stamp_shift(L['scope_off']))   # stamp alignment: same-shot r peaks


def same_shot(r, names):
    # row i holds shot i+1's readings against levels known after shot i (the frame of r['F'] and r['p'])
    seg, G = r['seg'], pd.DataFrame(index=r['p'].index)
    for k in names:
        v = r['inst'][k]
        for w in (WMAX, 200):
            with np.errstate(invalid='ignore', divide='ignore'):
                G[f'{k}/{w}'] = np.log(v.shift(-1) / seg_mean(v, seg, w))
    return G.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-3, 3)


def design(r, names):
    return pd.concat([r['F'], same_shot(r, names)], axis=1)


def main():
    tables, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    R = {}
    for run in runs:
        r = prepare(tables, run)
        # only this pulse's own readings (NaN where missing, feature 0): a filled value is an older pulse's
        x = r['x']
        r['inst'] = pd.DataFrame({k: x[k].where(x[k + '_ok']) for k in ('lab', 'clex', 'step')}, r['df'].index)
        r['inst']['scope'] = scope_amp(tables, run)
        r['inst'].loc[r['inst'].scope <= 0, 'scope'] = np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            r['ly'] = np.log(r['q'].shift(-1).values / r['p'].values)                 # target: log(Q[i+1] / level)
        r['vfit'] = r['train'] & (r['df'].index + 1 < cut_shot(r['df'])) & np.isfinite(r['ly'])
        # self-check: row j holds shot j+1's lab against the mean of up to WMAX lab readings of row j's segment
        j = int(np.flatnonzero(r['vfit'])[r['vfit'].sum() // 2])
        lab, seg = r['inst'].lab, r['seg']
        assert np.isclose(same_shot(r, ['lab'])[f'lab/{WMAX}'].iloc[j],
                          np.log(lab.iloc[j + 1] / lab.iloc[:j + 1][seg.iloc[:j + 1] == seg.iloc[j]].tail(WMAX).mean()))
        R[run] = r
    srcs = {'pooled runs': lambda run: runs, 'other runs only': lambda run: [o for o in runs if o != run]}
    if len(runs) == 1:
        del srcs['other runs only']
    rows, coefs = [], {}
    for run, r in R.items():
        shots = scored_shots(r['df'])
        yq = r['df'].loc[shots, TARGET].values
        st, ch = (r['stable'] & ~r['train']).reindex(shots).values, r['jump'].reindex(shots).values
        cands = {f'hybrid (z={Z}, wmax={WMAX}), forecast': r['p'].values}
        for name, src in srcs.items():
            use = src(run)
            for sname, names in SAME_SHOT.items():
                lp, D = np.zeros(len(r['p'])), design(r, names)
                for hi in (False, True):
                    sel = {u: R[u]['vfit'] & ((R[u]['p'] > LEVEL_SPLIT) == hi) for u in use}
                    at = (r['p'] > LEVEL_SPLIT).values == hi
                    if not sum(s.sum() for s in sel.values()) or not at.any():
                        continue                           # no training shots in this regime: stay the hybrid
                    X = pd.concat([design(R[u], names)[sel[u]] for u in use])
                    m = RidgeCV(alphas=ALPHAS).fit(X, np.concatenate([R[u]['ly'][sel[u]] for u in use]))
                    if name == 'pooled runs':
                        coefs[f'{sname}, {"above" if hi else "below"} {LEVEL_SPLIT:g} nC'] = pd.Series(m.coef_ * X.std().values * 100, X.columns)
                    lp[at] = m.predict(D[at])
                cands[f'virtual BCM, {sname}, {name}'] = r['p'].values * np.exp(lp)
        # BCM lost at the cut: no BCM in the test region, everything frozen at the last 200 good training shots
        cut = cut_shot(r['df'])
        cal = r['q'].notna() & (r['df'].index < cut) & (r['df'].index >= cut - 400)
        cal &= cal.cumsum() > cal.sum() - 200
        q0, lab0 = r['q'][cal].mean(), r['inst'].lab[cal].mean()
        step_off = (r['inst'].step - r['q'])[cal].mean()
        cands['BCM lost: laser x frozen ratio'] = (q0 * r['inst'].lab / lab0).shift(-1).values
        cands['BCM lost: step - frozen offset'] = (r['inst'].step - step_off).shift(-1).values
        for name, v in cands.items():
            e = np.abs(yq - pd.Series(v, r['df'].index).shift(1).reindex(shots).values)
            rows.append({'run': run, 'estimate': name, 'test MAE': score(r, v), 'stable': np.nanmean(e[st]),
                         'after change': np.nanmean(e[ch]) if ch.any() else np.nan})
        r['cands'] = cands
    T = pd.DataFrame(rows)
    T['vs hybrid %'] = T.groupby('run')['test MAE'].transform(lambda s: 100 * (s / s.iloc[0] - 1))
    pd.set_option('display.width', 250)
    print('MAE (nC) against the BCM on each run\'s test region (last 10 %); the hybrid is the next-shot forecast, for scale')
    print(T.round(4).fillna('').to_string(index=False))
    print('\npooled ridge: change of the estimate per std of each feature, % (largest 8)')
    for k, c in coefs.items():
        print(f'  {k}: ' + '  '.join(f'{n} {v:+.2f}' for n, v in c.reindex(c.abs().sort_values(ascending=False).index)[:8].items()))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best = 'virtual BCM, lab, clex, scope, pooled runs'
    for run, r in R.items():
        shots = scored_shots(r['df'])
        yq = r['df'].loc[shots, TARGET].values
        show = {f'hybrid (z={Z}, wmax={WMAX}), forecast': 'C2', best: 'C3'}
        V = {k: pd.Series(r['cands'][k], r['df'].index).shift(1).reindex(shots).values for k in show}
        ch = r['jump'].reindex(shots).values
        starts = shots[np.r_[False, np.diff(ch.astype(int)) == 1]][:3]
        fig = plt.figure(figsize=(15, 9))
        axes = [fig.add_subplot(2, 1, 1)] + [fig.add_subplot(2, len(starts), len(starts) + i + 1) for i in range(len(starts))]
        for a, (lo, hi) in zip(axes, [(shots[0], shots[-1])] + [(s - 15, s + 30) for s in starts]):
            m = (shots >= lo) & (shots <= hi)
            a.plot(shots[m], yq[m], '.', color='k', ms=3, label='BCM_GUN')
            for k, c in show.items():
                a.plot(shots[m], V[k][m], '-', color=c, lw=1, label=f'{k}: MAE {np.nanmean(np.abs(yq - V[k])):.4f} nC')
            a.set_xlabel('shot'); a.set_ylabel('charge (nC)')
        axes[0].legend(loc='upper left', fontsize=8); axes[0].set_title(f'{run}: same-shot virtual BCM, test region')
        for a in axes[1:]:
            a.set_title('zoom on a level change')
        fig.tight_layout(); fig.savefig(os.path.join(out, f'virtual_bcm_{run}.png'), dpi=110)
        print('->', os.path.join(out, f'virtual_bcm_{run}.png'))


if __name__ == '__main__':
    main()
