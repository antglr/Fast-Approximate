# Are the forecasters' gains real, and how sure is each forecast? Reads the next-shot forecasts gru_forecast.py saves
# (gru_forecasts.npz: full-run arrays, the forecast made after shot i for shot i+1).
# 1. Moving-block bootstrap of the test-region MAE ratio against the 9-shot mean and against the hybrid. Blocks keep the
#    error autocorrelation (a level change gives a run of large errors).
# 2. 90 % prediction intervals for the combined forecaster: forecast x (1 + [q05, q95] of the relative residual).
#    'rolling': quantiles of the last CAL good residuals of the run, each used only once its shot is measured (causal,
#    so it follows the jitter when the charge regime changes); 'fixed': quantiles of the run's training part. Forecasts
#    made fewer than YOUNG shots into a segment use quantiles of such forecasts, pooled over all runs' training parts.
# Usage: python -I forecast_uncertainty.py <tables dir> <gru_forecasts.npz> <out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from combined_forecast import prepare
from nbfuncs import TARGET, scored_shots

BLOCK, NBOOT = 50, 4000
CAL, YOUNG, LO, HI = 500, 5, 0.05, 0.95
BASE, HYB, SHOW = '9-shot mean (baseline)', 'hybrid (z=6, wmax=20)', 'combined, ridge on pooled runs'


def boot_gain(e_ref, e, rng):
    # % MAE reduction against the reference, with a 95 % moving-block bootstrap interval
    n = len(e)
    starts = rng.integers(0, n - BLOCK + 1, (NBOOT, -(-n // BLOCK)))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :n]
    g = 100 * (1 - e[idx].mean(1) / e_ref[idx].mean(1))
    return 100 * (1 - e.mean() / e_ref.mean()), *np.percentile(g, [2.5, 97.5])


def main():
    tables, npz, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    F = np.load(npz)
    rng = np.random.default_rng(0)
    pd.set_option('display.width', 250)

    rows = []
    for run in runs:
        R = {k.split('|', 1)[1]: F[k] for k in F.files if k.startswith(run + '|')}
        # no good shot in the last 9 rows (after a beam trip): hold the last mean, so every forecaster is scored on the
        # same shots (the official score drops these few shots for the baseline only)
        R[BASE] = pd.Series(R[BASE]).ffill().values
        df = prepare(tables, run)['df']
        shots = scored_shots(df)
        y = df.loc[shots, TARGET].values
        E = {k: np.abs(y - pd.Series(v, df.index).shift(1).reindex(shots).values) for k, v in R.items()}
        ok = np.all([np.isfinite(e) for e in E.values()], 0)
        for k, e in E.items():
            row = {'run': run, 'forecast': k, 'test MAE': e[ok].mean()}
            for ref in (BASE, HYB):
                if k != ref and not (ref == HYB and k == BASE):
                    g, lo, hi = boot_gain(E[ref][ok], e[ok], rng)
                    row[f'vs {ref[:11]}: gain %'] = f'{g:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]' + (' *' if lo > 0 or hi < 0 else '')
            rows.append(row)
    print(f'next-shot test MAE (nC) and its % reduction against the 9-shot mean and the hybrid, with 95 % moving-block '
          f'bootstrap intervals ({BLOCK}-shot blocks); * = the interval excludes 0')
    print(pd.DataFrame(rows).round(4).fillna('').to_string(index=False))

    # prediction intervals
    P = {}
    for run in runs:
        r = prepare(tables, run)
        df, seg = r['df'], r['seg']
        f = pd.Series(F[f'{run}|{SHOW}'], df.index)
        res = r['q'].shift(-1) / f - 1                      # relative residual of the forecast made after shot i
        young = (seg.groupby(seg).cumcount() < YOUNG).values
        P[run] = dict(r=r, f=f, res=res, young=young)
    pool = pd.concat([P[u]['res'][P[u]['young'] & P[u]['r']['train']] for u in runs]).dropna()
    qy = pool.quantile([LO, HI]).values
    print(f'\nprediction intervals for "{SHOW}", nominal {100 * (HI - LO):.0f} %; forecasts made < {YOUNG} shots after a level '
          f'change use the pooled training quantiles {100 * qy[0]:+.1f} / {100 * qy[1]:+.1f} % ({len(pool)} forecasts)')
    rows = []
    for run, d in P.items():
        r, f, res, young = d['r'], d['f'], d['res'], d['young']
        df = r['df']
        mature = res.where(~young)
        valid = mature.dropna()
        # residual i is known once shot i+1 is measured, so it may set the interval of the forecast made after shot i+1
        roll = pd.DataFrame({q: valid.rolling(CAL, min_periods=50).quantile(q) for q in (LO, HI)})
        roll = roll.reindex(df.index).ffill().shift(1)
        fixed = mature[r['train']].quantile([LO, HI]).values
        shots = scored_shots(df)
        y = df.loc[shots, TARGET].values
        fp = f.shift(1).reindex(shots).values
        yg = young[np.searchsorted(df.index, shots) - 1]
        st, ch = (r['stable'] & ~r['train']).reindex(shots).values, r['jump'].reindex(shots).values
        for name, q in (('rolling', roll.shift(1).reindex(shots).values), ('fixed', np.tile(fixed, (len(shots), 1)))):
            q = np.where(yg[:, None], qy, q)
            lo, hi = fp * (1 + q[:, 0]), fp * (1 + q[:, 1])
            inside, w = (y >= lo) & (y <= hi), 100 * (hi - lo) / fp
            row = {'run': run, 'calibration': name, 'coverage %': 100 * np.nanmean(inside), 'mean width % of level': np.nanmean(w),
                   'coverage, stable %': 100 * inside[st].mean(), 'width, stable %': np.nanmean(w[st])}
            row['coverage, after change %'] = 100 * inside[ch].mean() if ch.any() else np.nan
            rows.append(row)
            if name == 'rolling':
                d['band'] = shots, y, fp, lo, hi
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for run, d in P.items():
        shots, y, fp, lo, hi = d['band']
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.fill_between(shots, lo, hi, color='C3', alpha=0.25, lw=0, label=f'{100 * (HI - LO):.0f} % interval (rolling calibration)')
        ax.plot(shots, fp, '-', color='C3', lw=0.8, label=SHOW)
        out_ = (y < lo) | (y > hi)
        ax.plot(shots[~out_], y[~out_], '.', color='k', ms=2, label='actual BCM_GUN')
        ax.plot(shots[out_], y[out_], '.', color='C1', ms=4, label=f'outside the interval ({100 * np.nanmean(out_):.1f} %)')
        ax.set_xlabel('shot'); ax.set_ylabel('charge (nC)'); ax.legend(loc='upper left', fontsize=8)
        ax.set_title(f'{run}: next-shot forecast with prediction interval, test region')
        fig.tight_layout(); fig.savefig(os.path.join(out, f'interval_{run}.png'), dpi=110)
        print('->', os.path.join(out, f'interval_{run}.png'))


if __name__ == '__main__':
    main()
