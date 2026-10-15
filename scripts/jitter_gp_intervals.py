# Prediction intervals of jitter_gp.py's GPs, split by whether the shot has a camera frame. Without one the model has
# only the laser and errs ~3x more (jitter_maxloss.py), which the fixed noise term (the BCM noise alone) does not know.
# Adds a camera-missing variance per regime and GP: mean square residual minus mean predictive variance on the
# camera-missing early-stopping rows (never the test region), on camera-missing shots only. Coverage of the 90 % and
# 99 % intervals and the 90 % half-width in pC on the test regions, before and after.
# Usage: python -I jitter_gp_intervals.py <tables dir> <jitter_gp out dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from jitter_sequence import trailing
from same_shot_regression import REGIMES, stitched

GPS = ['GP on best residual, fixed noise', 'GP on best residual, learned noise']


def main():
    tables, gdir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    F = stitched(tables, runs)
    G = pd.read_parquet(os.path.join(gdir, 'gp_predictions.parquet'))
    assert (G.run.values == F.run.values).all() and (G.shot.values == F.shot.values).all()
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    ok = level.notna()
    fit_rows = F.train & ok
    k = fit_rows.groupby(F.seg).cumsum()
    va = fit_rows & (k > 0.9 * k.groupby(F.seg).transform('max'))
    nocam = ~(F.BTV_intensity > 0)
    test = ~F.train & ok
    rows = []
    for n in GPS:
        r2, var = (yf - G[f'{n}|mu']) ** 2, G[f'{n}|var']
        extra = pd.Series(0.0, F.index)
        for rname in REGIMES:
            m = va & nocam & (F.regime == rname)
            add = max(0.0, r2[m].mean() - var[m].mean()) if m.sum() >= 20 else 0.0
            extra[(F.regime == rname) & nocam] = add
            print(f'{n}, {rname}: camera-missing extra std {100 * np.sqrt(add):.2f} % ({m.sum()} early-stopping rows)')
        for tag, v in (('', var), (' + camera-missing term', var + extra)):
            z = np.sqrt(r2 / v)
            for s, g in F[test].groupby('seg'):
                for cam, i in (('camera', g.index[~nocam[g.index]]), ('no camera', g.index[nocam[g.index]]), ('all', g.index)):
                    if len(i):
                        rows.append({'model': n + tag, 'seg': s, 'shots': cam, 'n': len(i), 'cover 90 %': 100 * (z[i] < 1.645).mean(),
                                     'cover 99 %': 100 * (z[i] < 2.576).mean(), '90 % half-width pC': (1645 * np.sqrt(v[i]) * np.exp(level[i])).mean()})
    D = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    for col in ('cover 90 %', 'cover 99 %', '90 % half-width pC'):
        print(f'\n{col}, test regions')
        print(D.pivot_table(index=['model', 'shots'], columns='seg', values=col).round(1).to_string())


if __name__ == '__main__':
    main()
