# Decomposing the target by timescale: Welch power spectra (per shot, one-sided) of jitter_sequence.py's target and of
# the residuals of its current-shot ridge and best model, on each test region, against the BCM noise floor (white, at
# the level of bcm_noise_model.py's sigma_n(Q) = sqrt(A^2 + (B Q)^2) for the segment's mean charge). Lines would mean a
# periodic part worth modelling on its own; a residual above the floor in some band is what is left to predict there.
# Prints the variance per band of period (in shots) for the target and the residuals, and the floor's share.
# Rows are the stitched good shots, so the few gaps are ignored.
# Usage: python -I jitter_spectrum.py <tables dir> <jitter_sequence out dir> <out png> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.signal import welch
from jitter_gp import A, B
from jitter_sequence import trailing
from same_shot_regression import stitched

MODELS = {'ridge residual': 'ridge, current shot', 'best residual': 'ridge, + past BCM + GRU, L=64, past BCM'}
BANDS = [(2, 4), (4, 10), (10, 50), (50, np.inf)]          # period in shots (1 shot = 1.2 s)


def main():
    tables, pdir, png, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    F = stitched(tables, runs)
    P = pd.read_parquet(os.path.join(pdir, 'jitter_predictions.parquet'))
    assert (P.run.values == F.run.values).all() and (P.shot.values == F.shot.values).all()
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    test = ~F.train & level.notna()
    lost = yf < np.log(0.5)                                  # BCM-lost shots would dominate the spectra: set to 0
    # self-check: white noise of variance v has a flat one-sided density 2 v, and the band variances add up to v
    f, s = welch(np.random.default_rng(0).normal(0, 2, 2 ** 16), nperseg=128)
    assert abs(s[1:-1].mean() - 8) < 0.2 and abs(np.trapezoid(s, f) - 4) < 0.1

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    segs = [g for g in sorted(set(F.seg[test])) if (test & (F.seg == g)).sum() >= 200]
    fig, axes = plt.subplots(2, (len(segs) + 1) // 2, figsize=(16, 7))
    rows = []
    for a, g in zip(axes.flat, segs):
        i = F.index[test & (F.seg == g)]
        q = 1000 * F.Q[i].mean()
        floor = (A ** 2 + (B * q) ** 2) / q ** 2                 # noise variance, log units
        series = {'target': yf[i].where(~lost[i], 0), **{k: (yf[i] - P[m][i]).where(~lost[i], 0) for k, m in MODELS.items()}}
        for (name, v), c in zip(series.items(), ('0.5', 'C0', 'C3')):
            f, s = welch(100 * v.values, nperseg=min(128, len(i) // 2))
            a.semilogy(f[1:], s[1:], color=c, label=name)
            per = 1 / np.maximum(f, 1e-12)
            rows.append({'seg': g, 'series': name, **{f'{lo}-{hi} shots %²': np.trapezoid(np.where((per >= lo) & (per < hi), s, 0), f) for lo, hi in BANDS}})
        rows.append({'seg': g, 'series': 'BCM noise floor', **{f'{lo}-{hi} shots %²': 1e4 * floor * 2 * (1 / lo - (0 if hi == np.inf else 1 / hi)) for lo, hi in BANDS}})
        a.axhline(2e4 * floor, color='k', ls='--', label='BCM noise floor (white)')
        a.set_title(f'segment {g}, {F.Q[i].mean():.2f} nC, {len(i)} test shots', fontsize=9)
        a.set_xlabel('frequency (1 / shots)'); a.set_ylabel('%² per (1 / shot)'); a.legend(fontsize=7)
    for a in axes.flat[len(segs):]:
        a.axis('off')
    fig.suptitle('test regions: power spectra of the BCM jitter and the model residuals', fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=100)
    print('->', png)
    pd.set_option('display.width', 250)
    print('variance (%² of the trailing level) per band of period; the white floor splits as 2 sigma² x bandwidth')
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
