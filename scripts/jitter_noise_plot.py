# The noise distribution against the jitter models, per test region: histograms of the BCM jitter (jitter_sequence.py's
# target) and of a model's residual, with a Gaussian of the BCM's own noise std from bcm_noise_floor.py's three-cornered
# hat (lab as the third reading, test region, spikes removed). The noise is only known as a width, not shot by shot, so
# it is drawn as a Gaussian. A residual as narrow as the Gaussian is at the floor. Is the noise Gaussian? Where the
# residual is at the floor it is mostly noise, so its shape is checked (log y, to see the tails): skewness, excess
# kurtosis, share beyond 4 sigma (Gaussian 0.006 %), lag-1 autocorrelation (white noise 0). Model-free check of the same:
# d = BCM - k camera, k = cov(BCM, lab) / cov(camera, lab), which cancels the charge and leaves the two noises.
# Usage: python -I jitter_noise_plot.py <tables dir> <jitter_sequence out dir> <out png> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from bcm_noise_floor import hat
from jitter_sequence import trailing
from same_shot_regression import stitched

MODEL = 'ridge, + past BCM + GRU, L=64, past BCM'


def main():
    tables, pdir, png, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    F = stitched(tables, runs)
    P = pd.read_parquet(os.path.join(pdir, 'jitter_predictions.parquet'))
    assert (P.run.values == F.run.values).all() and (P.shot.values == F.shot.values).all()
    fast = lambda v: np.log(v.where(v > 0)) - trailing(np.log(v.where(v > 0)), F.seg)
    S = pd.DataFrame({'BCM': fast(F.Q), 'camera': fast(F.BTV_intensity), 'lab': fast(F.lab)})
    dev = S - S.groupby(F.seg).transform('median')                         # spikes as bcm_noise_floor.py, on these three
    spike = (dev.abs() > 5 * 1.4826 * dev.abs().groupby(F.seg).transform('median')).any(axis=1)
    test = ~F.train & S.BCM.notna()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    segs = [s for s in sorted(set(F.seg[test])) if (test & (F.seg == s)).sum() >= 200]
    fig, axes = plt.subplots(2, (len(segs) + 1) // 2, figsize=(16, 7))
    for a, s in zip(axes.flat, segs):
        i = test & (F.seg == s)
        sig = hat(S[i & ~spike].dropna().values)[0]
        y, res = 100 * S.BCM[i], 100 * (S.BCM[i] - P[MODEL][i])
        T = S[i].dropna()
        d = 100 * (T.BCM - np.cov(T.BCM, T.lab)[0, 1] / np.cov(T.camera, T.lab)[0, 1] * T.camera)
        for name, v in (('model residual', res), ('BCM - k camera', d)):
            rows.append({'seg': s, 'series': name, 'rms %': np.sqrt((v ** 2).mean()), 'BCM noise %': sig, 'skew': stats.skew(v),
                         'excess kurtosis': stats.kurtosis(v), 'beyond 4 sigma %': 100 * (np.abs(v - v.mean()) > 4 * v.std()).mean(),
                         'lag-1 autocorr': v.autocorr(1)})
        lim = 4 * y.std()
        bins = np.linspace(-lim, lim, 61)
        a.hist(y, bins, density=True, color='0.75', label=f'BCM jitter, rms {np.sqrt((y ** 2).mean()):.2f} %')
        a.hist(res, bins, density=True, histtype='step', color='C3', lw=1.5, label=f'model residual, rms {np.sqrt((res ** 2).mean()):.2f} %')
        x = np.linspace(-lim, lim, 400)
        a.plot(x, np.exp(-x ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi)), 'k--', lw=1.2, label=f'BCM noise (3-cornered hat), {sig:.2f} %')
        a.set_title(f'segment {s}, {F.regime[i].iloc[0]}, {F.Q[i].mean():.2f} nC, {i.sum()} shots', fontsize=9)
        a.set_yscale('log'); a.set_ylim(0.5 / (i.sum() * (bins[1] - bins[0])), 3 / (sig * np.sqrt(2 * np.pi)))     # from half a shot per bin
        a.set_xlabel('% of the trailing level'); a.legend(fontsize=7)
    for a in axes.flat[len(segs):]:
        a.axis('off')
    fig.suptitle(f'test regions: BCM jitter, residual of "{MODEL}", and the BCM noise floor', fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=100)
    print('->', png)
    pd.set_option('display.width', 250)
    print('test regions, all shots (spikes kept); Gaussian: skew 0, excess kurtosis 0, beyond 4 sigma 0.006 %')
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
