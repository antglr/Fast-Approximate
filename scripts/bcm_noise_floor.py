# How much of the BCM's shot-to-shot jitter is its own measurement noise? Three-cornered hat on the BCM, the camera
# intensity (BTV0125) and a third reading of the same shot. Each jitter (fast part, as jitter_sequence.py: against the
# mean of the previous SLOW readings; logs, the step in nC over the BCM level) = gain x the true charge jitter + its own
# noise. With independent noises,
#   noise variance of a = var(a) - cov(a, b) cov(a, c) / cov(b, c)
# so the gains (the camera's response, the step's calibration) do not matter. For the BCM and the camera the third
# reading need only correlate with the charge: a laser meter, which sees only the laser-driven part, works too (its
# own "noise" then includes the rest of the charge jitter). Third readings: the beam-loading step (useless at low
# charge, r ~ 0.02), clex, lab; their agreement is the check. Shots where all three read are used,
# whole segments (the noise does not drift), and with 5-MAD spikes removed (any reading); 95 % block bootstrap
# intervals (50-shot blocks). The BCM noise is the floor for the residual of any same-shot model of the jitter: it is
# compared with jitter_sequence.py's residual rms (target rms x sqrt(1 - jitter explained), test regions).
# Usage: python -I bcm_noise_floor.py <tables dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from jitter_sequence import trailing
from same_shot_regression import stitched

BLOCK, BOOT = 50, 300


def hat(S):
    # noise std of each column (%) from the 3x3 covariance; NaN where the variance left is negative
    C = np.cov(S.T)
    v = [C[a, a] - C[a, b] * C[a, c] / C[b, c] for a, b, c in ((0, 1, 2), (1, 0, 2), (2, 0, 1))]
    return 100 * np.sqrt(np.where(np.array(v) > 0, v, np.nan))


def boot(S, rng):
    n = len(S) // BLOCK
    starts = rng.integers(0, len(S) - BLOCK, (BOOT, n))
    return np.array([hat(S[(s[:, None] + np.arange(BLOCK)).ravel()]) for s in starts])


def main():
    tables, runs = sys.argv[1], sys.argv[2:]
    F = stitched(tables, runs)
    level = trailing(np.log(F.Q), F.seg)
    cam = np.log(F.BTV_intensity.where(F.BTV_intensity > 0))
    fast = lambda v: np.log(v.where(v > 0)) - trailing(np.log(v.where(v > 0)), F.seg)
    S = pd.DataFrame({'BCM': fast(F.Q), 'camera': fast(F.BTV_intensity), 'clex': fast(F.clex), 'lab': fast(F.lab),
                      'step': (F.step - trailing(F.step, F.seg)) / np.exp(level)})
    dev = S - S.groupby(F.seg).transform('median')
    spike = (dev.abs() > 5 * 1.4826 * dev.abs().groupby(F.seg).transform('median')).any(axis=1)
    rng = np.random.default_rng(0)
    rows = []
    for third in ('step', 'clex', 'lab'):
        T = S[['BCM', 'camera', third]].dropna()
        for s, g in T.groupby(F.seg[T.index]):
            for clean in (False, True):
                X = g[~spike[g.index]].values if clean else g.values
                if len(X) < 4 * BLOCK:
                    continue
                lo, hi = np.nanpercentile(boot(X, rng), [2.5, 97.5], axis=0)
                r, h = np.corrcoef(X.T), hat(X)
                rows.append({'third': third, 'seg': s, 'spikes removed': clean, 'n': len(X), 'BCM jitter %': 100 * X[:, 0].std(),
                             'r BCM-cam': r[0, 1], 'r BCM-3rd': r[0, 2], 'r cam-3rd': r[1, 2],
                             **{f'{k} noise %': f'{v:.2f} [{a:.2f}, {b:.2f}]' for k, v, a, b in zip(('BCM', 'camera', '3rd'), h, lo, hi)},
                             'BCM noise share %': 100 * (h[0] / (100 * X[:, 0].std())) ** 2})
    # the same on the test regions only (spikes removed), to set against the models' residuals there
    test = []
    for third in ('step', 'clex', 'lab'):
        T = S[['BCM', 'camera', third]][~F.train & ~spike].dropna()
        for s, g in T.groupby(F.seg[T.index]):
            if len(g) >= 4 * BLOCK:
                lo, hi = np.nanpercentile(boot(g.values, rng), [2.5, 97.5], axis=0)
                test.append({'third': third, 'seg': s, 'BCM noise %': f'{hat(g.values)[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}]'})
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(f'jitter = fast part against the mean of the previous readings; noise std in % of the level, 95 % block bootstrap; '
          f'spikes = any of BCM, camera, clex, lab, step > 5 MAD within the segment ({spike.mean():.2%} of shots)')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f'\nBCM noise %, test regions only (last 10 % of each segment, >= {4 * BLOCK} shots), spikes removed')
    print(pd.DataFrame(test).pivot(index='third', columns='seg', values='BCM noise %').to_string())


if __name__ == '__main__':
    main()
