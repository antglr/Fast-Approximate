# More from the BCM noise estimate (bcm_noise_floor.py's three-cornered hat on BCM, camera and a third reading X), on
# the training parts only, so that a model may use the result:
#   1. a stronger X: a linear combination of lab, clex, scope and step (0 where missing) fitted to the BCM jitter,
#      cross-fitted (weights from the other half of the segment's training part, so they cannot pick up this half's
#      BCM noise). The hat holds for any X whose noise is independent of the BCM's and camera's, whatever it sees of the
#      charge, so a combination is allowed and a tighter one narrows the interval.
#   2. the noise against the charge: sigma_n(Q)^2 = a^2 + (b Q)^2, a fixed part in pC and a part proportional to the
#      charge, fitted across segments (non-negative least squares weighted by the bootstrap variance; intervals from
#      refitting each bootstrap replicate).
#   3. is the noise Gaussian? The same algebra one order up gives the noise's own 3rd and 4th cumulants, with no model:
#        k3(n_B) = k3(B) - cum(BBC) cum(BBX) / cum(BCX),   k4(n_B) = k4(B) - cum(BBBC) cum(BBBX) / cum(BBCX)
#      -> skewness and excess kurtosis of the noise (0 for a Gaussian), with spikes kept (the tails are the question)
#      and removed. Only identified if the true charge jitter is itself non-Gaussian (denominators); its excess kurtosis
#      is printed too.
#   4. is the noise steady? The hat (lab) per 1000-shot block, in pC: a drifting noise would also show as excess
#      kurtosis in 3.
# Usage: python -I bcm_noise_model.py <tables dir> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from jitter_sequence import trailing
from same_shot_regression import stitched

BLOCK, BOOT = 50, 300


def cum(*v):
    # joint cumulant of 3 or 4 zero-mean series
    if len(v) == 3:
        return np.mean(v[0] * v[1] * v[2])
    a, b, c, d = v
    E = lambda x, y: np.mean(x * y)
    return np.mean(a * b * c * d) - E(a, b) * E(c, d) - E(a, c) * E(b, d) - E(a, d) * E(b, c)


def noise(M):
    # columns B, C, X -> BCM noise std (log units), noise skewness, noise excess kurtosis, true-charge excess kurtosis
    B, C, X = (M - M.mean(0)).T
    vs = np.mean(B * C) * np.mean(B * X) / np.mean(C * X)
    vn = np.mean(B * B) - vs
    k3 = cum(B, B, B) - cum(B, B, C) * cum(B, B, X) / cum(B, C, X)
    k4s = cum(B, B, B, C) * cum(B, B, B, X) / cum(B, B, C, X)
    sn = np.sqrt(vn) if vn > 0 else np.nan
    return np.array([sn, k3 / sn ** 3, (cum(B, B, B, B) - k4s) / sn ** 4, k4s / vs ** 2])


def boot(M, rng):
    starts = rng.integers(0, len(M) - BLOCK, (BOOT, len(M) // BLOCK))
    return np.array([noise(M[(s[:, None] + np.arange(BLOCK)).ravel()]) for s in starts])


def main():
    tables, runs = sys.argv[1], sys.argv[2:]
    F = stitched(tables, runs)
    level = trailing(np.log(F.Q), F.seg)
    fast = lambda v: np.log(v.where(v > 0)) - trailing(np.log(v.where(v > 0)), F.seg)
    S = pd.DataFrame({'BCM': fast(F.Q), 'camera': fast(F.BTV_intensity), 'lab': fast(F.lab), 'clex': fast(F.clex),
                      'scope': fast(F.scope), 'step': (F.step - trailing(F.step, F.seg)) / np.exp(level)})
    dev = S - S.groupby(F.seg).transform('median')
    spike = (dev.abs() > 5 * 1.4826 * dev.abs().groupby(F.seg).transform('median')).any(axis=1)
    keep = F.train & S[['BCM', 'camera']].notna().all(axis=1)
    rng = np.random.default_rng(0)
    # self-check on synthetic data: Gaussian noise under a non-Gaussian charge -> skew, kurtosis ~ 0, std recovered
    s = rng.exponential(1, 200000)
    t = noise(np.c_[s + rng.normal(0, 0.5, len(s)), 2 * s + rng.normal(0, 1, len(s)), s + rng.normal(0, 1, len(s))])
    assert abs(t[0] - 0.5) < 0.02 and abs(t[1]) < 0.3 and abs(t[2]) < 0.5 and abs(t[3] - 6) < 1, t

    rows, fitset = [], []
    for sg, g in S[keep].groupby(F.seg[keep]):
        if len(g) < 8 * BLOCK:
            continue
        Z, half = g[['lab', 'clex', 'scope', 'step']].fillna(0).values, np.arange(len(g)) >= len(g) // 2
        comb = np.zeros(len(g))
        for h in (half, ~half):
            Zh = np.c_[Z[~h], np.ones((~h).sum())]
            comb[h] = np.c_[Z[h], np.ones(h.sum())] @ np.linalg.lstsq(Zh, g.BCM.values[~h], rcond=None)[0]
        g = g.assign(comb=comb)
        q = F.Q[g.index].mean()
        for third in ('lab', 'comb'):
            for clean in (False, True):
                T = g[['BCM', 'camera', third]][~spike[g.index]] if clean else g[['BCM', 'camera', third]]
                M = T.dropna().values
                b = boot(M, rng)
                est, (lo, hi) = noise(M), np.nanpercentile(b, [2.5, 97.5], axis=0)
                rows.append({'seg': sg, 'third': third, 'spikes removed': clean, 'n': len(M), 'Q nC': q,
                             'r BCM-X': np.corrcoef(M[:, 0], M[:, 2])[0, 1],
                             **{k: f'{e:.3g} [{a:.3g}, {z:.3g}]' for k, e, a, z in
                                zip(('noise %', 'noise skew', 'noise ex. kurt', 'charge ex. kurt'), est * [100, 1, 1, 1], lo * [100, 1, 1, 1], hi * [100, 1, 1, 1])}})
                if third == 'comb' and clean:
                    fitset.append((q, 1000 * q * est[0], 1000 * q * b[:, 0]))
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('BCM noise from the three-cornered hat on the training parts; [95 % block bootstrap]; X = lab, or comb = '
          'cross-fitted combination of lab, clex, scope, step')
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # noise in pC against the charge (comb, spikes removed): sigma^2 = a^2 + (b Q)^2
    q = 1000 * np.array([f[0] for f in fitset]); sd = np.array([f[1] for f in fitset]); bs = np.array([f[2] for f in fitset]).T
    w = 1 / np.nanvar(bs ** 2, axis=0)
    fit = lambda v2: np.sqrt(nnls(np.c_[np.ones_like(q), q ** 2] * np.sqrt(w)[:, None], v2 * np.sqrt(w))[0])
    ab = fit(sd ** 2)
    abb = np.array([fit(r ** 2) for r in bs if np.isfinite(r).all()])
    lo, hi = np.percentile(abb, [2.5, 97.5], axis=0)
    print(f'\nsigma_n(Q)^2 = a^2 + (b Q)^2: a = {ab[0]:.1f} pC [{lo[0]:.1f}, {hi[0]:.1f}], b = {100 * ab[1]:.2f} % [{100 * lo[1]:.2f}, {100 * hi[1]:.2f}]')
    print(pd.DataFrame({"Q nC": q / 1000, "noise pC": sd, "fit pC": np.sqrt(ab[0] ** 2 + (ab[1] * q) ** 2)}).round(2).to_string(index=False))

    print('\nBCM noise pC per 1000-shot block (whole segments, X = lab; NaN = negative variance left, too noisy)')
    for sg in sorted(set(F.seg)):
        g = S.loc[F.seg == sg, ['BCM', 'camera', 'lab']].dropna()
        blocks = [g.iloc[k:k + 1000] for k in range(0, len(g) - 999, 1000)]
        if blocks:
            print(sg, np.round([1000 * noise(b.values)[0] * F.Q[b.index].mean() for b in blocks], 1))


if __name__ == '__main__':
    main()
