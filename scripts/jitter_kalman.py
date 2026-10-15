# Kalman level filter instead of jitter_sequence.py's slow part (mean log charge of the previous 50 good shots).
# Local level on y = log Q: l_t = l_{t-1} + w, var(w) = q per machine pulse (so skipped pulses widen the level), and
# y_t = l_t + e_t, var(e_t) = the BCM noise of bcm_noise_model.py, sigma_n(Q)^2 / Q^2 with sigma_n^2 = A^2 + (B Q)^2,
# plus c^2 for the shot-to-shot jitter. Innovations beyond GATE sigma do not update the level (BCM-lost shots, spikes);
# after RESET rejected shots in a row the level jumps to the reading (a real step). The prediction for shot t uses
# shots < t only, as the trailing mean. q and c per regime on the early-stopping rows (last 10 % of each training part),
# by the level's one-step MAE in pC. Compared with the trailing mean and a robust EMA (half-life 25 shots, 4-sigma
# gate on the training-part jitter MAD, as adnan_01). Test MAE in pC per segment, of the level alone and of the level
# plus jitter_sequence.py's ridges refitted on each level's target (inputs unchanged: fast parts of the instruments).
# Plot: the charge and each level on the last 600 shots of every segment (test region shaded).
# Usage: python -I jitter_kalman.py <tables dir> <out png> <run> [<run> ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from jitter_gp import A, B
from jitter_sequence import ALPHAS, CV, GROUP, LAGS, LINEAR, NAMES, features, trailing
from same_shot_regression import REGIMES, stitched

GATE, RESET = 4.0, 5
QS = 10.0 ** np.arange(-8, -2.9, 0.5)       # level random-walk variance per pulse, log units
CS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)          # shot-to-shot jitter std on top of the BCM noise, log units
HALF = 25


def kalman(y, gap, seg, q, c, gate=GATE):
    # one-step prediction of the level from the readings before each shot; NaN on each segment's first shot
    out = np.full(len(y), np.nan)
    l = p = np.nan
    rej = 0
    for i in range(len(y)):
        if i == 0 or seg[i] != seg[i - 1]:
            l, p, rej = np.nan, 1.0, 0
        if not np.isfinite(l):
            l = y[i]
            continue
        p += q * gap[i]
        out[i] = l
        Q = 1000 * np.exp(l)
        s = p + (A ** 2 + (B * Q) ** 2) / Q ** 2 + c ** 2
        if not np.isfinite(y[i]):
            continue
        if (y[i] - l) ** 2 < gate ** 2 * s:
            k = p / s
            l, p, rej = l + k * (y[i] - l), p * (1 - k), 0
        elif (rej := rej + 1) >= RESET:
            l, p, rej = y[i], s, 0
    return out


def ema(y, seg, sd, half=HALF):
    # robust EMA: readings more than GATE sd from the current level are skipped
    a, out, l = 1 - 0.5 ** (1 / half), np.full(len(y), np.nan), np.nan
    for i in range(len(y)):
        if i == 0 or seg[i] != seg[i - 1]:
            l = np.nan
        if not np.isfinite(l):
            l = y[i]
            continue
        out[i] = l
        if np.isfinite(y[i]) and abs(y[i] - l) < GATE * sd[i]:
            l += a * (y[i] - l)
    return out


def main():
    tables, png, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    F = stitched(tables, runs)
    y = np.log(F.Q.where(F.Q > 0))
    seg = F.seg.values
    gap = F.shot.diff().where(F.seg.eq(F.seg.shift())).fillna(1).clip(lower=1).values
    print(f'pulses between rows: {pd.Series(gap).value_counts().sort_index().head(6).to_dict()}')
    old = trailing(y, F.seg)
    ok = old.notna()
    fit_rows = F.train & ok
    k = fit_rows.groupby(F.seg).cumsum()
    va = fit_rows & (k > 0.9 * k.groupby(F.seg).transform('max'))
    test = ~F.train & ok
    mae = lambda lev, i: 1000 * (np.exp(lev[i]) - F.Q[i]).abs().mean()

    # self-check: on a random walk plus white noise, the Kalman level beats the 50-shot mean, and a step is followed
    rng = np.random.default_rng(0)
    n = 20000
    lv = np.cumsum(rng.normal(0, 3e-3, n)) + np.log(0.5) + np.r_[np.zeros(n // 2), 0.5 * np.ones(n - n // 2)]
    ys = pd.Series(lv + rng.normal(0, 0.03, n))
    kl, tl = kalman(ys.values, np.ones(n), np.zeros(n), 9e-6, 0.03), trailing(ys, pd.Series(np.zeros(n)))
    e = lambda a: np.nanmean((a[100:] - lv[100:]) ** 2)
    assert e(kl) < 0.5 * e(tl.values), (e(kl), e(tl.values))
    assert abs(kl[n // 2 + 20] - lv[n // 2 + 20]) < 0.05

    levels = {'trailing 50': old}
    sd = pd.Series(np.nan, F.index)
    for rname in REGIMES:
        R = F.regime == rname
        d = (y - old)[fit_rows & R]
        sd[R] = 1.4826 * (d - d.median()).abs().median()
    levels[f'robust EMA, half-life {HALF}'] = pd.Series(ema(y.values, seg, sd.values), F.index)
    X = features(F)
    # as features(), but the inputs' slow part is the same EMA as the target's (a level mismatch would hide a gain)
    Xr = pd.DataFrame({k: F[k] if k in LINEAR else np.log(F[k].where(F[k] > 0)) for k in NAMES})
    Xe = (Xr - Xr.apply(lambda c: pd.Series(ema(c.values, seg, np.full(len(c), np.inf)), F.index))).fillna(0)
    Xe[[c for c in X if c.endswith(' missing')]] = X[[c for c in X if c.endswith(' missing')]]
    levels[f'robust EMA, half-life {HALF}, inputs too'] = levels[f'robust EMA, half-life {HALF}']

    def with_ridge(lev, R):
        # early-stopping MAE of the level plus a current-shot ridge fitted on the rest of the training part
        yf = y[R] - lev
        fr = (fit_rows & ~va)[R] & yf.notna()
        p = RidgeCV(alphas=ALPHAS, cv=CV).fit(X[R][fr], yf[fr]).predict(X[R].fillna(0))
        return mae(lev + p, F.index[(R & va).values])

    for tname, score in (('level alone', lambda lev, R: mae(lev, F.index[(R & va).values])), ('level + ridge', with_ridge)):
        kal = pd.Series(np.nan, F.index)
        for rname in REGIMES:
            R = (F.regime == rname) & ok
            run = lambda q, c: pd.Series(kalman(y.values[R], gap[R], seg[R], q, c), F.index[R])
            best = min((score(run(q, c), R), q, c) for q in QS for c in CS)
            kal[R] = run(*best[1:])
            print(f'{rname}, tuned on {tname}: q = {best[1]:.1e} per pulse (level drifts {100 * np.sqrt(50 * best[1]):.2f} % '
                  f'per 50 pulses), c = {best[2]:.2f}, early-stopping MAE {best[0]:.1f} pC', flush=True)
        levels[f'Kalman, tuned on {tname}'] = kal

    rows = []
    for lname, lev in levels.items():
        X = Xe if lname.endswith('inputs too') else features(F)
        yf = y - lev
        past = pd.concat({f'BCM lag {j}': yf.groupby(F.seg).shift(j) for j in range(1, LAGS + 1)}, axis=1).fillna(0)
        preds = {'level only': pd.Series(0.0, F.index)}
        for name, XR in (('+ ridge, current shot', X), ('+ ridge, + past BCM', pd.concat([X, past], axis=1))):
            preds[name] = pd.Series(np.nan, F.index)
            for rname in REGIMES:
                R = (F.regime == rname) & ok & lev.notna()
                fr = fit_rows & R & yf.notna()
                preds[name][R] = RidgeCV(alphas=ALPHAS, cv=CV).fit(XR[fr], yf[fr]).predict(XR[R])
        for name, p in preds.items():
            for s, g in F[test].groupby('seg'):
                rows.append({'model': f'{lname}, {name}', 'seg': s, 'MAE pC': mae(lev + p, g.index)})
    D = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    order = [f'{l}, {p}' for p in ('level only', '+ ridge, current shot', '+ ridge, + past BCM') for l in levels]
    print('\nMAE pC, test regions')
    print(D.pivot_table(index='model', columns='seg', values='MAE pC').reindex(order).round(1).to_string())

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    segs = sorted(set(F.seg))
    fig, axes = plt.subplots((len(segs) + 1) // 2, 2, figsize=(16, 3 * ((len(segs) + 1) // 2)))
    for a, g in zip(axes.flat, segs):
        i = F.index[F.seg == g][-600:]
        a.plot(i, F.Q[i], '.', color='0.6', ms=2, label='BCM charge')
        for (name, lev), c in zip({k: v for k, v in levels.items() if not k.endswith('inputs too')}.items(), ('k', 'C2', 'C0', 'C3')):
            a.plot(i, np.exp(lev[i]), color=c, lw=1, label=name)
        t = i[~F.train[i]]
        a.axvspan(t[0], t[-1], color='C1', alpha=0.1, label='test region')
        lo, hi = F.Q[i].quantile([0.01, 0.99])
        a.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        a.set_title(f'segment {g}, {F.regime[i[0]]}', fontsize=9); a.set_ylabel('nC')
    axes.flat[0].legend(fontsize=7, ncol=3)
    for a in axes.flat[len(segs):]:
        a.axis('off')
    axes.flat[len(segs) - 1].set_xlabel('row (stitched good shots)')
    fig.suptitle('slow part: trailing mean vs robust EMA vs Kalman level (one-step predictions from earlier shots)', fontsize=10)
    fig.tight_layout(); fig.savefig(png, dpi=100)
    print('->', png)


if __name__ == '__main__':
    main()
