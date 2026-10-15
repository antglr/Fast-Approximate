# Shared data, split and score for the per-regime study (branch adnan_01, docs/workplan_adnan_01.md).
# One row = one shot of a clean segment (results/datasets/segment_XX.npz). Everything causal unless marked "scoring".
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = {"low": [0, 2, 3, 4], "high_rep": [5], "high_nonrep": [1, 6, 7]}
REGIME_OF = {s: r for r, segs in REGIMES.items() for s in segs}
TEST = 0.10            # last 10 % of each segment: test (challenge rule)
VAL = 0.10             # last 10 % of the training part: validation (early stopping, model choice)
K = 50                 # shots: trailing window for the level and the fast parts
DROPOUT = 0.30         # below 30 % of the local median: dropout, never trained on or scored (challenge rule)
POSITIVE = ["lab_laser_power", "SVD_s1", "BTV_intensity", "SVD_s5", "SVD_s6", "SVD_s4", "PKI15A", "BTV_mean_x"]


def trailing_mean(a, ok, k=K):
    """Mean of the last k rows where ok, strictly before each row (row n never sees itself). a: (n,) or (n, d)."""
    a2 = a.reshape(len(a), -1).astype(float)
    out = np.full_like(a2, np.nan)
    for j in range(a2.shape[1]):
        m = ok & np.isfinite(a2[:, j])
        c = pd.Series(a2[m, j]).rolling(k, min_periods=1).mean().to_numpy()
        col = np.full(len(a2), np.nan)
        col[np.flatnonzero(m)] = c
        out[:, j] = pd.Series(col).shift(1).ffill().to_numpy()
    return out.reshape(a.shape)


def mad_outliers(a, ref, k):
    """|a - median| > k robust sigmas, median and sigma from the rows ref (the fit part)."""
    med = np.nanmedian(a[ref], 0)
    sig = 1.4826 * np.nanmedian(np.abs(a[ref] - med), 0) + 1e-12
    return np.abs(a - med) > k * sig


def load(datasets="results/datasets", fast_filter=None, robust=None):
    """{segment: dict}.
    fast_filter(a, ok) -> level of a (causal), or {regime: fast_filter}; default = trailing K-shot mean.
    robust = k: input spikes beyond k MAD (fit part) become missing (a camera spike blanks the whole camera group), and
    shots whose jitter is beyond k MAD are not trained on (they are still scored)."""
    filters = fast_filter if isinstance(fast_filter, dict) else {}
    manifest = json.loads((Path(datasets) / "manifest.json").read_text())
    names = list(manifest["features"])
    out = {}
    for info in manifest["segments"]:
        z = np.load(Path(datasets) / info["file"])
        X, y, valid = z["X"].astype(float), z["y"].astype(float), z["valid"]
        n = len(y)
        rows = np.arange(n)
        regime = REGIME_OF[info["segment"]]
        fast_filter = filters.get(regime, trailing_mean) if filters or fast_filter is None else fast_filter
        test0 = int((1 - TEST) * n)
        val0 = int((1 - VAL) * test0)
        med = pd.Series(np.where(valid, y, np.nan)).shift(1).rolling(101, min_periods=5).median().bfill().to_numpy()
        good = valid & (y > DROPOUT * med) & (y > 0)
        logy = np.log(np.where(y > 0, y, np.nan))
        level = fast_filter(logy, good)                       # log of the charge level, BCM up to the previous shot
        level = np.where(np.isfinite(level), level, np.nanmean(logy[good & (rows < val0)]))
        Xl = X.copy()                                          # positive readings in log
        pos = [names.index(f) for f in POSITIVE if f in names]
        with np.errstate(invalid="ignore", divide="ignore"):
            Xl[:, pos] = np.log(np.where(X[:, pos] > 0, X[:, pos], np.nan))
        miss = ~np.isfinite(Xl)
        jitter = logy - level
        learn = good.copy()
        spike = np.zeros(n, bool)
        if robust:
            cam = [i for i, f in enumerate(names) if f.startswith(("BTV", "SVD"))]
            fit0 = good & (rows < val0)
            raw_fast = Xl - fast_filter(Xl, np.ones(n, bool))
            out_x = mad_outliers(raw_fast, fit0, robust)
            spike = out_x[:, cam].any(1)
            out_x[spike[:, None] & np.isin(np.arange(len(names)), cam)[None]] = True
            Xl = np.where(out_x, np.nan, Xl)                   # spikes are missing readings: the filters skip them
            learn &= ~mad_outliers(jitter, fit0, robust)
        ok_rows = np.isfinite(Xl)
        lvl = np.column_stack([fast_filter(Xl[:, j], ok_rows[:, j]) for j in range(Xl.shape[1])])
        Xfast = Xl - lvl
        bad = ~np.isfinite(Xfast)
        out[info["segment"]] = dict(
            segment=info["segment"], regime=regime, run=info["run"], n=n,
            y=y, logy=logy, level=level, jitter=jitter, X=Xl, Xfast=np.where(bad, 0.0, Xfast),
            missing=np.c_[miss.any(1), spike].astype(float), good=good,
            fit=learn & (rows < val0), val=learn & (rows >= val0) & (rows < test0), test=good & (rows >= test0),
            train=learn & (rows < test0))
    return out, names


def score(seg, pred_y, mask=None):
    """Scores of a charge prediction (nC) on mask (default: the test shots). The slow/fast split is scoring-only."""
    m = seg["test"] if mask is None else mask
    m = m & np.isfinite(pred_y) & (pred_y > 0)
    y, p, lv = seg["y"][m], pred_y[m], seg["level"][m]
    r = np.log(p) - np.log(y)                                  # log error ~ relative error
    j = np.log(y) - lv                                         # true jitter against the trailing level
    slow = pd.Series(r).rolling(51, center=True, min_periods=1).median().to_numpy()
    mae = np.mean(np.abs(p - y))
    return {
        "n": int(m.sum()), "level_nC": float(np.mean(y)),
        "mae_nC": float(mae), "mae_pct": float(100 * mae / np.mean(y)),
        "r2": float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "jitter_rms_pct": float(100 * np.sqrt(np.mean(j ** 2))),
        "resid_rms_pct": float(100 * np.sqrt(np.mean(r ** 2))),
        "jitter_explained": float(1 - np.mean(r ** 2) / np.mean(j ** 2)),
        "slow_share": float(np.mean(slow ** 2) / np.mean(r ** 2)),
    }


# BCM noise floor, rms % of the level, test regions (bcm_noise_floor.py, README: mean of clex / lab as third reading)
NOISE_FLOOR_PCT = {0: 4.8, 3: 9.3, 4: 11.0, 5: 0.72, 1: 1.22, 6: 0.78}
