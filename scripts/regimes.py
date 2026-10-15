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


def load(datasets="results/datasets", fast_filter=None):
    """{segment: dict}. fast_filter(a, ok) -> level of a (causal); default = trailing K-shot mean (Phase 1 swaps it)."""
    fast_filter = fast_filter or trailing_mean
    manifest = json.loads((Path(datasets) / "manifest.json").read_text())
    names = list(manifest["features"])
    out = {}
    for info in manifest["segments"]:
        z = np.load(Path(datasets) / info["file"])
        X, y, valid = z["X"].astype(float), z["y"].astype(float), z["valid"]
        n = len(y)
        rows = np.arange(n)
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
        Xfast = Xl - fast_filter(Xl, np.ones(n, bool))
        miss = ~np.isfinite(Xl)
        out[info["segment"]] = dict(
            segment=info["segment"], regime=REGIME_OF[info["segment"]], run=info["run"], n=n,
            y=y, logy=logy, level=level, jitter=logy - level, X=Xl, Xfast=np.where(miss | ~np.isfinite(Xfast), 0.0, Xfast),
            missing=miss.any(1).astype(float), good=good,
            fit=good & (rows < val0), val=good & (rows >= val0) & (rows < test0), test=good & (rows >= test0),
            train=good & (rows < test0))
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
