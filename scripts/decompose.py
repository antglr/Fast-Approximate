# Phase 1 of docs/workplan_adnan_01.md: separate the slow level from the shot-to-shot jitter, per regime.
#   1. spectra: how the charge variance splits into slow / mid / fast bands, and how coherent it is with the camera and laser
#   2. causal level filters (grid), checked for future leaks, chosen per regime on the validation block
# python3 -I scripts/decompose.py results/datasets results/decompose
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from regimes import REGIMES, load, score  # noqa: E402

BANDS = {"slow (> 1000 shots)": (0, 1 / 1000), "mid (50-1000)": (1 / 1000, 1 / 50), "fast (< 50 shots)": (1 / 50, 0.5)}


# ---------------------------------------------------------------------------- causal filters
# Each maps the series of good values x[0..m) to est[i] = the level after seeing x[0..i]; causal() turns that into
# "level of row n from the good rows strictly before n", as regimes.trailing_mean does.
def f_sma(x, k):
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()


def f_median(x, k):
    return pd.Series(x).rolling(k, min_periods=1).median().to_numpy()


def f_ema(x, hl):
    # = the steady-state Kalman filter of a local-level model (random walk + white noise)
    return pd.Series(x).ewm(halflife=hl, adjust=True).mean().to_numpy()


def f_holt(x, alpha, beta):
    # local linear trend, one step ahead (level + slope): the causal counterpart of a Hodrick-Prescott trend
    est = np.empty(len(x))
    lv, b = x[0], 0.0
    for i, v in enumerate(x):
        prev = lv
        lv = alpha * v + (1 - alpha) * (lv + b)
        b = beta * (lv - prev) + (1 - beta) * b
        est[i] = lv + b
    return est


def f_robust_ema(x, hl, c=3.0):
    # EMA whose update is clipped at c robust sigmas, so single wild shots barely move the level
    a = 1 - 0.5 ** (1 / hl)
    est = np.empty(len(x))
    lv, s = x[0], np.std(x[:50]) + 1e-9
    for i, v in enumerate(x):
        e = v - lv
        lv += a * np.clip(e, -c * s, c * s)
        s = 0.98 * s + 0.02 * 1.25 * abs(e) + 1e-12
        est[i] = lv
    return est


def f_cp_reset(x, z, wmax):
    # mean since the last level change (|x - mean| > z robust sigmas), capped at wmax shots: the hybrid of hybrid_forecast.py
    est = np.empty(len(x))
    start, s = 0, np.std(x[:50]) + 1e-9
    for i, v in enumerate(x):
        lo = max(start, i - wmax)
        mu = x[lo:i].mean() if i > lo else v
        if i - start > 3 and abs(v - mu) > z * s:
            start = i
        else:
            s = 0.98 * s + 0.02 * 1.25 * abs(v - mu) + 1e-12
        lo = max(start, i + 1 - wmax)
        est[i] = x[lo:i + 1].mean()
    return est


GRID = (
    [("sma", partial(f_sma, k=k), f"k={k}") for k in (10, 20, 50, 100, 200, 500)]
    + [("ema", partial(f_ema, hl=h), f"hl={h}") for h in (5, 10, 25, 50, 100, 250)]
    + [("median", partial(f_median, k=k), f"k={k}") for k in (11, 25, 51, 101, 201)]
    + [("holt", partial(f_holt, alpha=a, beta=b), f"a={a},b={b}") for a in (0.02, 0.05, 0.1) for b in (0.001, 0.01)]
    + [("robust_ema", partial(f_robust_ema, hl=h), f"hl={h}") for h in (10, 25, 50, 100)]
    + [("cp_reset", partial(f_cp_reset, z=z, wmax=w), f"z={z},w={w}") for z in (4, 6, 8) for w in (20, 50, 200)]
)


def causal(fn):
    def level(a, ok):
        a2 = a.reshape(len(a), -1).astype(float)
        out = np.full_like(a2, np.nan)
        for j in range(a2.shape[1]):
            m = ok & np.isfinite(a2[:, j])
            col = np.full(len(a2), np.nan)
            if m.sum():
                col[np.flatnonzero(m)] = fn(a2[m, j])
            out[:, j] = pd.Series(col).shift(1).ffill().to_numpy()
        return out.reshape(a.shape)
    return level


def leak_test():
    """Changing the future must not change any past level."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, 600)) + rng.normal(0, 0.05, 600)
    ok = rng.random(600) > 0.2
    for name, fn, par in GRID:
        a = causal(fn)(x, ok)
        x2 = x.copy()
        x2[300:] += rng.normal(0, 1, 300)
        b = causal(fn)(x2, ok)
        assert np.allclose(a[:301], b[:301], equal_nan=True), f"{name} {par} looks ahead"
    print(f"leak test passed for all {len(GRID)} filters", flush=True)


# ---------------------------------------------------------------------------- 1. spectra
def spectra(segs, names, out):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (regime, ss) in zip(axes, REGIMES.items()):
        for s in ss:
            g = segs.get(s)
            if g is None or g["n"] < 1000:
                continue
            tr = np.arange(g["n"]) < int(0.9 * g["n"])           # training part only
            y = pd.Series(np.where(g["good"], g["logy"], np.nan))[tr].ffill().bfill().to_numpy()
            nper = min(4096, len(y) // 4)
            f, p = signal.welch(y - y.mean(), nperseg=nper)
            ax.loglog(f[1:], p[1:], label=f"seg {s}")
            tot = np.trapezoid(p, f)
            row = {"regime": regime, "segment": s, "rms_pct": 100 * np.std(y)}
            for b, (lo, hi) in BANDS.items():
                sel = (f >= lo) & (f < hi)
                row[f"var share {b}"] = np.trapezoid(p[sel], f[sel]) / tot
            for inst in ("BTV_intensity", "lab_laser_power"):
                x = pd.Series(g["X"][:, names.index(inst)])[tr].ffill().bfill().to_numpy()
                fc, c = signal.coherence(y, x, nperseg=nper)
                for b, (lo, hi) in BANDS.items():
                    sel = (fc >= lo) & (fc < hi) & (fc > 0)
                    row[f"coh {inst} {b}"] = c[sel].mean() if sel.any() else np.nan
            if regime == "high_rep":                              # the sawtooth period
                sel = (f > 1 / 6000) & (f < 1 / 300)
                row["peak period (shots)"] = 1 / f[sel][np.argmax(p[sel])]
            rows.append(row)
        ax.set_title(regime)
        ax.set_xlabel("frequency (1/shot)")
        for lo, hi in BANDS.values():
            ax.axvline(hi, color="grey", lw=0.5, ls=":")
        ax.legend()
    axes[0].set_ylabel("PSD of log BCM (training part)")
    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=120)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 2. filters
def acf1(a):
    a = a - a.mean()
    return float(np.sum(a[1:] * a[:-1]) / np.sum(a * a))


def evaluate(datasets, names, robust=None):
    rows = []
    for name, fn, par in GRID:
        segs, _ = load(datasets, fast_filter=causal(fn), robust=robust)
        for regime, ss in REGIMES.items():
            G = [segs[s] for s in ss if s in segs]
            X = np.concatenate([np.c_[g["Xfast"], g["missing"]] for g in G])
            t = np.nan_to_num(np.concatenate([g["jitter"] for g in G]))
            fit = np.concatenate([g["fit"] for g in G])
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X[fit], t[fit])
            for g in G:
                if g["val"].sum() < 50:
                    continue
                pj = m.predict(np.c_[g["Xfast"], g["missing"]])
                v, te = g["val"], g["test"]
                j = g["jitter"]
                cam = g["Xfast"][:, names.index("BTV_intensity")]
                lab = g["Xfast"][:, names.index("lab_laser_power")]
                level_only = score(g, np.exp(g["level"]), v)
                with_ridge = score(g, np.exp(g["level"] + pj), v)
                rows.append(dict(
                    filter=name, param=par, regime=regime, segment=g["segment"],
                    val_level_mae_pct=level_only["mae_pct"], val_ridge_mae_pct=with_ridge["mae_pct"],
                    val_jitter_explained=with_ridge["jitter_explained"], val_jitter_acf1=acf1(j[v]),
                    val_corr_camera=np.corrcoef(j[v], cam[v])[0, 1], val_corr_lab=np.corrcoef(j[v], lab[v])[0, 1],
                    test_level_mae_pct=score(g, np.exp(g["level"]))["mae_pct"],
                    test_ridge_mae_pct=score(g, np.exp(g["level"] + pj))["mae_pct"]))
        print(f"{name:10s} {par:14s} done", flush=True)
    return pd.DataFrame(rows)


def main():
    plt.switch_backend("Agg")                                  # here, not at import: notebooks import this module
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    ap.add_argument("--robust", type=float, default=0, help="k: MAD outlier handling (regimes.load)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    leak_test()
    segs, names = load(a.datasets, robust=a.robust or None)
    bands = spectra(segs, names, out)
    bands.to_csv(out / "bands.csv", index=False)
    df = evaluate(a.datasets, names, a.robust or None)
    df.to_csv(out / "filters.csv", index=False)
    per = df.groupby(["regime", "filter", "param"]).mean(numeric_only=True).drop(columns="segment").reset_index()
    best_param = per.loc[per.groupby(["regime", "filter"])["val_ridge_mae_pct"].idxmin()]
    choice = per.loc[per.groupby("regime")["val_ridge_mae_pct"].idxmin(), ["regime", "filter", "param"]]
    sma50 = per[(per["filter"] == "sma") & (per["param"] == "k=50")]
    (out / "choice.json").write_text(json.dumps(choice.to_dict("records"), indent=1))
    with open(out / "summary.md", "w") as f:
        f.write("# Phase 1: level / jitter decomposition per regime\n\n## Spectra (training parts)\n\n")
        f.write(bands.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Best parameter of each filter family (chosen on validation), regime means over segments\n\n")
        f.write(best_param.sort_values(["regime", "val_ridge_mae_pct"]).to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Reference: the current trailing 50-shot mean\n\n")
        f.write(sma50.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Chosen per regime\n\n" + choice.to_markdown(index=False) + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
