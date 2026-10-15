# Are the gains real, and how wide should the error bars be? Reads the per-shot predictions saved by
# regime_phase2.py (same shot) and feature_forecast.py (h shots ahead); nothing is retrained.
#   1. circular block bootstrap of the test shots (50-shot blocks): 95 % interval of each MAE reduction
#   2. 90 % intervals, constant width (split-conformal on the validation block) against heteroscedastic ones:
#      prediction x exp(q x sigma_t), sigma_t = robust spread of the last 200 log errors KNOWN at prediction time
#      (shots <= t - h), q = the 5 / 95 % quantiles of the normalised validation errors. Coverage is also given on the
#      calm and the noisy half of the test shots (by sigma_t): a good interval holds 90 % in both.
# python3 -I scripts/uncertainty_regimes.py results/datasets_aligned results/aligned
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from regimes import load  # noqa: E402

BLOCK, NBOOT, WIN = 50, 2000, 200


def boot_gain(err_a, err_b, rng):
    """% MAE reduction of b against a, and its 95 % circular-block-bootstrap interval."""
    n = len(err_a)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, (NBOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None]) % n
    idx = idx.reshape(NBOOT, -1)[:, :n]
    ga = err_a[idx].mean(1)
    gb = err_b[idx].mean(1)
    g = 100 * (1 - gb / ga)
    return 100 * (1 - err_b.mean() / err_a.mean()), *np.percentile(g, [2.5, 97.5])


def rolling_sigma(r, known, h):
    """Robust spread of the last WIN log errors known at each row (errors of good shots up to row - h)."""
    s = pd.Series(np.where(known, np.abs(r), np.nan))
    sig = 1.4826 * s.rolling(WIN, min_periods=30).median()
    return sig.shift(h).to_numpy()


def intervals(seg, pred, h):
    y, good = seg["y"], seg["good"]
    r = np.log(y) - np.log(pred)
    known = good & np.isfinite(r)
    sig = rolling_sigma(r, known, h)
    v, te = seg["val"] & np.isfinite(sig) & np.isfinite(r), seg["test"] & np.isfinite(sig) & np.isfinite(r)
    out = {}
    qlo, qhi = np.quantile(r[v], [0.05, 0.95])                       # constant width
    inside = (r[te] > qlo) & (r[te] < qhi)
    zlo, zhi = np.quantile(r[v] / sig[v], [0.05, 0.95])              # scaled by the recent spread
    lo, hi = zlo * sig[te], zhi * sig[te]
    inside_h = (r[te] > lo) & (r[te] < hi)
    calm = sig[te] <= np.median(sig[te])
    for tag, ins, width in (("constant", inside, np.full(te.sum(), qhi - qlo)), ("scaled", inside_h, hi - lo)):
        out[f"{tag} cover"] = ins.mean()
        out[f"{tag} cover calm"] = ins[calm].mean()
        out[f"{tag} cover noisy"] = ins[~calm].mean()
        out[f"{tag} width %"] = 100 * np.mean(width)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("results", help="folder with phase2/ and feature_forecast/")
    ap.add_argument("--robust", type=float, default=5)
    a = ap.parse_args()
    res = Path(a.results)
    segs, _ = load(a.datasets, robust=a.robust or None)
    rng = np.random.default_rng(0)
    out = res / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)

    # same shot: is the GRU really better than the ridge after robust preparation? and the ensemble?
    P = np.load(res / "phase2/predictions.npz")
    rows, ivs = [], []
    for s, g in segs.items():
        if g["test"].sum() < 50 or f"seg{s}|ridge" not in P:
            continue
        te = g["test"]
        err = {k.split("|")[1]: np.abs(P[k][te] - g["y"][te]) for k in P.files if k.startswith(f"seg{s}|")}
        for a_, b_ in (("level_only", "ridge"), ("ridge", "gru"), ("ridge", "ridge+gru"), ("gru", "mean_ensemble"),
                       ("ridge", "gp")):
            g0, lo, hi = boot_gain(err[a_], err[b_], rng)
            rows.append(dict(task="same shot", segment=s, regime=g["regime"], comparison=f"{b_} vs {a_}",
                             gain_pct=g0, lo95=lo, hi95=hi, significant=bool(lo > 0 or hi < 0)))
        ivs.append(dict(task="same shot", segment=s, regime=g["regime"], model="mean_ensemble", h=0,
                        **intervals(g, P[f"seg{s}|mean_ensemble"], 1)))

    # forecasting: every model against the 9-shot mean, every horizon
    F = np.load(res / "feature_forecast/predictions.npz")
    keys = pd.DataFrame([k.split("|") for k in F.files], columns=["h", "seg", "model"])
    for (h, sg), d in keys.groupby(["h", "seg"]):
        s, hh = int(sg[3:]), int(h[1:])
        g = segs[s]
        te = g["test"] & np.isfinite(F[f"{h}|{sg}|mean9 (baseline)"])
        base = np.abs(F[f"{h}|{sg}|mean9 (baseline)"][te] - g["y"][te])
        for m in d.model:
            if m == "mean9 (baseline)":
                continue
            g0, lo, hi = boot_gain(base, np.abs(F[f"{h}|{sg}|{m}"][te] - g["y"][te]), rng)
            rows.append(dict(task="forecast", h=hh, segment=s, regime=g["regime"], comparison=f"{m} vs mean9",
                             gain_pct=g0, lo95=lo, hi95=hi, significant=bool(lo > 0)))
        ivs.append(dict(task="forecast", h=hh, segment=s, regime=g["regime"], model="gru L=32, all",
                        **intervals(g, F[f"{h}|{sg}|gru L=32, all"], hh)))

    sig, iv = pd.DataFrame(rows), pd.DataFrame(ivs)
    sig.to_csv(out / "significance.csv", index=False)
    iv.to_csv(out / "intervals.csv", index=False)
    fmt = lambda r: f"{r.gain_pct:+.1f} [{r.lo95:+.1f}, {r.hi95:+.1f}]" + ("" if r.significant else " ns")  # noqa: E731
    sig["cell"] = sig.apply(fmt, axis=1)
    with open(out / "summary.md", "w") as f:
        f.write("# Bootstrap (50-shot circular blocks) and intervals\n\n## Same shot: % MAE reduction [95 %]\n\n")
        f.write(sig[sig.task == "same shot"].pivot(index="comparison", columns="segment", values="cell").to_markdown())
        f.write("\n\n## Forecast: % MAE reduction vs the 9-shot mean [95 %]\n\n")
        f.write(sig[sig.task == "forecast"].pivot_table(index=["h", "comparison"], columns="segment", values="cell",
                                                         aggfunc="first").to_markdown())
        f.write("\n\n## 90 % intervals on the test shots\n\n")
        f.write(iv.round(3).to_markdown(index=False))
        f.write("\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
