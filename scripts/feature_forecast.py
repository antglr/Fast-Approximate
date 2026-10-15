# Phase 3 of docs/workplan_adnan_01.md: forecast the charge h shots ahead from everything known at t - h:
# the BCM history and every instrument, per regime, with the robust preparation.
#   target: log Q[t] - log(mean of the last 9 good shots up to t - h)   (the challenge baseline, so 0 = baseline)
#   inputs at t - h: BCM history relative to that baseline (last shot, means over 3 / 50 / 200 good shots, robust EMA,
#   Holt), the fast parts of the instruments and their slow parts (log reading - its 200-shot trailing mean), flags
#   models: ridge, XGBoost, GRU over the last 32 known shots (regime_bench.py), plus "history only" ridge and the GP
#   of level_forecast.py for reference
# python3 -I scripts/feature_forecast.py results/datasets results/feature_forecast
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from regimes import REGIMES, load, score, trailing_mean  # noqa: E402
from decompose import f_holt, f_robust_ema  # noqa: E402
from regime_bench import fit_tabular, fit_torch  # noqa: E402

HORIZONS = [1, 10, 50]


def upto(seg, fn):
    """fn over the good shots, value at row r = after seeing the good shots up to and including r."""
    good = seg["fit"] | seg["val"] | seg["test"] | seg["good"]
    idx = np.flatnonzero(good)
    col = np.full(seg["n"], np.nan)
    col[idx] = fn(seg["logy"][idx])
    return pd.Series(col).ffill().bfill().to_numpy()


# In results/datasets these columns hold the NEXT shot's reading on part of the rows (the meter_shift rule of
# pulses.py is not applied per shot): row t - 1 correlates with shot t's jitter as much as row t does (lab, seg 7:
# 0.56 against 0.34). For forecasting they are only used one row later, so nothing from shot t can leak in.
LATE = ("lab_laser_power", "PGLA_drop")


def features(seg, names, late_fix=True):
    """Row r: what is known after shot r. Returns (baseline log level at r, history features, instrument features)."""
    base = upto(seg, lambda x: pd.Series(x).rolling(9, min_periods=1).mean().to_numpy())
    hist = [upto(seg, lambda x, k=k: pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()) - base
            for k in (1, 3, 50, 200)]
    hist += [upto(seg, lambda x: f_robust_ema(x, 25)) - base, upto(seg, lambda x: f_holt(x, 0.05, 0.001)) - base]
    H = np.column_stack(hist + [seg["good"].astype(float)])
    Xl = seg["X"]
    slow = Xl - trailing_mean(Xl, np.ones(len(Xl), bool), 200)   # per column, skips missing readings
    inst = np.column_stack([seg["Xfast"], np.nan_to_num(np.where(np.isfinite(slow), slow, 0.0))])
    late = [names.index(f) for f in LATE if f in names] if late_fix else []
    for c in late + [len(names) + c for c in late]:
        inst[1:, c], inst[0, c] = inst[:-1, c].copy(), 0.0
    return base, H, np.column_stack([inst, seg["missing"]])


def shift(a, h):
    """Row t gets row t - h (zero-padded): inputs known h shots before the target."""
    out = np.zeros_like(a)
    out[h:] = a[:-h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    ap.add_argument("--robust", type=float, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--no-late-fix", action="store_true", help="keep lab / PGLA on their row: shows the h = 1 leak")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    segs, names = load(a.datasets, robust=a.robust or None)
    F = {s: features(g, names, not a.no_late_fix) for s, g in segs.items()}
    rows = []
    for h in map(int, a.horizons.split(",")):
        for regime, ss in REGIMES.items():
            G = [segs[s] for s in ss if s in segs]
            base = [shift(F[g["segment"]][0][:, None], h)[:, 0] for g in G]
            for g, b in zip(G, base):
                b[:h] = np.nan
            T = [np.nan_to_num(g["logy"] - b) for g, b in zip(G, base)]
            Xh = [shift(F[g["segment"]][1], h) for g in G]
            Xa = [np.c_[shift(F[g["segment"]][1], h), shift(F[g["segment"]][2], h)] for g in G]
            ok = lambda m: [mm & (np.arange(g["n"]) >= h + 9) for mm, g in zip(m, G)]  # noqa: E731
            fit, val = ok([g["fit"] for g in G]), ok([g["val"] for g in G])
            preds = {"mean9 (baseline)": [np.zeros(g["n"]) for g in G]}
            for name, XX, model in (("ridge, history only", Xh, "ridge"), ("ridge, all", Xa, "ridge"),
                                    ("xgboost, all", Xa, "xgboost")):
                X, t = np.concatenate(XX), np.concatenate(T)
                pr = fit_tabular(model, X, t, np.concatenate(fit), np.concatenate(val), a.quick)
                preds[name] = [pr(x) for x in XX]
            preds["gru L=32, all"] = fit_torch("gru", 32, Xa, T, fit, val, a.quick)(Xa)
            for i, g in enumerate(G):
                if g["test"].sum() < 50:
                    continue
                for name, p in preds.items():
                    sc = score(g, np.exp(base[i] + p[i]))
                    rows.append(dict(h=h, regime=regime, segment=g["segment"], model=name, **sc))
            print(f"h={h} {regime} done", flush=True)
            pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    df = pd.DataFrame(rows)
    ref = df[df.model == "mean9 (baseline)"].set_index(["h", "segment"])["mae_nC"]
    df["vs_baseline_pct"] = 100 * (df.mae_nC / ref.reindex(pd.MultiIndex.from_frame(df[["h", "segment"]])).to_numpy() - 1)
    df.to_csv(out / "scores.csv", index=False)
    with open(out / "summary.md", "w") as f:
        for metric in ("vs_baseline_pct", "mae_pct"):
            tab = df.pivot_table(index=["h", "model"], columns=["regime", "segment"], values=metric)
            f.write(f"\n## {metric}\n\n{tab.to_markdown(floatfmt='.2f')}\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
