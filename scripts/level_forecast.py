# Phase 2b of docs/workplan_adnan_01.md: forecast the charge level h shots ahead from the BCM history alone, per regime.
#   baselines: mean of the last 9 good shots (challenge baseline), robust EMA (phase 1 choice), Holt
#   GP on time: Matern 5/2 + white noise; for the repetitive regime + a periodic x RBF term (sawtooth ~2000 shots).
#   Hyperparameters fitted on the training part (subsampled); each test forecast conditions only on the last W good shots.
# python3 -I scripts/level_forecast.py results/datasets results/level_forecast --robust 5
import argparse
import sys
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from regimes import load, score  # noqa: E402
from decompose import f_holt, f_robust_ema  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = [1, 10, 50]
W, STRIDE = 2000, 10          # context: the last W good shots, every STRIDE-th one (+ the last 20 at full rate)


class TimeGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik, periodic):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        if periodic:
            per = gpytorch.kernels.PeriodicKernel()
            per.period_length = 2048.0
            k = k + gpytorch.kernels.ScaleKernel(per * gpytorch.kernels.RBFKernel())
        self.covar_module = k

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_hyper(t, y, periodic):
    x = torch.tensor(t, dtype=torch.float64, device=DEVICE)[:, None]
    yy = torch.tensor(y, dtype=torch.float64, device=DEVICE)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE).double()
    m = TimeGP(x, yy, lik, periodic).to(DEVICE).double()
    for mod in m.covar_module.modules():
        if isinstance(mod, (gpytorch.kernels.MaternKernel, gpytorch.kernels.RBFKernel)):
            mod.lengthscale = 300.0
    m.train(), lik.train()
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
    for _ in range(200):
        opt.zero_grad()
        (-mll(m(x), yy)).backward()
        opt.step()
    return m, lik


def gp_forecast(seg, periodic, rows):
    """Level forecasts (log) for the rows `rows`, for every horizon, each from the good shots before row - h + 1."""
    good = seg["fit"] | seg["val"] | seg["test"]
    idx = np.flatnonzero(good)
    ly = seg["logy"]
    tr = np.flatnonzero(seg["fit"] | seg["val"])[::STRIDE]
    mu0 = ly[tr].mean()
    m, lik = fit_hyper(tr.astype(float), ly[tr] - mu0, periodic)
    m.eval(), lik.eval()
    out = {}
    for h in HORIZONS:
        preds = np.full(len(rows), np.nan)
        for k, r in enumerate(rows):
            past = idx[idx <= r - h]                       # known when forecasting row r, h shots ahead
            if len(past) < 30:
                continue
            ctx = past[past > r - h - W]
            ctx = np.unique(np.r_[ctx[::-1][::STRIDE][::-1], ctx[-20:]])
            x = torch.tensor(ctx, dtype=torch.float64, device=DEVICE)[:, None]
            yy = torch.tensor(ly[ctx] - mu0, dtype=torch.float64, device=DEVICE)
            m.set_train_data(x, yy, strict=False)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds[k] = m(torch.tensor([[float(r)]], dtype=torch.float64, device=DEVICE)).mean.item() + mu0
        out[h] = preds
    return out


def causal_series(seg, fn, h):
    """Level of row r from the good shots up to row r - h (fn: the decompose.py filters)."""
    good = seg["fit"] | seg["val"] | seg["test"]
    idx = np.flatnonzero(good)
    est = fn(seg["logy"][idx])
    col = np.full(seg["n"], np.nan)
    col[idx] = est
    return pd.Series(col).ffill().shift(h).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    ap.add_argument("--robust", type=float, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    segs, _ = load(a.datasets, robust=a.robust or None)
    res = []
    for s, g in segs.items():
        if g["test"].sum() < 50:
            continue
        rows = np.flatnonzero(g["test"])
        cands = {
            "mean9 (baseline)": lambda h: causal_series(g, lambda x: pd.Series(x).rolling(9, min_periods=1).mean().to_numpy(), h),
            "robust_ema hl=25": lambda h: causal_series(g, lambda x: f_robust_ema(x, 25), h),
            "holt a=0.05,b=0.001": lambda h: causal_series(g, lambda x: f_holt(x, 0.05, 0.001), h),
        }
        for name, fn in cands.items():
            for h in HORIZONS:
                lv = fn(h)
                res.append(dict(regime=g["regime"], segment=s, model=name, h=h, **score(g, np.exp(lv))))
        gp = gp_forecast(g, g["regime"] == "high_rep", rows)
        for h in HORIZONS:
            p = np.full(g["n"], np.nan)
            p[rows] = gp[h]
            res.append(dict(regime=g["regime"], segment=s, model="gp" + (" periodic" if g["regime"] == "high_rep" else ""),
                            h=h, **score(g, np.exp(p))))
        print(f"segment {s} done", flush=True)
        pd.DataFrame(res).to_csv(out / "scores.csv", index=False)
    df = pd.DataFrame(res)
    df["model"] = df["model"].str.replace("gp periodic", "gp")
    base = df[df.model == "mean9 (baseline)"].set_index(["segment", "h"])["mae_nC"]
    df["vs_baseline_pct"] = 100 * (df["mae_nC"] / base.reindex(pd.MultiIndex.from_frame(df[["segment", "h"]])).to_numpy() - 1)
    with open(out / "summary.md", "w") as f:
        for metric in ("mae_pct", "vs_baseline_pct"):
            tab = df.pivot_table(index=["h", "model"], columns=["regime", "segment"], values=metric)
            f.write(f"\n## {metric}\n\n{tab.to_markdown(floatfmt='.2f')}\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
