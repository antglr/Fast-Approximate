# Phase 2 of docs/workplan_adnan_01.md: jitter models per regime, a sparse GP, and ensembles chosen on validation.
#   base models (pooled per regime, level + jitter framing): ridge, XGBoost, GRU (L = 16), ridge + GRU, sparse GP (ARD)
#   ensembles: equal-weight mean, non-negative stacking weights fitted on the validation block
#   intervals: the GP's own 90 % interval, and a split-conformal one around the stack (validation residual quantiles)
# python3 -I scripts/regime_phase2.py results/datasets results/phase2 --robust 5 --filters results/decompose_robust/choice.json
import argparse
import json
import sys
import time
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from regimes import REGIMES, NOISE_FLOOR_PCT, load, score  # noqa: E402
from regime_bench import DEVICE, fit_tabular, fit_torch, inputs  # noqa: E402


# ---------------------------------------------------------------------------- sparse variational GP
class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, Z):
        q = gpytorch.variational.CholeskyVariationalDistribution(len(Z))
        s = gpytorch.variational.VariationalStrategy(self, Z, q, learn_inducing_locations=True)
        super().__init__(s)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Z.shape[1]))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_gp(X, t, fit, val, inducing=256, epochs=60):
    """Returns predict(X) -> (mean, std) in target units, and the ARD lengthscales (small = relevant input)."""
    keep = X[fit].std(0) > 1e-6                                 # constant columns: unbounded lengthscale, singular K_ZZ
    mu, sd = X[fit][:, keep].mean(0), X[fit][:, keep].std(0)
    tmu, tsd = t[fit].mean(), t[fit].std() + 1e-9
    f64 = lambda a: torch.tensor((a[:, keep] - mu) / sd, dtype=torch.float64, device=DEVICE).clamp(-5, 5)  # noqa: E731
    xf, tf = f64(X[fit]), torch.tensor((t[fit] - tmu) / tsd, dtype=torch.float64, device=DEVICE)
    xv, tv = f64(X[val]), torch.tensor((t[val] - tmu) / tsd, dtype=torch.float64, device=DEVICE)
    torch.manual_seed(0)
    Z = xf[torch.randperm(len(xf), device=DEVICE)[:inducing]].clone()
    Z += 0.01 * torch.randn_like(Z)                            # many rows repeat (missing = 0): keep K_ZZ invertible
    model = SVGP(Z).to(DEVICE).double()
    lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE).double()
    opt = torch.optim.Adam([*model.parameters(), *lik.parameters()], lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(xf))
    best, state = np.inf, None
    for epoch in range(epochs):
        model.train(), lik.train()
        perm = torch.randperm(len(xf), device=DEVICE)
        for i in range(0, len(xf), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            (-mll(model(xf[b]), tf[b])).backward()
            opt.step()
        model.eval(), lik.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            v = (model(xv).mean - tv).abs().mean().item()
        if v < best:
            best, state = v, ({k: p.clone() for k, p in model.state_dict().items()},
                              {k: p.clone() for k, p in lik.state_dict().items()})
    model.load_state_dict(state[0]), lik.load_state_dict(state[1])
    model.eval(), lik.eval()

    def predict(Xn):
        x = f64(Xn)
        ms, ss = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(x), 4096):
                p = lik(model(x[i:i + 4096]))
                ms.append(p.mean.cpu().numpy()), ss.append(p.stddev.cpu().numpy())
        return np.concatenate(ms) * tsd + tmu, np.concatenate(ss) * tsd
    ls = np.full(len(keep), np.inf)
    ls[keep] = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    return predict, ls


# ---------------------------------------------------------------------------- per regime
def regime_models(G, names, quick):
    """Jitter predictions of every base model on every row of every segment of the regime."""
    XT = [inputs(g, "jitter") for g in G]
    X = np.concatenate([x for x, _ in XT])
    t = np.nan_to_num(np.concatenate([t for _, t in XT]))
    fit = np.concatenate([g["fit"] for g in G])
    val = np.concatenate([g["val"] for g in G])
    cut = np.cumsum([0] + [g["n"] for g in G])
    split = lambda a: [a[cut[i]:cut[i + 1]] for i in range(len(G))]  # noqa: E731
    preds, extra = {}, {}
    t0 = time.time()
    for name in ("ridge", "xgboost"):
        preds[name] = split(fit_tabular(name, X, t, fit, val, quick)(X))
    gru = fit_torch("gru", 16, [x for x, _ in XT], [np.nan_to_num(t) for _, t in XT],
                    [g["fit"] for g in G], [g["val"] for g in G], quick)
    preds["gru"] = gru([x for x, _ in XT])
    # ridge + GRU: the GRU learns the ridge's residual, so the ridge keeps extrapolating where the GRU saturates
    res = [np.nan_to_num(tt) - r for (_, tt), r in zip(XT, preds["ridge"])]
    rg = fit_torch("gru", 16, [x for x, _ in XT], res, [g["fit"] for g in G], [g["val"] for g in G], quick)
    preds["ridge+gru"] = [r + o for r, o in zip(preds["ridge"], rg([x for x, _ in XT]))]
    gp, ls = fit_gp(X, t, fit, val, epochs=15 if quick else 60)
    m, s = gp(X)
    preds["gp"], extra["gp_std"] = split(m), split(s)
    extra["gp_lengthscales"] = dict(zip(names + ["missing", "camera_spike"], map(float, ls)))
    print(f"  base models {time.time() - t0:.0f}s", flush=True)
    return preds, extra


def stack(G, preds):
    """Equal mean and non-negative weights fitted on the validation shots of the regime."""
    base = list(preds)
    A = np.concatenate([np.column_stack([preds[b][i] for b in base])[g["val"]] for i, g in enumerate(G)])
    y = np.concatenate([np.nan_to_num(g["jitter"])[g["val"]] for g in G])
    w, _ = nnls(A, y)
    preds["mean_ensemble"] = [np.mean([preds[b][i] for b in base], 0) for i in range(len(G))]
    preds["stack_nnls"] = [np.column_stack([preds[b][i] for b in base]) @ w for i in range(len(G))]
    return dict(zip(base, map(float, w)))


def main():
    gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6).__enter__()
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    ap.add_argument("--robust", type=float, default=5)
    ap.add_argument("--filters")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    filters = None
    if a.filters:
        from decompose import GRID, causal
        fns = {(n, p): fn for n, fn, p in GRID}
        filters = {c["regime"]: causal(fns[c["filter"], c["param"]]) for c in json.loads(Path(a.filters).read_text())}
    segs, names = load(a.datasets, fast_filter=filters, robust=a.robust or None)
    rows, info, saved = [], {}, {}
    for regime, ss in REGIMES.items():
        G = [segs[s] for s in ss if s in segs]
        print(regime, flush=True)
        preds, extra = regime_models(G, names, a.quick)
        weights = stack(G, preds)
        # split-conformal 90 % interval around the stack: validation quantiles of the log residual
        r_val = np.concatenate([(np.nan_to_num(g["jitter"]) - preds["stack_nnls"][i])[g["val"]] for i, g in enumerate(G)])
        q_lo, q_hi = np.quantile(r_val, [0.05, 0.95])
        info[regime] = {"stack_weights": weights, "gp_lengthscales": extra["gp_lengthscales"],
                        "conformal_q": [float(q_lo), float(q_hi)]}
        for i, g in enumerate(G):
            if g["test"].sum() < 50:
                continue
            te = g["test"]
            j = g["jitter"]
            for name in ["level_only"] + list(preds):
                p = np.zeros(g["n"]) if name == "level_only" else preds[name][i]
                saved[f"seg{g['segment']}|{name}"] = np.exp(g["level"] + p).astype(np.float32)
                sc = score(g, np.exp(g["level"] + p))
                row = dict(regime=regime, segment=g["segment"], model=name, **sc)
                row["gap_to_floor"] = sc["resid_rms_pct"] / NOISE_FLOOR_PCT.get(g["segment"], np.nan)
                if name == "gp":
                    s = extra["gp_std"][i]
                    row["cover90"] = float(np.mean(np.abs(j[te] - p[te]) < 1.645 * s[te]))
                    row["width90_pct"] = float(100 * np.mean(2 * 1.645 * s[te]))
                if name == "stack_nnls":
                    row["cover90"] = float(np.mean((j[te] - p[te] > q_lo) & (j[te] - p[te] < q_hi)))
                    row["width90_pct"] = float(100 * (q_hi - q_lo))
                rows.append(row)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    (out / "info.json").write_text(json.dumps(info, indent=1))
    np.savez_compressed(out / "predictions.npz", **saved)      # same-shot charge predictions (nC), every row
    df = pd.DataFrame(rows)
    with open(out / "summary.md", "w") as f:
        for metric in ("mae_pct", "jitter_explained", "gap_to_floor", "cover90", "width90_pct"):
            tab = df.pivot_table(index="model", columns=["regime", "segment"], values=metric)
            f.write(f"\n## {metric}\n\n{tab.to_markdown(floatfmt='.3f')}\n")
        f.write("\n## stacking weights, GP lengthscales, conformal quantiles\n\n```\n" + json.dumps(info, indent=1) + "\n```\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
