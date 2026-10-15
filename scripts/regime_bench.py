# Phase 0 of docs/workplan_adnan_01.md: every model family under one split and one score, per regime, two framings.
#   absolute: log Q from the shot's own readings, one model per segment (the 08_simple_models setting)
#   jitter:   log Q - trailing 50-shot BCM level, from the fast parts of the readings, one model per regime
# python3 -I scripts/regime_bench.py results/datasets results/regime_bench [--quick]
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from regimes import REGIMES, NOISE_FLOOR_PCT, load, score  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = 5.0


# ---------------------------------------------------------------------------- inputs
def inputs(seg, framing):
    if framing == "absolute":
        x = np.where(np.isfinite(seg["X"]), seg["X"], np.nanmedian(seg["X"][seg["fit"]], 0))
        return np.c_[x, seg["missing"]], seg["logy"]
    return np.c_[seg["Xfast"], seg["missing"]], seg["jitter"]


def to_pred(seg, framing, out):
    return np.exp(out if framing == "absolute" else seg["level"] + out)


# ---------------------------------------------------------------------------- tabular models
def tabular(name, quick):
    n_trees = 100 if quick else 400
    return {
        "linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "random_forest": lambda: RandomForestRegressor(n_trees, min_samples_leaf=20, max_features=0.5, n_jobs=-1,
                                                       random_state=0),
        "xgboost": lambda: XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8,
                                        colsample_bytree=0.8, early_stopping_rounds=50, tree_method="hist",
                                        device="cuda" if DEVICE == "cuda" else "cpu", random_state=0),
    }[name]()


def fit_tabular(name, X, t, fit, val, quick):
    m = tabular(name, quick)
    if name == "xgboost":
        m.fit(X[fit], t[fit], eval_set=[(X[val], t[val])], verbose=False)
        m.set_params(device="cpu")                             # predict on host arrays
    else:
        m.fit(X[fit], t[fit])
    return m.predict


# ---------------------------------------------------------------------------- sequence models (torch)
class Net(nn.Module):
    def __init__(self, kind, d, L, hidden=32):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.body = nn.Sequential(nn.Linear(d * L, 64), nn.ReLU(), nn.Linear(64, hidden), nn.ReLU())
        else:
            self.body = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](d, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)                       # untrained model = 0 = the baseline of the framing
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if self.kind == "mlp":
            return self.head(self.body(x.flatten(1)))[:, 0]
        return self.head(self.body(x)[0][:, -1])[:, 0]


def windows(X, L):
    """(n, L, d): row n and the L-1 rows before it in the same segment, zero-padded at the start."""
    pad = np.r_[np.zeros((L - 1, X.shape[1])), X]
    return np.lib.stride_tricks.sliding_window_view(pad, (L, X.shape[1]))[:, 0]


def fit_torch(kind, L, segs_X, segs_t, fits, vals, quick, seeds=3):
    """segs_*: lists over the segments of one pool. Returns predict(list of X) -> list of outputs."""
    allfit = np.concatenate([x[f] for x, f in zip(segs_X, fits)])
    mu, sd = allfit.mean(0), allfit.std(0) + 1e-9
    tmu, tsd = np.concatenate([t[f] for t, f in zip(segs_t, fits)]).mean(), \
        np.concatenate([t[f] for t, f in zip(segs_t, fits)]).std() + 1e-9
    prep = lambda x: windows(np.clip((x - mu) / sd, -CLIP, CLIP), L).astype(np.float32)  # noqa: E731
    W = [prep(x) for x in segs_X]
    T = [((t - tmu) / tsd).astype(np.float32) for t in segs_t]
    tr = lambda masks: (torch.tensor(np.concatenate([w[m] for w, m in zip(W, masks)]), device=DEVICE),  # noqa: E731
                        torch.tensor(np.concatenate([t[m] for t, m in zip(T, masks)]), device=DEVICE))
    xf, tf = tr(fits)
    xv, tv = tr(vals)
    nets = []
    for seed in range(1 if quick else seeds):
        torch.manual_seed(seed)
        net = Net(kind, W[0].shape[2], L).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(40 if quick else 300):
            net.train()
            perm = torch.randperm(len(xf), device=DEVICE)
            for i in range(0, len(xf), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                nn.functional.l1_loss(net(xf[b]), tf[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = nn.functional.l1_loss(net(xv), tv).item()
            if v < best - 1e-5:
                best, best_state, bad = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 20:
                    break
        net.load_state_dict(best_state)
        nets.append(net.eval())

    def predict(segs_X_new):
        outs = []
        for x in segs_X_new:
            w = torch.tensor(prep(x), device=DEVICE)
            with torch.no_grad():
                o = np.mean([torch.cat([n(w[i:i + 8192]) for i in range(0, len(w), 8192)]).cpu().numpy()
                             for n in nets], 0)
            outs.append(o * tsd + tmu)
        return outs
    return predict


# ---------------------------------------------------------------------------- bench
MODELS = ["linear", "ridge", "random_forest", "xgboost", "mlp", "rnn", "gru", "lstm"]
SEQ_L = {"mlp": 1, "rnn": 16, "gru": 16, "lstm": 16}


def run(segs, framing, pool, models, quick):
    rows = []
    pools = ([[s] for s in segs if segs[s]["fit"].sum() > 300] if pool == "segment"
             else [[s for s in ss if s in segs] for ss in REGIMES.values()])
    for group in pools:
        G = [segs[s] for s in group]
        XT = [inputs(g, framing) for g in G]
        for name in ["level_only"] + models:
            t0 = time.time()
            if name == "level_only":                            # the framing's baseline
                outs = [np.full(g["n"], np.nanmean(g["logy"][g["train"]])) if framing == "absolute"
                        else np.zeros(g["n"]) for g in G]
            elif name in SEQ_L:
                pr = fit_torch(name, SEQ_L[name], [x for x, _ in XT], [np.nan_to_num(t) for _, t in XT],
                               [g["fit"] for g in G], [g["val"] for g in G], quick)
                outs = pr([x for x, _ in XT])
            else:
                X = np.concatenate([x for x, _ in XT])
                t = np.nan_to_num(np.concatenate([t for _, t in XT]))
                fit = np.concatenate([g["fit"] for g in G])
                val = np.concatenate([g["val"] for g in G])
                pr = fit_tabular(name, X, t, fit, val, quick)
                outs = [pr(x) for x, _ in XT]
            for g, o in zip(G, outs):
                if g["test"].sum() < 50:
                    continue
                sc = score(g, to_pred(g, framing, o))
                floor = NOISE_FLOOR_PCT.get(g["segment"], np.nan)
                rows.append(dict(framing=framing, pool=pool, regime=g["regime"], segment=g["segment"], model=name,
                                 floor_pct=floor, gap_to_floor=sc["resid_rms_pct"] / floor, sec=time.time() - t0, **sc))
            print(f"{framing:8s} {pool:7s} {str(group):12s} {name:14s} {time.time() - t0:6.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--robust", type=float, default=0, help="k: MAD outlier handling of inputs and training targets")
    ap.add_argument("--filters", help="choice.json of decompose.py: level filter per regime")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("device", DEVICE, flush=True)
    filters = None
    if a.filters:
        from decompose import GRID, causal
        fns = {(n, p): fn for n, fn, p in GRID}
        filters = {c["regime"]: causal(fns[c["filter"], c["param"]]) for c in json.loads(Path(a.filters).read_text())}
        print("filters", Path(a.filters).read_text(), flush=True)
    segs, _ = load(a.datasets, fast_filter=filters, robust=a.robust or None)
    models = a.models.split(",")
    rows = []
    for framing, pool in [("absolute", "segment"), ("jitter", "segment"), ("jitter", "regime")]:
        rows += run(segs, framing, pool, models, a.quick)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    df = pd.DataFrame(rows)
    order = ["level_only"] + models
    with open(out / "summary.md", "w") as f:
        for metric, fmt in [("mae_pct", "{:.2f}"), ("r2", "{:.2f}"), ("jitter_explained", "{:.2f}"),
                            ("gap_to_floor", "{:.2f}"), ("slow_share", "{:.2f}")]:
            f.write(f"\n## {metric}\n\n")
            for (framing, pool), d in df.groupby(["framing", "pool"], sort=False):
                tab = d.pivot_table(index="model", columns=["regime", "segment"], values=metric).reindex(order)
                f.write(f"### {framing} / {pool}\n\n{tab.to_markdown(floatfmt='.3f')}\n\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
