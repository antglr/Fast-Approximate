# Forecasting the BCM charge N shots ahead (notebook 10_01): everything known up to shot t-N, predict y(t).
# History features at the forecast origin o = t-N, trend baselines, and a multi-horizon MLP.
import numpy as np
import pandas as pd
import torch
from torch import nn

ALPHAS = [1 / 2, 1 / 8, 1 / 32, 1 / 128, 1 / 512]   # EWMA smoothing factors (time scales ~2 ... 512 shots)
LAST_Y = 8                                          # last BCM readings used as inputs
LAST_X = 4                                          # last feature vectors used as inputs
REF_ALPHA = 1 / 32                                  # targets are residuals w.r.t. this EWMA of the BCM


def ewma(a, alpha):
    """Causal exponentially weighted mean up to and including each row (columns independently)."""
    return pd.DataFrame(a).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def segment_arrays(s):
    """Standardised (on the fit shots) BCM and features, and their EWMAs at every row."""
    ym, ysd = s["y"][s["fit"]].mean(), s["y"][s["fit"]].std()
    yz = np.clip((s["y"] - ym) / ysd, -6, 6)                       # clip BCM glitches in the inputs
    X = s["X"].copy()
    med = np.nanmedian(X[s["fit"]], axis=0)
    X = np.where(np.isnan(X), med, X)
    xz = (X - X[s["fit"]].mean(0)) / (X[s["fit"]].std(0) + 1e-12)
    return {"ym": ym, "ysd": ysd, "yz": yz, "xz": xz,
            "y_ewma": np.column_stack([ewma(yz[:, None], a)[:, 0] for a in ALPHAS]),
            "x_ewma": np.hstack([ewma(xz, a) for a in ALPHAS])}


def history_features(arr, origin, use_x=True):
    """Inputs known at the forecast origin (row `origin` included, nothing after it)."""
    y_last = arr["yz"][origin[:, None] + np.arange(-LAST_Y + 1, 1)]
    parts = [y_last, arr["y_ewma"][origin]]
    if use_x:
        parts += [arr["xz"][origin[:, None] + np.arange(-LAST_X + 1, 1)].reshape(len(origin), -1), arr["x_ewma"][origin]]
    return np.hstack(parts)


def forecast_samples(s, arr, horizons, use_x=True, min_history=LAST_Y):
    """All (t, N) samples: features at origin t-N, horizon, reference level, standardised target."""
    ref_col = ALPHAS.index(REF_ALPHA)
    out = []
    for n in horizons:
        t = np.arange(min_history + n - 1, len(arr["yz"]))
        o = t - n
        F = history_features(arr, o, use_x)
        out.append(pd.DataFrame({"t": t, "N": n, "ref": arr["y_ewma"][o, ref_col]}).assign(_F=list(F)))
    df = pd.concat(out, ignore_index=True)
    F = np.vstack(df.pop("_F").to_numpy())
    return df, F


# ---------------------------------------------------------------------------- multi-horizon MLP
class HorizonMLP(nn.Module):
    """history features + log(N) -> residual of y(t) w.r.t. the reference EWMA at the origin."""

    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, f, logn):
        return self.net(torch.cat([f, logn[:, None]], dim=1)).squeeze(-1)


def train_mlp(F, logn, target, fit, val, seed, model=None, lr=1e-3, max_epochs=150, patience=10, batch=512):
    """Train (or fine-tune if `model` is given) with MSE, early stopping on `val`; returns (model, predictions)."""
    torch.manual_seed(seed)
    model = HorizonMLP(F.shape[1]) if model is None else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    Ft, Lt, Tt = (torch.tensor(a, dtype=torch.float32) for a in (F, logn, target))
    i_fit, i_val = np.flatnonzero(fit), np.flatnonzero(val)
    rng = np.random.default_rng(seed)
    best, state, wait = np.inf, {k: v.clone() for k, v in model.state_dict().items()}, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(i_fit), max(1, len(i_fit) // batch)):
            opt.zero_grad()
            ((model(Ft[b], Lt[b]) - Tt[b]) ** 2).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lv = ((model(Ft[i_val], Lt[i_val]) - Tt[i_val]) ** 2).mean().item()
        if lv < best - 1e-6:
            best, state, wait = lv, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model, np.concatenate([model(Ft[b], Lt[b]).numpy() for b in np.array_split(np.arange(len(F)), 20)])
