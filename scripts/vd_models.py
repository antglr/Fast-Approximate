# Building blocks of the two-stage virtual diagnostic (notebooks 08_05, 08_06).
# Stage 1 = trend (model on the raw features, averaged over the previous K shots of its own predictions);
# stage 2 = the shot's deviation from the previous K shots, from the same deviation of the features.
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 51                 # shots: trend window and fluctuation smoothing
CUT = 0.7              # BCM lost after this fraction of each segment
VAL_FRACTION = 0.2     # last part of the training shots: validation block
NO_BEAM = 0.2          # shots below this fraction of the training median are not learned from, nor scored
PC = 1000              # 1 nC = 1000 pC
WINDOWS = [None, 4000, 2000, 1000]


# ---------------------------------------------------------------------------- data
def load_segments(datasets, min_valid=500):
    """{segment: dict} with the features, target, chronological masks (fit / val / train / test) and deviations."""
    manifest = json.loads((Path(datasets) / "manifest.json").read_text())
    out = {}
    for info in manifest["segments"]:
        z = np.load(Path(datasets) / info["file"])
        X, y, v = z["X"], z["y"], z["valid"]
        if v.sum() < min_valid:
            continue
        n = len(y)
        rows = np.arange(n)
        cut = int(CUT * n)
        val_row = int((1 - VAL_FRACTION) * cut)
        good = v & (y > NO_BEAM * np.median(y[v & (rows < cut)]))
        out[info["segment"]] = {
            "run": info["run"], "X": X, "y": y, "rows": rows, "cut": cut, "val_row": val_row,
            "fit": good & (rows < val_row), "val": good & (rows >= val_row) & (rows < cut),
            "train": good & (rows < cut), "test": good & (rows >= cut),
            "X_dev": X - past_mean(X), "y_dev": y - past_mean(y[:, None])[:, 0],
        }
    return out, manifest["features"]


def past_mean(a, k=K):
    """Mean of the previous k rows (causal, excludes the current one)."""
    return pd.DataFrame(a).shift(1).rolling(k, min_periods=1).mean().bfill().to_numpy()


def smooth(a):
    return pd.Series(a).rolling(K, center=True, min_periods=1).mean().to_numpy()


def fast(a):
    return a - smooth(a)


def standardise(a, ref):
    return np.nan_to_num((a - np.nanmean(a[ref], 0)) / (np.nanstd(a[ref], 0) + 1e-12))


# ---------------------------------------------------------------------------- classic models
class LinearPlusTrees(BaseEstimator, RegressorMixin):
    """Linear fit first, then trees on its residual (keeps the linear extrapolation)."""

    def __init__(self, linear, trees):
        self.linear, self.trees = linear, trees

    def fit(self, X, y):
        self.linear_ = clone(self.linear).fit(X, y)
        self.trees_ = clone(self.trees).fit(X, y - self.linear_.predict(X))
        return self

    def predict(self, X):
        return self.linear_.predict(X) + self.trees_.predict(X)


class BoundedRidge(BaseEstimator, RegressorMixin):
    """Ridge with sign constraints on chosen coefficients: positive = column indices forced to a coefficient >= 0.
    Alpha from an unconstrained RidgeCV; then the constrained least squares on the augmented system."""

    def __init__(self, positive=()):
        self.positive = positive

    def fit(self, X, y):
        self.imp_ = SimpleImputer(strategy="median").fit(X)
        self.sc_ = StandardScaler().fit(self.imp_.transform(X))
        Z = self.sc_.transform(self.imp_.transform(X))
        alpha = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z, y).alpha_
        self.mean_ = y.mean()
        lo = np.full(Z.shape[1], -np.inf)
        lo[list(self.positive)] = 0.0
        A = np.vstack([Z, np.sqrt(alpha) * np.eye(Z.shape[1])])
        b = np.r_[y - self.mean_, np.zeros(Z.shape[1])]
        self.coef_ = lsq_linear(A, b, bounds=(lo, np.full(Z.shape[1], np.inf))).x
        return self

    def predict(self, X):
        return self.sc_.transform(self.imp_.transform(X)) @ self.coef_ + self.mean_


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def hgb(**kw):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=0, **kw)


def stage1_default():
    return LinearPlusTrees(ridge(), hgb())


def stage1(s, train, end_row, window, factory=stage1_default, X=None):
    """Stage-1 model on the training shots (only the last `window` before end_row if given) -> prediction, trend."""
    X = s["X"] if X is None else X
    use = train & (s["rows"] >= (end_row - window if window else 0))
    p = factory().fit(X[use], s["y"][use]).predict(X)
    return p, past_mean(p[:, None])[:, 0]


def choose_window(s, factory=stage1_default, X=None):
    """Stage-1 window with the smallest trend error on the validation block (fitted on the shots before it)."""
    errs = {}
    for w in WINDOWS:
        p, _ = stage1(s, s["fit"], s["val_row"], w, factory, X)
        errs[w] = np.sqrt(np.mean((smooth(s["y"][s["val"]]) - smooth(p[s["val"]])) ** 2))
    return min(errs, key=errs.get)


# ---------------------------------------------------------------------------- neural stage 2
def loss_ccc(p, t):
    pm, tm = p.mean(), t.mean()
    return 1 - 2 * ((p - pm) * (t - tm)).mean() / (p.var(unbiased=False) + t.var(unbiased=False) + (pm - tm) ** 2)


def loss_mse(p, t):
    return ((p - t) ** 2).mean()


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def monotone_penalty(model, xb, cols):
    """Mean amount by which the output DEcreases when the inputs in `cols` increase (0 if monotone increasing)."""
    xb = xb.clone().requires_grad_(True)
    grad = torch.autograd.grad(model(xb).sum(), xb, create_graph=True)[0][:, cols]
    return torch.relu(-grad).mean()


def train_mlp(loss, Xin, target, fit, val, seed, model=None, lr=1e-3, mono_cols=None, mono_weight=1.0,
              max_epochs=200, patience=15):
    """Train (or fine-tune, if `model` is given) on `fit` rows, early-stop on `val` rows; returns (model, predictions)."""
    torch.manual_seed(seed)
    model = MLP(Xin.shape[1]) if model is None else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, Tt = torch.tensor(Xin, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    i_fit, i_val = np.flatnonzero(fit), np.flatnonzero(val)
    rng = np.random.default_rng(seed)
    best, state, wait = np.inf, {k: v.clone() for k, v in model.state_dict().items()}, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(i_fit), max(1, len(i_fit) // 256)):
            opt.zero_grad()
            l = loss(model(Xt[b]), Tt[b])
            if mono_cols is not None:
                l = l + mono_weight * monotone_penalty(model, Xt[b], mono_cols)
            l.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lv = loss(model(Xt[i_val]), Tt[i_val]).item()
        if lv < best - 1e-5:
            best, state, wait = lv, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model, model(Xt).numpy()


def stage2_inputs(s):
    """Inputs of the neural stage 2 (standardised on the fit shots): [feature deviations, raw features]."""
    return np.hstack([standardise(s["X_dev"], s["fit"]), standardise(s["X"], s["fit"])])


def stage2(s, loss=loss_ccc, seeds=range(6), **kw):
    """Neural stage 2 trained on this segment: predicted deviation from the previous K shots, for every row."""
    Xin = stage2_inputs(s)
    tm, ts = s["y_dev"][s["fit"]].mean(), s["y_dev"][s["fit"]].std()
    out = np.mean([train_mlp(loss, Xin, (s["y_dev"] - tm) / ts, s["fit"], s["val"], sd, **kw)[1] for sd in seeds], 0)
    out = out * ts + tm
    return out + np.mean(s["y_dev"][s["fit"]] - out[s["fit"]])


# ---------------------------------------------------------------------------- scores
def scores_pc(y, p):
    e = PC * (p - y)
    slow = smooth(e)
    fy, fp = fast(y), fast(p)
    return {"std [pC]": e.std(), "bias [pC]": e.mean(), "slow [pC]": slow.std(), "fast [pC]": (e - slow).std(),
            "fluct CCC": 2 * np.cov(fy, fp, bias=True)[0, 1] / (fy.var() + fp.var() + (fy.mean() - fp.mean()) ** 2)}
