# LSTM forecasters with attention readouts and autoencoder embeddings (docs/plan_attention_ae.md).
# Same inputs, target, split and scores as feature_forecast.py: inputs known after shot t - h, as a window of the last
# L shots; target log Q[t] - log(9-shot mean up to t - h); zero-initialised output, so an untrained model is the
# 9-shot mean. One model per (regime, horizon, variant), L1 loss, early stopping on validation, seeds averaged.
# python3 -I scripts/attn_forecast.py results/datasets_aligned results/attention --horizons 1,10 [--variants ...] [--L 64]
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from regimes import REGIMES, load, score  # noqa: E402
from feature_forecast import features, shift  # noqa: E402
from regime_bench import windows  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HID, CODE, CLIP = 64, 8, 5.0
VARIANTS = ["lstm", "lstm_additive", "lstm_general", "lstm_mha", "lstm_self", "lstm_vsn_additive",
            "ae_frozen", "ae_joint", "seq_ae"]


# ---------------------------------------------------------------------------- building blocks
class AdditiveAttn(nn.Module):                      # Bahdanau: v' tanh(W h_t + U q)
    def __init__(self, d):
        super().__init__()
        self.W, self.U, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d), nn.Linear(d, 1, bias=False)

    def forward(self, H, q):
        a = torch.softmax(self.v(torch.tanh(self.W(H) + self.U(q)[:, None]))[..., 0], dim=1)
        return (a[..., None] * H).sum(1), a


class GeneralAttn(nn.Module):                       # Luong general: h_t' W q / sqrt(d)
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)

    def forward(self, H, q):
        a = torch.softmax((H @ self.W(q)[..., None])[..., 0] / H.shape[-1] ** 0.5, dim=1)
        return (a[..., None] * H).sum(1), a


class MHAttn(nn.Module):                            # 4 heads, query = last state, keys = states + positions
    def __init__(self, d, L):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        self.mha = nn.MultiheadAttention(d, 4, batch_first=True)

    def forward(self, H, q):
        K = H + self.pos[:, -H.shape[1]:]
        c, a = self.mha(q[:, None], K, K, need_weights=True, average_attn_weights=True)
        return c[:, 0], a[:, 0]


class VSN(nn.Module):                               # per-step softmax weights over the inputs (TFT-like)
    def __init__(self, d):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d, 32), nn.ELU(), nn.Linear(32, d))

    def forward(self, x):
        w = torch.softmax(self.g(x), dim=-1)
        return x * w * x.shape[-1], w


class AE(nn.Module):                                # per-shot autoencoder: inputs -> CODE -> inputs
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 32), nn.ELU(), nn.Linear(32, CODE))
        self.dec = nn.Sequential(nn.Linear(CODE, 32), nn.ELU(), nn.Linear(32, d))


class Forecaster(nn.Module):
    def __init__(self, variant, d, L):
        super().__init__()
        self.variant = variant
        self.vsn = VSN(d) if variant == "lstm_vsn_additive" else None
        self.ae = AE(d) if variant.startswith("ae_") else None
        d_in = CODE if self.ae is not None else d
        self.lstm = nn.LSTM(d_in, HID, batch_first=True)
        self.attn = {"lstm_additive": AdditiveAttn, "lstm_vsn_additive": AdditiveAttn, "ae_frozen": AdditiveAttn,
                     "ae_joint": AdditiveAttn, "lstm_general": GeneralAttn}.get(variant)
        self.attn = self.attn(HID) if self.attn else (MHAttn(HID, L) if variant == "lstm_mha" else None)
        self.block = nn.TransformerEncoderLayer(HID, 4, 128, dropout=0.1, batch_first=True) \
            if variant == "lstm_self" else None
        self.head = nn.Linear(2 * HID if self.attn is not None else HID, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        extra = {}
        if self.vsn is not None:
            x, extra["vsn"] = self.vsn(x)
        if self.ae is not None:
            z = self.ae.enc(x)
            extra["recon"] = self.ae.dec(z)
            x = z
        H, _ = self.lstm(x)
        q = H[:, -1]
        if self.block is not None:
            H = self.block(H)
            q = H[:, -1]
        if self.attn is not None:
            c, extra["attn"] = self.attn(H, q)
            q = torch.cat([c, q], -1)
        return self.head(q)[:, 0], extra


class SeqAE(nn.Module):                             # LSTM sequence autoencoder; forecast head on the code
    def __init__(self, d, L):
        super().__init__()
        self.enc = nn.LSTM(d, HID, batch_first=True)
        self.code = nn.Linear(HID, 16)
        self.dec_in = nn.Linear(16, HID)
        self.dec = nn.LSTM(HID, HID, batch_first=True)
        self.out = nn.Linear(HID, d)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.L = L

    def encode(self, x):
        return self.code(self.enc(x)[0][:, -1])

    def reconstruct(self, z, L):
        return self.out(self.dec(self.dec_in(z)[:, None].expand(-1, L, -1))[0])

    def forward(self, x):
        return self.head(self.encode(x))[:, 0], {}


# ---------------------------------------------------------------------------- training
def batches(n, bs, shuffle):
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def pretrain_ae(model, Xstep, epochs=30):
    """Per-shot AE on the fit rows' feature vectors (reconstruction MSE)."""
    ae = model.ae if isinstance(model, Forecaster) else None
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    for _ in range(epochs):
        for b in batches(len(Xstep), 1024, True):
            opt.zero_grad()
            nn.functional.mse_loss(ae.dec(ae.enc(Xstep[b])), Xstep[b]).backward()
            opt.step()
    with torch.no_grad():
        return nn.functional.mse_loss(ae.dec(ae.enc(Xstep)), Xstep).item()


def pretrain_seq(model, xf, epochs=15):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for b in batches(len(xf), 512, True):
            opt.zero_grad()
            nn.functional.mse_loss(model.reconstruct(model.encode(xf[b]), xf.shape[1]), xf[b]).backward()
            opt.step()
    with torch.no_grad():
        b = torch.arange(min(4096, len(xf)), device=DEVICE)
        return nn.functional.mse_loss(model.reconstruct(model.encode(xf[b]), xf.shape[1]), xf[b]).item()


def fit_variant(variant, xf, tf, xv, tv, L, seed, max_epochs=150, patience=15):
    torch.manual_seed(seed)
    d = xf.shape[2]
    model = (SeqAE(d, L) if variant == "seq_ae" else Forecaster(variant, d, L)).to(DEVICE)
    info = {}
    if variant.startswith("ae_"):
        info["ae_recon_mse"] = pretrain_ae(model, xf[:, -1])
        if variant == "ae_frozen":
            for p in model.ae.parameters():
                p.requires_grad_(False)
    if variant == "seq_ae":
        info["seq_recon_mse"] = pretrain_seq(model, xf)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 1e-3, weight_decay=1e-5)
    best, state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        for b in batches(len(xf), 512, True):
            opt.zero_grad()
            out, extra = model(xf[b])
            loss = nn.functional.l1_loss(out, tf[b])
            if variant == "ae_joint":
                loss = loss + 0.5 * nn.functional.mse_loss(extra["recon"], xf[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([nn.functional.l1_loss(model(xv[b])[0], tv[b]).item() for b in batches(len(xv), 4096, False)])
        if v < best - 1e-5:
            best, state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    info.update(epochs=epoch + 1, val_l1=best)
    return model.eval(), info


def predict(model, x):
    outs, attn, vsn = [], [], []
    with torch.no_grad():
        for b in batches(len(x), 4096, False):
            o, extra = model(x[b])
            outs.append(o.cpu().numpy())
            if "attn" in extra:
                attn.append(extra["attn"].cpu().numpy())
            if "vsn" in extra:
                vsn.append(extra["vsn"].mean(1).cpu().numpy())
    return np.concatenate(outs), (np.concatenate(attn) if attn else None), (np.concatenate(vsn) if vsn else None)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    ap.add_argument("--horizons", default="1,5,10,20,50")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"L{a.L}_h{a.horizons.replace(',', '-')}"
    segs, names = load(a.datasets, robust=5)
    F = {s: features(g, names, late_fix=False) for s, g in segs.items()}   # aligned data: lab / PGLA on their row
    in_names = ([f"hist_{k}" for k in ("last", "mean3", "mean50", "mean200", "robust_ema", "holt", "good")]
                + [f"fast_{n}" for n in names] + [f"slow_{n}" for n in names] + ["missing", "camera_spike"])
    L = a.L
    rows, saved, interp = [], {}, {}
    for h in map(int, a.horizons.split(",")):
        for regime, ss in REGIMES.items():
            G = [segs[s] for s in ss if s in segs]
            base = []
            for g in G:
                b = shift(F[g["segment"]][0][:, None], h)[:, 0]
                b[:h] = np.nan
                base.append(b)
            T = [np.nan_to_num(g["logy"] - b) for g, b in zip(G, base)]
            X = [np.c_[shift(F[g["segment"]][1], h), shift(F[g["segment"]][2], h)] for g in G]
            ok = lambda m: [mm & (np.arange(g["n"]) >= h + 9) for mm, g in zip(m, G)]  # noqa: E731
            fit, val = ok([g["fit"] for g in G]), ok([g["val"] for g in G])
            allfit = np.concatenate([x[f] for x, f in zip(X, fit)])
            mu, sd = allfit.mean(0), allfit.std(0) + 1e-9
            tt = np.concatenate([t[f] for t, f in zip(T, fit)])
            tmu, tsd = tt.mean(), tt.std() + 1e-9
            W = [torch.tensor(windows(np.clip((x - mu) / sd, -CLIP, CLIP), L).astype(np.float32), device=DEVICE)
                 for x in X]                                   # row t: inputs of rows t-h-L+1 .. t-h (shifted)
            Tn = [torch.tensor(((t - tmu) / tsd).astype(np.float32), device=DEVICE) for t in T]
            cat = lambda ms: (torch.cat([w[torch.tensor(m, device=DEVICE)] for w, m in zip(W, ms)]),  # noqa: E731
                              torch.cat([t[torch.tensor(m, device=DEVICE)] for t, m in zip(Tn, ms)]))
            xf, tf = cat(fit)
            xv, tv = cat(val)
            for variant in a.variants.split(","):
                t0 = time.time()
                preds, attns, vsns, infos = [], [], [], []
                for seed in range(a.seeds):
                    model, info = fit_variant(variant, xf, tf, xv, tv, L, seed)
                    infos.append(info)
                    pr = [predict(model, w) for w in W]
                    preds.append([p[0] for p in pr])
                    attns.append([p[1] for p in pr])
                    vsns.append([p[2] for p in pr])
                sec = (time.time() - t0) / a.seeds
                for i, g in enumerate(G):
                    o = np.mean([p[i] for p in preds], 0) * tsd + tmu
                    pred = np.exp(base[i] + o)
                    saved[f"h{h}|seg{g['segment']}|{variant}"] = pred.astype(np.float32)
                    if g["test"].sum() >= 50:
                        rows.append(dict(h=h, L=L, regime=regime, segment=g["segment"], model=variant, sec_per_fit=sec,
                                         epochs=np.mean([x["epochs"] for x in infos]),
                                         **{k: np.mean([x[k] for x in infos]) for k in infos[0]
                                            if k.endswith("mse")}, **score(g, pred)))
                    te = g["test"]
                    if attns[0][i] is not None:                # mean attention per lag on the test shots
                        interp[f"attn|h{h}|seg{g['segment']}|{variant}"] = \
                            np.mean([a_[i][te] for a_ in attns], 0).mean(0).tolist()
                    if vsns[0][i] is not None:
                        interp[f"vsn|h{h}|seg{g['segment']}|{variant}"] = \
                            dict(zip(in_names, np.mean([v_[i][te] for v_ in vsns], 0).mean(0).tolist()))
                print(f"h={h:3d} {regime:12s} {variant:18s} {sec:6.1f}s/fit", flush=True)
                pd.DataFrame(rows).to_csv(out / f"scores_{tag}.csv", index=False)
            del W, xf, xv
            torch.cuda.empty_cache()
    np.savez_compressed(out / f"predictions_{tag}.npz", **saved)
    (out / f"interpret_{tag}.json").write_text(json.dumps(interp))
    print("done", tag)


if __name__ == "__main__":
    main()
