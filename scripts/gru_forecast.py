# Next-shot GRU forecaster (the sequence model of the Hard tier), compared with the hybrid and the combined forecaster.
# Input: the last L shots, each with
#   q      the charge if the hybrid used the shot (else 0, with skip = 1)
#   level  the hybrid level after that shot
#   lab, clex, step  the laser meters and the beam-loading-step charge
# Charges are divided by the hybrid level after the newest shot, the meters by their window mean, so one model serves
# 0.3 to 20 nC. Output: the next shot relative to that level (0 = the hybrid), trained with an L1 loss (the metric is
# MAE) on every good next shot of each run's first 90 %; the last 10 % of those are the validation set for early
# stopping. Each variant is trained with several seeds and their predictions averaged. Scored on each run's last 10 %
# with score_forecast (h=1). To try more inputs, add channels in windows().
# Usage: python -I gru_forecast.py <tables dir> <out dir> <run> [<run> ...]
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from combined_forecast import ALPHAS, WMAX, Z, prepare, score
from nbfuncs import TARGET, cut_shot, rolling_baseline, scored_shots

L = 64                  # shots of history
H = 32                  # GRU hidden size
SEEDS = (0, 1, 2)
BATCH, EPOCHS, PATIENCE, LR = 256, 40, 5, 1e-3
CHANNELS = ['q', 'skip', 'level', 'lab', 'clex', 'step']
CHARGE_ONLY = 3         # the first 3 channels


def windows(r):
    # -> X (shots, L, channels) for the window ending at each shot n >= L-1, and y = the next good shot / level - 1
    p, x, used = r['p'].values, r['x'], r['used']
    raw = {'q': np.where(used, r['df'][TARGET].values, np.nan), 'skip': (~used).astype(float), 'level': p,
           'lab': x.lab.values, 'clex': x.clex.values, 'step': x.step.values}
    W = {k: np.lib.stride_tricks.sliding_window_view(v, L) for k, v in raw.items()}     # row j = shots j .. j+L-1
    pn = p[L - 1:, None]
    f = {'q': W['q'] / pn - 1, 'skip': W['skip'], 'level': W['level'] / pn - 1,
         'lab': W['lab'] / W['lab'].mean(1, keepdims=True) - 1, 'clex': W['clex'] / W['clex'].mean(1, keepdims=True) - 1,
         'step': (W['step'] - W['step'].mean(1, keepdims=True)) / pn}          # the step has an offset at low charge
    # clip: a feature gone wild (as the step did at 0.3 nC) must not dominate
    X = np.nan_to_num(np.stack([f[c] for c in CHANNELS], -1), nan=0, posinf=0, neginf=0).clip(-5, 5).astype(np.float32)
    y = r['q'].shift(-1).values[L - 1:] / p[L - 1:] - 1
    j = len(X) // 2
    assert np.isclose(X[j, -1, 0], (raw['q'][j + L - 1] / p[j + L - 1] - 1) if used[j + L - 1] else 0, atol=1e-6)
    return X, y


class GRU(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', scale)                # per-channel training std: the relative charges are ~0.03
        self.gru, self.out = torch.nn.GRU(len(scale), H, batch_first=True), torch.nn.Linear(H, 1)

    def forward(self, X):
        return self.out(self.gru(X / self.scale)[0][:, -1]).squeeze(-1)


def predict(model, X):
    with torch.no_grad():
        return torch.cat([model(b) for b in torch.from_numpy(X).split(4096)]).numpy()


def fit(R, runs, c, seed):
    # train on the good next shots of the runs' first 90 %, early-stop on the last 10 % of those
    tr, va = [], []
    for run in runs:
        r = R[run]
        shot = r['df'].index.values[L - 1:]
        ok = np.flatnonzero(np.isfinite(r['wy']) & (shot + 1 < cut_shot(r['df'])))
        k = int(0.9 * len(ok))
        tr.append((r['wX'][ok[:k], :, :c], r['wy'][ok[:k]])); va.append((r['wX'][ok[k:], :, :c], r['wy'][ok[k:]]))
    Xt, yt = torch.from_numpy(np.concatenate([a for a, _ in tr])), torch.from_numpy(np.concatenate([b for _, b in tr]))
    Xv, yv = np.concatenate([a for a, _ in va]), np.concatenate([b for _, b in va])
    torch.manual_seed(seed)
    model = GRU(Xt.std((0, 1)).clamp(min=1e-6))
    opt = torch.optim.Adam(model.parameters(), LR)
    best, state, wait = np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(Xt)).split(BATCH):
            loss = (model(Xt[b]) - yt[b].float()).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v = np.mean(np.abs(predict(model, Xv) - yv))
        if v < best:
            best, state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return model, epoch + 1, best, np.mean(np.abs(yv))


def main():
    tables, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    torch.set_num_threads(4)                     # more threads are slower for a GRU this small (9x at 12)
    R = {run: prepare(tables, run) for run in runs}
    for r in R.values():
        r['wX'], r['wy'] = windows(r)
    variants = {'GRU, pooled runs': (lambda run: runs, len(CHANNELS)),
                'GRU, pooled runs, charge only': (lambda run: runs, CHARGE_ONLY)}
    if len(runs) > 1:
        variants['GRU, other runs only'] = (lambda run: [o for o in runs if o != run], len(CHANNELS))
    preds = {}                                   # (variant, run) -> per-seed next-shot forecasts
    for name, (src, c) in variants.items():
        for use in dict.fromkeys(tuple(src(run)) for run in runs):
            for seed in SEEDS:
                t0 = time.time()
                model, epochs, val, val0 = fit(R, use, c, seed)
                print(f'{name:30s} trained on {len(use)} run(s), seed {seed}: {epochs} epochs, val L1 {100 * val:.2f} % (hybrid {100 * val0:.2f} %), '
                      f'{time.time() - t0:.0f} s', flush=True)
                for run in runs:
                    if tuple(src(run)) == use:
                        p = R[run]['p'].values
                        g = p.copy()                 # before the first full window: the hybrid
                        g[L - 1:] = p[L - 1:] * (1 + predict(model, R[run]['wX'][:, :, :c]))
                        preds.setdefault((name, run), []).append(g)

    ridge = RidgeCV(alphas=ALPHAS).fit(pd.concat([R[u]['F'][R[u]['fit']] for u in runs]),
                                       np.concatenate([R[u]['y'][R[u]['fit']] for u in runs]))
    rows = []
    for run, r in R.items():
        shots = scored_shots(r['df'])
        yq = r['df'].loc[shots, TARGET].values
        st, ch = (r['stable'] & ~r['train']).reindex(shots).values, r['jump'].reindex(shots).values
        cands = {'9-shot mean (baseline)': rolling_baseline(r['df']).values, f'hybrid (z={Z}, wmax={WMAX})': r['p'].values,
                 'combined, ridge on pooled runs': r['p'].values * (1 + r['on'] * ridge.predict(r['F']))}
        spread = {}
        for name in variants:
            cands[name] = np.mean(preds[(name, run)], 0)
            spread[name] = np.std([score(r, g) for g in preds[(name, run)]])
        for name, v in cands.items():
            e = np.abs(yq - pd.Series(v, r['df'].index).shift(1).reindex(shots).values)
            rows.append({'run': run, 'forecast': name, 'test MAE': score(r, v), 'seed std': spread.get(name, np.nan),
                         'stable': np.nanmean(e[st]), 'after change': np.nanmean(e[ch]) if ch.any() else np.nan})
        r['cands'] = cands
    T = pd.DataFrame(rows)
    T['dMAE %'] = T.groupby('run')['test MAE'].transform(lambda s: 100 * (1 - s / s.iloc[0]))
    pd.set_option('display.width', 250)
    print(f'\nnext-shot MAE (nC) on each run\'s test region (last 10 %); GRU = mean of {len(SEEDS)} seeds\' forecasts, '
          f'seed std = spread of the single-seed MAEs')
    print(T.round(4).fillna('').to_string(index=False))

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for run, r in R.items():
        shots = scored_shots(r['df'])
        yq = r['df'].loc[shots, TARGET].values
        show = {'9-shot mean (baseline)': 'C0', f'hybrid (z={Z}, wmax={WMAX})': 'C2', 'GRU, pooled runs': 'C3'}
        V = {k: pd.Series(r['cands'][k], r['df'].index).shift(1).reindex(shots).values for k in show}
        # zooms on the first level changes in the test region, where the forecasts differ most
        ch = r['jump'].reindex(shots).values
        starts = shots[np.r_[False, np.diff(ch.astype(int)) == 1]][:3]
        fig = plt.figure(figsize=(15, 9))
        axes = [fig.add_subplot(2, 1, 1)] + [fig.add_subplot(2, len(starts), len(starts) + i + 1) for i in range(len(starts))]
        for a, (lo, hi) in zip(axes, [(shots[0], shots[-1])] + [(s - 15, s + 30) for s in starts]):
            m = (shots >= lo) & (shots <= hi)
            a.plot(shots[m], yq[m], '.', color='k', ms=3, label='actual BCM_GUN')
            for k, c in show.items():
                a.plot(shots[m], V[k][m], '-', color=c, lw=1, label=f'{k}: test MAE {np.nanmean(np.abs(yq - V[k])):.4f} nC')
            a.set_xlabel('shot'); a.set_ylabel('charge (nC)')
        axes[0].legend(loc='upper left', fontsize=8); axes[0].set_title(f'{run}: next-shot forecast, test region')
        for a in axes[1:]:
            a.set_title('zoom on a level change')
        fig.tight_layout(); fig.savefig(os.path.join(out, f'gru_{run}.png'), dpi=110)
        print('->', os.path.join(out, f'gru_{run}.png'))


if __name__ == '__main__':
    main()
