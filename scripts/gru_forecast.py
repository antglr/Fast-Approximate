# Next-shot GRU forecaster (the sequence model of the Hard tier), compared with the hybrid and the combined forecaster.
# Like the ridge in combined_forecast.py, the GRU corrects the hybrid level on stable beam. Input: the last L shots of
# the current segment (since the last level change), each with
#   q      the charge if the hybrid used the shot (else 0, with skip = 1; steps before the segment are skipped too)
#   level  the hybrid level after that shot
#   lab, clex, step  the laser meters and the beam-loading-step charge
# Charges are divided by the hybrid level after the newest shot, the meters by their window mean, so one model serves
# 0.3 to 20 nC; then each channel by its training std, clipped at CLIP. Output: the next shot relative to that level
# (0 = the hybrid), trained with an L1 loss (the metric is MAE) on the stable shots of each run's first 90 %, the last
# 10 % of those for early stopping; applied once the segment is AGE_MIN shots old. Each variant is trained with
# several seeds and their forecasts averaged. Scored on each run's last 10 % with score_forecast (h=1).
# To try more inputs, add channels in windows(). The notebook variables (klystron, gun field, magnets, temperatures,
# humidities, laser scope) were screened and add nothing for the next shot (see the README).
# Usage: python -I gru_forecast.py <tables dir> <out dir> <run> [<run> ...]
import os, sys, time, warnings
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
LASER = ('lab', 'clex', 'step')                    # dropped in the charge-only variant
CLIP = 5                # clip the standardised inputs (in training std): a laser step must not saturate the GRU
STABLE_ONLY = True      # train on stable beam, apply once the segment is AGE_MIN old (as the ridge correction)
VARIANTS = ('pooled runs', 'pooled runs, charge only', 'other runs only')


def windows(r):
    # -> X (shots, L, channels) for the window ending at each shot n >= L-1, and y = the next good shot / level - 1
    p, x, used = r['p'].values, r['x'], r['used']
    raw = {'q': np.where(used, r['df'][TARGET].values, np.nan), 'skip': (~used).astype(float), 'level': p,
           'lab': x.lab.values, 'clex': x.clex.values, 'step': x.step.values}
    W = {k: np.lib.stride_tricks.sliding_window_view(v, L) for k, v in raw.items()}     # row j = shots j .. j+L-1
    # only the current segment (since the last level change), as in the ridge's features: a window reaching back to
    # the old level never occurs in the stable training shots, and the GRU then predicts nonsense (+26 % on run 3)
    seg = np.lib.stride_tricks.sliding_window_view(r['seg'].values, L)
    old = seg != seg[:, -1:]
    W = {k: np.where(old, 1.0 if k == 'skip' else np.nan, v) for k, v in W.items()}
    pn = p[L - 1:, None]
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                        # all-NaN windows (meter gaps)
        f = {'q': W['q'] / pn - 1, 'skip': W['skip'], 'level': W['level'] / pn - 1,
             'lab': W['lab'] / np.nanmean(W['lab'], 1, keepdims=True) - 1,
             'clex': W['clex'] / np.nanmean(W['clex'], 1, keepdims=True) - 1,
             'step': (W['step'] - np.nanmean(W['step'], 1, keepdims=True)) / pn}   # the step has an offset at low charge
    # clip: a feature gone wild (as the step did at 0.3 nC) must not dominate
    X = np.nan_to_num(np.stack([f[c] for c in CHANNELS], -1), nan=0, posinf=0, neginf=0).clip(-5, 5).astype(np.float32)
    y = r['q'].shift(-1).values[L - 1:] / p[L - 1:] - 1
    j = len(X) // 2
    assert np.isclose(X[j, -1, 0], (raw['q'][j + L - 1] / p[j + L - 1] - 1) if used[j + L - 1] else 0, atol=1e-6)
    assert (X[old][:, 1] == 1).all() and (X[old][:, 0] == 0).all()           # steps before the segment are blanked
    return X, y


class GRU(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', scale)                # per-channel training std: the relative charges are ~0.03
        self.gru, self.out = torch.nn.GRU(len(scale), H, batch_first=True), torch.nn.Linear(H, 1)

    def forward(self, X):
        return self.out(self.gru((X / self.scale).clamp(-CLIP, CLIP))[0][:, -1]).squeeze(-1)


def predict(model, X):
    with torch.no_grad():
        return torch.cat([model(b) for b in torch.from_numpy(X).split(4096)]).numpy()


def fit(R, runs, c, seed):
    # train on the good next shots of the runs' first 90 % (or the stable ones), early-stop on the last 10 % of those
    tr, va = [], []
    for run in runs:
        r = R[run]
        shot = r['df'].index.values[L - 1:]
        ok = r['fit'][L - 1:] & r['used'][L - 1:] if STABLE_ONLY else np.isfinite(r['wy']) & (shot + 1 < cut_shot(r['df']))
        ok = np.flatnonzero(ok)
        k = int(0.9 * len(ok))
        tr.append((r['wX'][ok[:k]][:, :, c], r['wy'][ok[:k]])); va.append((r['wX'][ok[k:]][:, :, c], r['wy'][ok[k:]]))
    Xt, yt = torch.from_numpy(np.concatenate([a for a, _ in tr])), torch.from_numpy(np.concatenate([b for _, b in tr]))
    Xv, yv = np.concatenate([a for a, _ in va]), np.concatenate([b for _, b in va])
    torch.manual_seed(seed)
    model = GRU(Xt.std((0, 1)).clamp(min=1e-6))
    torch.nn.init.zeros_(model.out.weight); torch.nn.init.zeros_(model.out.bias)       # starts as the hybrid ...
    opt = torch.optim.Adam(model.parameters(), LR)
    best, state, wait = np.mean(np.abs(yv)), {k: t.clone() for k, t in model.state_dict().items()}, 0   # ... and may stay it
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
    every, charge = list(range(len(CHANNELS))), [i for i, k in enumerate(CHANNELS) if k not in LASER]
    variants = {'GRU, pooled runs': (lambda run: runs, every), 'GRU, pooled runs, charge only': (lambda run: runs, charge),
                'GRU, other runs only': (lambda run: [o for o in runs if o != run], every)}
    variants = {k: v for k, v in variants.items() if k[5:] in VARIANTS and (len(runs) > 1 or 'other' not in k)}
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
                        # off when the newest shot was skipped (e.g. the BCM reads far below the step charge): windows
                        # of skipped shots never occur in the stable training shots (-13 % on run 3's 1.67 nC plateau)
                        on = R[run]['on'][L - 1:] & R[run]['used'][L - 1:] if STABLE_ONLY else 1
                        g[L - 1:] = p[L - 1:] * (1 + on * predict(model, R[run]['wX'][:, :, c]))
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
    np.savez(os.path.join(out, 'gru_forecasts.npz'), **{f'{run}|{k}': v for run, r in R.items() for k, v in r['cands'].items()})
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
