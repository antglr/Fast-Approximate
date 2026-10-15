# Gaussian process for the shot-to-shot jitter, with the BCM noise from bcm_noise_model.py as its noise term.
# Same target, inputs (current shot), split, regimes and scores as jitter_sequence.py. Kernel on the standardised
# inputs (clipped at CLIP): scale x ARD RBF + linear, constant mean (gpytorch, exact GP on N training shots evenly
# spread over each regime's training rows, early-stopping rows and 5-MAD outliers excluded). Noise term, per shot, in log units:
#   fixed    sigma_n(Q) / Q,  sigma_n(Q)^2 = A^2 + (B Q)^2,  Q = the trailing level: the measured BCM noise, not learned
#   learned  one Gaussian noise variance per regime, fitted with the kernel (the usual GP)
# Two targets: the jitter itself (GP alone, as the ridge) and the residual of jitter_sequence.py's best model (GP on top:
# a correction and an uncertainty for it). Besides the errors, the test scores check the predictive uncertainty:
# coverage of the 90 % and 99 % intervals (mean +- 1.645 / 2.576 predictive std) and the mean 90 % half-width in pC.
# Usage: python -I jitter_gp.py <tables dir> <jitter_sequence out dir> <out dir> <run> [<run> ...]
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import gpytorch
from jitter_sequence import CLIP, features, trailing
from same_shot_regression import REGIMES, stitched

A, B = 10.7, 0.0072          # bcm_noise_model.py, training parts: fixed pC and fraction of the charge
N, ITERS, LR = 2000, 100, 0.05      # 2000 shots: 1.3 s per step (4000: 11 s)
BEST = 'ridge, + past BCM + GRU, L=64, past BCM'
torch.set_default_dtype(torch.float64)


class GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean = gpytorch.means.ConstantMean()
        self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=x.shape[1])) + gpytorch.kernels.LinearKernel()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.cov(x))


def gp(x, y, noise, xt, noise_t):
    # noise: per-shot variances (fixed) or None (learned); returns the predictive mean and variance at xt and the noise std
    # fitted on y / its std: gpytorch's initial scales (~0.7) are far from log-charge variances (~1e-3), and Adam
    # would not get there within ITERS
    sc = y.std()
    y = y / sc
    noise, noise_t = (None, None) if noise is None else (noise / sc ** 2, noise_t / sc ** 2)
    lik = (gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise, learn_additional_noise=False) if noise is not None
           else gpytorch.likelihoods.GaussianLikelihood())
    model = GP(x, y, lik)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    with gpytorch.settings.max_cholesky_size(N + 1):
        for _ in range(ITERS):
            opt.zero_grad(); loss = -mll(model(x), y); loss.backward(); opt.step()
        model.eval(); lik.eval()
        with torch.no_grad():
            mu, var = [], []
            for b in torch.arange(len(xt)).split(4000):
                f = model(xt[b])
                p = lik(f, noise=noise_t[b]) if noise is not None else lik(f)
                mu.append(p.mean); var.append(p.variance)
    return sc.item() * torch.cat(mu).numpy(), sc.item() ** 2 * torch.cat(var).numpy(), sc.item() * lik.noise.sqrt().mean().item()


def main():
    tables, pdir, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    torch.set_num_threads(8)
    os.makedirs(out, exist_ok=True)
    F = stitched(tables, runs)
    P = pd.read_parquet(os.path.join(pdir, 'jitter_predictions.parquet'))
    assert (P.run.values == F.run.values).all() and (P.shot.values == F.shot.values).all()
    level = trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    X = features(F)
    ok = level.notna()
    fit_rows = F.train & ok
    k = fit_rows.groupby(F.seg).cumsum()
    va = fit_rows & (k > 0.9 * k.groupby(F.seg).transform('max'))
    q = 1000 * np.exp(level)                                                # trailing level, pC
    nvar = (A ** 2 + (B * q) ** 2) / q ** 2                                 # BCM noise variance, log units
    # self-check: with the true noise fixed, a 1-D sine is recovered and the 90 % interval covers ~90 %
    g = torch.Generator().manual_seed(0)
    xs = torch.rand(300, 1, generator=g) * 6; ys = torch.sin(xs[:, 0]) + 0.3 * torch.randn(300, generator=g)
    m, v, _ = gp(xs, ys, torch.full((300,), 0.09), xs, torch.full((300,), 0.09))
    assert np.mean((m - np.sin(xs[:, 0].numpy())) ** 2) < 0.01 and 0.85 < np.mean(np.abs(ys.numpy() - m) < 1.645 * np.sqrt(v)) < 0.95

    preds = {}
    for rname in REGIMES:
        R = (F.regime == rname) & ok
        tr = np.flatnonzero(fit_rows & R & ~va)
        tr = tr[np.linspace(0, len(tr) - 1, min(N, len(tr))).astype(int)]
        sd = X.iloc[tr].std().replace(0, 1)
        Z = torch.from_numpy(((X[R] / sd).clip(-CLIP, CLIP)).values)
        idx = pd.Series(np.arange(R.sum()), F.index[R])
        zt, nt = Z[idx[F.index[tr]].values], torch.from_numpy(nvar[F.index[tr]].values)
        for target, base in (('jitter', 0.0), ('best residual', P[BEST])):
            y = (yf - base)[R]
            yt = y[F.index[tr]]
            dev = (yt - yt.median()).abs()
            use = torch.from_numpy((dev < 5 * 1.4826 * dev.median()).values)   # Gaussian likelihood: no BCM-lost shots or spikes
            for noise in ('fixed', 'learned'):
                t0 = time.time()
                mu, var, nsd = gp(zt[use], torch.from_numpy(yt.values)[use], nt[use] if noise == 'fixed' else None,
                                  Z, torch.from_numpy(nvar[R].values))
                name = f'GP on {target}, {noise} noise'
                p = preds.setdefault(name, pd.DataFrame(np.nan, F.index, ['mu', 'var']))
                p.loc[R, 'mu'], p.loc[R, 'var'] = mu + (base[R].values if target != 'jitter' else 0), var
                extra = '' if noise == 'fixed' else f', learned noise std {100 * nsd:.2f} %'
                print(f'{rname}, {name}: {time.time() - t0:.0f} s{extra}', flush=True)

    test = ~F.train & ok
    rows = []
    cand = {'ridge, current shot': None, BEST: None, **preds}
    for name, p in cand.items():
        mu = P[name] if p is None else p.mu
        for s, g in F[test].groupby('seg'):
            i = g.index
            e = np.exp(level[i] + mu[i]) - F.Q[i]
            r = {'model': name, 'seg': s, 'jitter explained %': 100 * (1 - ((yf[i] - mu[i]) ** 2).mean() / (yf[i] ** 2).mean()),
                 'MAE pC': 1000 * e.abs().mean()}
            if p is not None:
                z = (yf[i] - mu[i]).abs() / np.sqrt(p['var'][i])
                r.update({'cover 90 %': 100 * (z < 1.645).mean(), 'cover 99 %': 100 * (z < 2.576).mean(),
                          '90 % half-width pC': (1.645 * np.sqrt(p['var'][i]) * q[i]).mean()})
            rows.append(r)
    D = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    for col in ('jitter explained %', 'MAE pC', 'cover 90 %', 'cover 99 %', '90 % half-width pC'):
        print(f'\n{col}, test regions')
        print(D.pivot_table(index='model', columns='seg', values=col).reindex([c for c in cand if c in set(D.dropna(subset=[col]).model)]).round(1).to_string())
    pd.DataFrame({f'{n}|{c}': p[c] for n, p in preds.items() for c in p}).assign(run=F.run, shot=F.shot) \
        .to_parquet(os.path.join(out, 'gp_predictions.parquet'))


if __name__ == '__main__':
    main()
