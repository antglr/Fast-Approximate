# Hyperparameter search for jitter_sequence.py's best model (ridge + past BCM, then a GRU with the past BCM on its
# residual). Searched: trailing level length SLOW, past-BCM lags LAGS of the ridge, GRU hidden size H, window L,
# learning rate LR and batch size; TRIALS optuna (TPE) trials per regime, the first being the default, one seed.
# The test region is never used for choosing: each training part is split into fit / early-stopping (the 10 % before
# the last 10 %) / tuning (the last 10 %, jitter_sequence.py's early-stopping rows). Score = MAE in pC of
# exp(level + prediction) on the tuning rows. Then the best and the default configuration per regime are refitted as
# in jitter_sequence.py (early stopping on the last 10 %, SEEDS) and scored on the test regions.
# Usage: python -I jitter_tune.py <tables dir> <run> [<run> ...]
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import optuna
import torch
from sklearn.linear_model import RidgeCV
import jitter_sequence as js
from same_shot_regression import ALPHAS, CV, REGIMES, stitched

TRIALS = 20
SPACE = {'SLOW': (25, 50, 100, 200), 'LAGS': (4, 16, 32), 'H': (16, 32, 64), 'L': (16, 64, 128),
         'LR': (3e-4, 1e-3, 3e-3), 'BATCH': (128, 256, 512)}
DEFAULT = {'SLOW': 50, 'LAGS': 16, 'H': 32, 'L': 64, 'LR': 1e-3, 'BATCH': 256}


def run(F, cfg, R, tr, es, seeds):
    # level, prediction on the regime's rows (log units); ridge on tr, GRU fitted on tr and stopped on es
    for k in ('SLOW', 'H', 'LR', 'BATCH'):
        setattr(js, k, cfg[k])                       # module constants read by trailing / Seq / fit
    level = js.trailing(np.log(F.Q), F.seg)
    yf = np.log(F.Q) - level
    X = js.features(F)
    past = pd.concat({f'BCM lag {j}': yf.groupby(F.seg).shift(j) for j in range(1, cfg['LAGS'] + 1)}, axis=1).fillna(0)
    XR = pd.concat([X, past], axis=1)
    Rm = R & level.notna()
    trm = tr & level.notna()
    base = np.zeros(len(F), np.float32)
    base[Rm] = RidgeCV(alphas=ALPHAS, cv=CV).fit(XR[trm], yf[trm]).predict(XR[Rm])
    A = torch.from_numpy(np.vstack([np.c_[X.values, past['BCM lag 1'].values], np.zeros((1, X.shape[1] + 1))]).astype(np.float32))
    pos = F.groupby('seg').cumcount().values
    lag = np.arange(cfg['L'] - 1, -1, -1)
    I = np.where(pos[:, None] >= lag, np.arange(len(F))[:, None] - lag, len(F))
    y = yf.fillna(0).values.astype(np.float32)
    t, e = np.flatnonzero(trm), np.flatnonzero(es & level.notna())
    preds = [js.predict(js.fit('GRU', A, I, y - base, t, e, s)[0], A, I[Rm.values]) for s in seeds]
    p = pd.Series(np.nan, F.index)
    p[Rm] = base[Rm] + np.mean(preds, 0)
    return level, p


def main():
    tables, runs = sys.argv[1], sys.argv[2:]
    torch.set_num_threads(8)
    F = stitched(tables, runs)
    ok = js.trailing(np.log(F.Q), F.seg).notna()     # SLOW = 50 rows, for the splits only
    fit_rows = F.train & ok
    k = fit_rows.groupby(F.seg).cumsum() / fit_rows.groupby(F.seg).cumsum().groupby(F.seg).transform('max')
    tune, es, rest = fit_rows & (k > 0.9), fit_rows & (k > 0.8) & (k <= 0.9), fit_rows & (k <= 0.8)
    test = ~F.train & ok
    mae = lambda level, p, i: 1000 * (np.exp(level[i] + p[i]) - F.Q[i]).abs().mean()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows, best = [], {}
    for rname in REGIMES:
        R = F.regime == rname

        def objective(trial):
            t0 = time.time()
            cfg = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
            level, p = run(F, cfg, R, rest & R, es & R, (0,))
            rows.append({'regime': rname, 'trial': trial.number, **cfg, 'tuning MAE pC': mae(level, p, F.index[tune & R])})
            print(rows[-1], f'{time.time() - t0:.0f} s', flush=True)
            return rows[-1]['tuning MAE pC']

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.enqueue_trial(DEFAULT)
        study.optimize(objective, n_trials=TRIALS)
        best[rname] = study.best_params
    pd.set_option('display.width', 250)
    print('\nsearch (tuning rows = last 10 % of each training part)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    out = []
    for rname in REGIMES:
        R = F.regime == rname
        for tag, cfg in (('default', DEFAULT), ('tuned', best[rname])):
            level, p = run(F, cfg, R, fit_rows & ~tune & R, tune & R, js.SEEDS)
            for s, g in F[test & R].groupby('seg'):
                out.append({'config': tag, 'seg': s, 'MAE pC': mae(level, p, g.index)})
        print(f'{rname}: tuned {best[rname]}', flush=True)
    print('\nMAE pC, test regions (refitted as jitter_sequence.py, seeds', js.SEEDS, ')')
    print(pd.DataFrame(out).pivot_table(index='config', columns='seg', values='MAE pC').round(1).to_string())


if __name__ == '__main__':
    main()
