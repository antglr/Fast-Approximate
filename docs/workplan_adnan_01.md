# Workplan (branch adnan_01): models per charge regime

**Question.** Many architectures (linear, ridge, random forest, XGBoost, MLP, CNN, RNN, GRU, LSTM) seem to work at low
charge and to be limited at high charge. Is that the model, the data preparation, or the metric?

Regimes (clean segments of `results/datasets/`):

| Regime | Segments | Level | Character |
|---|---|---|---|
| low charge | 0, 2, 3, 4 | 0.1–0.25 nC | 8–14 % jitter, about half of it BCM noise |
| high, repetitive | 5 | ~5.5 nC | slow sawtooth following the lab laser |
| high, non-repetitive | 1, 6, 7 | 5–10 nC | drifts and steps, seg 7 ends with a jump to 18 nC |

## Phase 0: one split, one score (`scripts/regimes.py`, `scripts/regime_bench.py`)

- Split per segment: test = last 10 %, validation = last 10 % of the rest, fit = the remainder (challenge rule).
- Dropouts (below 30 % of the trailing local median) and invalid rows are never trained on or scored.
- Every model is scored with the same `regimes.score`: MAE (nC and % of level), R², jitter explained, residual rms
  against the BCM noise floor, and the slow share of the error.
- Two framings:
  - **absolute:** log Q from the shot's own readings, one model per segment (the `08_simple_models` setting);
  - **level + jitter:** log Q − the trailing level of past BCM, from the fast parts of the readings, per segment and
    pooled per regime (the `jitter_sequence` setting).
- `python3 -I scripts/regime_bench.py results/datasets results/regime_bench`

## Phase 1: level / jitter decomposition per regime (`scripts/decompose.py`)

- Spectra of the charge per regime: variance in slow / mid / fast bands, coherence with camera and laser per band.
- Causal level filters, each over a parameter grid: moving average, EMA (= steady-state Kalman local level),
  trailing median, Holt (local linear trend), outlier-robust EMA, change-point reset (the hybrid).
- Every filter passes a leak test: changing the future never changes a past level.
- Chosen per regime on the validation block, by the error of a ridge on the resulting jitter.
- `python3 -I scripts/decompose.py results/datasets results/decompose`

## Phase 2: models per regime and per component

- Level: GP with periodic + RBF kernel (repetitive), change-point Kalman / GP Matérn (non-repetitive), long EMA (low).
- Jitter: ridge + GRU backbone; sparse variational GP (GPyTorch) with ARD for uncertainty and input relevance.
- Ensembles: per-regime stacking of ridge, XGBoost, GRU and GP on the validation block; seed and block bagging;
  a regime-gated mixture of experts as the single deployable model.
- Then the same models with inputs known only up to shot t − N (forecasting).

## Running on Dalia

```bash
ssh dalia; cd ~/Fast-Approximate                 # own clone, synced from the laptop
sbatch slurm/run.sbatch scripts/regime_bench.py results/datasets results/regime_bench   # 1 GB200, 2 h
sbatch slurm/jupyter.sbatch                      # kernel for local VS Code; tunnel printed in slurm_logs/
```

`slurm/env.sh` sets the container (`optimization_clear.sif`, aarch64: compute nodes only) and `~/pyext`, which adds
gpytorch, PyWavelets, statsmodels and tabulate on top of it.
