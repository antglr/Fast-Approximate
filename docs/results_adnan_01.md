# Results (branch adnan_01): why high charge looked hard, and what fixes it

Plan: `workplan_adnan_01.md`. Same-shot charge (virtual diagnostic), level + jitter framing, pooled per regime,
chronological split (test = last 10 % of each segment), scored with `scripts/regimes.py`. Tables: `results/regime_bench*/`,
`results/decompose*/`, `results/phase2/`.

## 1. High charge was limited by outliers, not by the architectures

A few shots per segment carry huge fast parts: camera spikes and dark frames, and BCM readings that are far off but
above the 30 % dropout cut. Least-squares models (linear, ridge, MSE-trained MLP) chase them. The L1-trained, input-clipped
RNN / GRU / LSTM do not, which is why only they looked good at high charge.

Correlation of the charge jitter with the fast part of the camera intensity (training part):

| | seg 0 (low) | seg 1 (non-rep.) | seg 5 (rep.) | seg 7 (non-rep.) |
|---|---|---|---|---|
| as prepared | 0.12 | 0.09 | 0.07 | 0.20 |
| robust (5 MAD) | 0.69 | 0.80 | 0.75 | 0.76 |

Robust preparation (`regimes.load(robust=5)`): input readings beyond 5 robust sigmas of their fast part become missing
(a camera spike blanks the whole camera group, with a flag); shots whose jitter is beyond 5 robust sigmas are not
trained on (3–56 shots per segment) but are still scored.

| MAE % of level, test | seg 1 | seg 6 | seg 7 | seg 5 | seg 0 | seg 3 | seg 4 |
|---|---|---|---|---|---|---|---|
| trailing level only | 2.98 | 2.19 | 3.87 | 2.48 | 7.28 | 9.66 | 10.82 |
| ridge, as prepared | 2.86 | 1.28 | 3.20 | 2.48 | 5.15 | 8.23 | 9.08 |
| GRU, as prepared | 1.28 | 0.74 | 2.27 | 0.99 | 3.95 | 7.24 | 8.18 |
| ridge, robust + robust EMA level | 1.12 | 0.72 | 2.48 | 0.99 | 3.83 | 6.89 | 8.25 |
| GRU, robust + robust EMA level | 1.03 | 0.68 | 2.47 | 0.95 | 3.82 | 6.98 | 8.14 |
| ridge + GRU, robust | **1.01** | 0.68 | 2.54 | 0.95 | 3.82 | 6.89 | 8.24 |
| sparse GP (ARD), robust | 1.07 | 0.70 | 2.47 | 0.98 | 3.84 | 6.96 | **8.02** |
| mean of ridge, XGBoost, GRU, ridge+GRU, GP | 1.05 | 0.70 | 2.46 | 0.97 | 3.81 | **6.88** | 8.09 |

- After robust preparation all families are within ~10 % of each other in every regime. Trees are the weakest at
  high charge (XGBoost 1.30 / 1.26 % on segs 1 / 5).
- **Raw-charge R² is a misleading metric here:** small test variance and level extrapolation give R² = −1.0 for XGBoost
  on seg 5 (absolute framing) while its MAE is reasonable. Use MAE % of level and jitter explained.

## 2. Level / jitter decomposition (`scripts/decompose.py`)

- All 36 causal filters pass a leak test (changing the future never changes a past level).
- With robust data the **outlier-robust EMA (half-life 25 shots)** is chosen on validation in all three regimes;
  it beats the 50-shot mean by 1–5 % downstream.
- Steady-state Kalman local level = EMA; one-sided HP ≈ Holt. Change-point reset is worst on these clean, stable
  segments.
- Spectra (Lomb–Scargle on the good shots, no gap filling):
  - seg 5: the 1500–3000-shot sawtooth, plus a **~188-shot cycle (≈ 3.8 min) shared by charge, lab laser and camera**;
    seg 1 has a similar ~194–222-shot cycle. Laser-driven, hence forecastable over tens of shots.
  - A 2.5-row "period" everywhere: rows are not on a uniform pulse grid (acquisition skips pulses).
  - Welch spectra with gap filling show fake sinc lobes on seg 5: do not fill gaps before an FFT.

## 3. How far from the floor

Residual rms / BCM noise floor (`bcm_noise_floor.py`): low charge 1.05–1.25 (at the floor), seg 6 1.2, seg 1 1.5,
**seg 5 2.25** (most room left). Seg 7's test region is dominated by the final jump to 18 nC (jitter explained ≈ 0.1
for every model).

## 4. Ensembles and intervals (`scripts/regime_phase2.py`)

- The equal-weight mean is as good as the best single model; non-negative stacking fitted on the validation block is
  worse at high charge (weights sum to 0.73 on seg 5: it shrinks towards the level because the validation block drifts).
- 90 % intervals: split-conformal around the stack covers 79–99 %, the GP's own 84–99 %. Both have a constant width per
  regime; the low-charge segments 3 / 4 are under-covered, the high-charge ones over-covered → heteroscedastic noise
  model next.

## 5. Forecasting the level from the BCM history alone (`scripts/level_forecast.py`)

Change of the test MAE against the mean of the last 9 good shots (challenge baseline), negative = better.
GP on time (Matérn 5/2 + noise, + a 2048-shot periodic term on seg 5), hyperparameters fitted on the training part,
each forecast conditioned only on the good shots up to t − h (`results/level_forecast_v1/`).

| h | seg 1 | seg 6 | seg 7 | seg 5 | seg 0 | seg 3 | seg 4 |
|---|---|---|---|---|---|---|---|
| 1 | −1.9 | +1.6 | −4.5 | −1.7 | −3.9 | −1.1 | −1.1 |
| 10 | −3.3 | −7.8 | −0.2 | −1.0 | −6.4 | −3.3 | −0.9 |
| 50 | −2.5 | −5.1 | −3.1 | +0.3 | −3.3 | −8.1 | −2.7 |

- The GP beats the baseline on 6/7 segments at h = 1 and 50, and on all 7 at h = 10; gains are 1–8 %. Not yet tested
  for significance (block bootstrap as in `forecast_uncertainty.py`).
- The robust EMA, the best filter for the decomposition, is a poor forecaster (up to +7 % at h = 50 on seg 4): the
  best level for removing the trend is not the best guess of the next level.
- **Negative result:** adding the ~190-shot cycle seen in the spectra (seg 5: 2048 + 188 shots, seg 1: 200) does not
  help at h ≤ 10 and hurts at h = 50 (seg 5 +3.9 %, seg 1 +1.9 %; `results/level_forecast_periodic190/`). The cycle is
  in the spectrum but its phase does not hold long enough to extrapolate.

## 6. Forecasting with everything known at t − h (`scripts/feature_forecast.py`, notebook 14)

**Leak found in `results/datasets`:** `lab_laser_power` (and `PGLA_drop`) on row t − 1 correlates with shot t's charge
jitter as much as or more than row t (seg 7: 0.56 against 0.34). On part of the rows the column holds the *next*
shot's reading; `extract_features.py` does not apply the `meter_shift` rule of `pulses.py` per shot. With them on
their row, the next-shot gain from the instruments was 17–20 % on segs 5 and 7 and collapsed to 4 % at h = 2. The
forecaster now uses those two columns one row later (`--no-late-fix` reproduces the leak,
`results/feature_forecast_leak/`). The same-shot models on the team datasets are affected too: the lab column mixes
shots t and t + 1. The fix belongs in `extract_features.py`.

After the fix (`results/feature_forecast/`), every model beats the 9-shot mean at nearly every horizon by 1–12 %.
The instruments add 0.5–2 points over the BCM history at short horizons, and the gain does not grow with h: the
jitter is laser-driven and the next pulse's laser energy is not predictable. Forecasting tracks the level better;
the instruments pay in the same-shot virtual diagnostic.

## Notebooks (`notebooks/adnan/`, executed on Dalia)

- `11_robust_preparation.ipynb`: the outliers, the correlations before / after, all nine model families before / after, R² vs MAE.
- `12_trend_noise_decomposition.ipynb`: Lomb–Scargle spectra per regime, the gap-filling trap, the filter comparison.
- `13_models_gp_ensembles_floor.ipynb`: GP, ensembles, stacking weights, gap to the BCM noise floor, intervals.
- `14_forecasting_and_h1_leak.ipynb`: level forecasting, the h = 1 leak, forecast skill against the horizon.
