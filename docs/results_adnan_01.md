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
