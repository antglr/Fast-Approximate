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
- `15_alignment_significance_intervals.ipynb`: the pulse alignment, same-shot and forecast results on the aligned data, quadratic ridge, bootstrap, intervals.
- `16_attention_autoencoder_forecast.ipynb`: LSTM attention variants and autoencoder embeddings for forecasting.

## 7. Readings on the wrong shot in `results/datasets`, and the fix (`scripts/meter_alignment.py`, notebook 15)

`extract_features.py` puts group B (lab meter, RF) one **row** earlier. On the full runs that is often not one
**pulse**: 35–40 % of the rows of segs 1, 5 and 7 skip pulses, and the PC reads a shot 0.4–4.8 s after its trigger, so
a late read already holds this pulse's meter value. Section 6's h = 1 leak was this.

`meter_alignment.py` moves each reading to its own pulse with the rules of `pulses.py` (meter: previous pulse if read
< 2.2 s after the trigger, else this pulse; RF: its own stamp − 1), as `run_data.py` does; a pulse without its own
reading gets NaN. Output: `results/datasets_aligned/` (same layout). On segs 1, 4, 5, 7, 45–70 % of the lab and
26–45 % of the PGLA readings move; segs 0, 2, 3, 6 are unchanged. A camera check per read-delay bin confirms the
2.2 s switch on segs 5 and 7. Open: on seg 1, early reads follow neither pulse's camera well.

| lab vs charge jitter, row t / t − 1 | seg 4 | seg 5 | seg 1 | seg 7 |
|---|---|---|---|---|
| `results/datasets` | 0.10 / 0.14 | 0.17 / 0.18 | 0.51 / 0.43 | 0.34 / 0.56 |
| aligned | 0.17 / 0.08 | 0.30 / 0.10 | 0.66 / 0.23 | 0.65 / 0.21 |

On the aligned data, the forecast gain at h = 1 equals that at h = 2–3 without any workaround: no leak.

**Same-shot models on the aligned data** (`results/aligned/`): the nonlinear models improve at high charge
(GRU seg 5 0.95 → 0.89 %, seg 6 0.68 → 0.63 %; XGBoost seg 5 1.26 → 0.95 %; seg 5 now 1.96× the noise floor, was
2.25×). The linear ridge gets worse on segs 1 and 5 (1.12 → 1.29 %, 0.99 → 1.09 %): with its real signal, the lab
meter's relation to the charge is curved. A **quadratic ridge** fixes it: 0.88 % on seg 5, 2.21 % on seg 7 (GRU
0.89 / 2.40), 1.04 % on seg 1 (GRU 1.00). The GRU's edge at high charge was the curvature, not memory.

## 8. Significance (circular block bootstrap, 50-shot blocks; `scripts/uncertainty_regimes.py`)

- Same shot: the readings cut the error by 26–67 % against the level alone, significant on every segment. At high
  charge GRU / GP beat the linear ridge by 11–22 % (segs 1, 5, 6), significantly; at low charge they are
  indistinguishable.
- Forecast: the 9-shot mean is beaten significantly on 5–7 of 7 segments at h = 1–5 (seg 4 is the exception); at
  h = 50 mostly not significant. The instruments add ~0.5–1 point over the BCM history, within the intervals.

## 9. Heteroscedastic 90 % intervals

Split-conformal quantiles normalised by the robust spread of the last 200 errors known at prediction time. Against
constant-width intervals: coverage closer to 90 % on average, better in the noisy half of the test shots, and much
narrower at high charge for the same-shot model (seg 5: 8.2 → 4.7 % of the level at 93 % coverage; seg 1: 8.1 → 6.2 %).
Seg 6's forecasts stay under-covered (78–85 %): its test region is noisier than its short training part.

## 10. LSTM with attention and autoencoder embeddings (`scripts/attn_forecast.py`, notebook 16, plan `plan_attention_ae.md`)

Nine variants (LSTM; + additive / general / multi-head / self-attention; + variable selection; per-shot autoencoder
frozen or joint; sequence autoencoder), h = 1–50, aligned data, 3 seeds. All land within ~0.5 points of the GRU and
the ridge. The attention readouts rank best (mean rank 4.3–4.7 of 11 vs GRU 6.1, ridge 6.2) and gain ~1 point over the
GRU at h = 20 (significant on segs 1 and 6), but against the GRU they are significant in only 11 % of cases. The
attention weights stay within 5 % of uniform: a learned window average, no preferred lag. Variable selection
up-weights only the BCM-history inputs. Autoencoders do not help (per-shot AE 29 → 8 at 23 % reconstruction error;
`ae_frozen` and `seq_ae` rank last). L = 64 is the best context; 2–3 s per fit.
