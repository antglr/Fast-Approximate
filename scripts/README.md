# Next-shot charge forecasting (CLEAR, BCM_GUN)

Download a run's HDF5 chunks, reduce them to per-shot tables, and score next-shot forecasts against the
challenge baseline, the mean of the last 9 good shots. The split is chronological: train on the first 90 % of the run,
test on the last 10 %. Only the chunks kept in `../config/CLEAN_CUTS.toml` are used (below). Results per run are in
`../results/<run>/`.

```bash
pip install -r requirements.txt
scripts/fetch_h5.sh 2026-10-09_07-46-07 data/h5                    # 110 chunks, 17 GB (~10 min)
PY=python3 scripts/run_pipeline.sh 2026-10-09_07-46-07 data/h5 data/tables results
```

| Script | What it does |
|---|---|
| `fetch_h5.sh` | Downloads `measured_data/acquisition_<run>/h5/chunk_*.h5` from the public CERNBox share |
| `h5chunk.py` | Reads an HDF5 chunk as the same per-shot dicts as the original pickles. Skips the camera. Converts `pc_time` back to CERN local time. |
| `extract_rf.py`, `extract_laser.py` | Full RF traces (960 samples), and laser meters plus scope traces, each with its stamp offset → `<run>_rf.npz`, `<run>_laser.npz` |
| `reduce_chunks.py` | One table row per shot (kickoff-notebook features) → `<run>.parquet` |
| `nbfuncs.py` | Kickoff-notebook functions: scoring rules, dropouts, rolling baseline, safe pickle loader |
| `pulses.py` | Puts every reading on the 1.2 s pulse grid (see below) |
| `beam_loading_step.py` | `step()`: the beam-loading dip on the gun pickup PGLA, a second charge measurement |
| `plot_run.py` | Whole-run overview: BCM charge, step charge, laser, gun field; discarded chunks shaded red |
| `run_data.py` | Loads one run's tables onto the pulse grid, keeps only the clean chunks; shared shot labels (BCM-wrong, stable, level change) |
| `forecast_run.py` | Forecast scores for horizons 1–50, the next-shot plot and the stable-shot residuals |
| `forecasters.py` | `hybrid()`: the level-change-aware mean, shared by the two scripts below |
| `hybrid_forecast.py` | Tunes and scores the hybrid on one run |
| `combined_forecast.py` | The hybrid plus a ridge correction, trained across runs (below); run it after `run_pipeline.sh` |
| `gru_forecast.py` | GRU sequence model: a stable-beam correction of the hybrid from the last 64 shots, trained across runs (below); needs PyTorch |
| `gru_window_sweep.py` | Reruns the GRU for context windows of 8–256 shots |
| `forecast_uncertainty.py` | Bootstrap intervals for the MAE gains, and 90 % prediction intervals for the combined forecaster (below) |
| `screen_variables.py` | Tests whether the notebook variables (klystron, gun field, magnets, climate, laser scope, camera) help the next-shot forecast |
| `virtual_bcm.py` | Same-shot virtual BCM: shot n's charge from shot n's own laser readings (and camera, if present) plus the BCM history (below) |
| `camera_features.py` | Puts the camera features from `extract_features.py` (`results/datasets/`) on the pulse grid, drops dark frames → `<run>_camera.npz` |
| `same_shot_regression.py` | Same-shot charge from the reduced feature set alone (no BCM history), log-ridge per charge regime (below) |
| `jitter_sequence.py` | Shot-to-shot jitter against the trailing BCM level, from the instruments' own jitter: ridge vs RNN vs GRU per regime (below) |
| `bcm_noise_floor.py` | The BCM's own shot-to-shot noise (three-cornered hat: BCM, camera, step / clex / lab): the floor for the jitter models (below) |
| `jitter_input_screen.py` | Screens laser-pulse shape and RF flat-top readings as extra jitter inputs (below): they add nothing |
| `jitter_online.py` | Shot-by-shot refit of a linear correction on top of the jitter models, using the BCM up to the previous shot (below) |
| `jitter_noise_plot.py` | Distribution of the BCM noise vs the jitter models' residuals; is the noise Gaussian? (below) |
| `bcm_noise_model.py` | Tighter noise estimate, noise vs charge law, model-free skewness / kurtosis of the noise, noise stability (below) |
| `jitter_gp.py` | Gaussian process for the jitter with the measured BCM noise as a fixed per-shot noise term (below) |
| `jitter_maxloss.py` | Linear jitter models trained on mean |error|^p, p = 1 … ∞ (minimax): does it lower the largest error? (below) |
| `jitter_spectrum.py` | Power spectra of the jitter and the model residuals against the noise floor: decomposition by timescale (below) |

Run the scripts with `python -I`. The tables match the ones made from the pickles, except that the camera
columns are left out.

## Things the data does that matter

- **Clean chunks only.** `run_data.clean_run` applies `../config/CLEAN_CUTS.toml`, the team's visual inspection of
  BCM_GUN. Every forecasting script loads its data through it.
  - Each run is trimmed to its first to last kept chunk. The test region is the last 10 % of that span.
  - Discarded chunks in between lose their BCM reading: they count as dropouts, are never trained on or scored, and no
    forecast averages across them.
  - What goes:
    - run 1 (09_07) chunks 41–86 (stuck BCM, laser scan) and 107–109 (most of the 19 nC / 1 nC level changes);
    - run 3 (08_00) chunks 194–254 (the 0.3 nC part);
    - run 4 (07_21) chunks 0–10 (the early steps);
    - run 5 (08_15) chunks 2–3 and 75.
  - **So every test region is now stable beam,** except the last 7 shots of run 1 (below). The ~30 % gains that
    earlier versions of this README reported on runs 1 and 3 came from the level changes that are now cut; they are
    in the git history.
- **Pulse grid.** The machine fires every 1.2 s, but acquisition skips pulses, so rows are 1.2, 2.4 or 3.6 s apart.
  `pulses.align` puts each reading on its true pulse.
- **Laser meters.** A reading belongs to the previous pulse if it was read less than 2.2 s after the trigger, and to
  the same pulse otherwise (`meter_shift`). With this, lab-meter r with the charge jitter is 0.72; with a fixed row
  lag it is 0.42–0.45.
- **RF stamps are one pulse ahead.** Every RF trace holds the pulse before the one its stamp names (`rf_shift`).
  Checked on all 9 runs.
- **Beam-loading step.** PGLA mean(620–631) − mean(637–655) = 0.89 a.u. per nC. It is linear from 0.3 to 10.4 nC
  and matches the BCM within ~3 %.
- **The BCM is sometimes wrong.** It sits at a 1.67 nC ceiling in run 06_21-57-43, and stays stuck near 0 in parts
  of 09_07-46-07 and 09_11-48-33, while the beam-loading step sees full charge. Those stretches are in discarded
  chunks; the stuck-BCM check (`run_data.load_run`) still keeps the few remaining such shots out of training.
- **Missing readings.** After pulse alignment, ~10 % of lab/clex readings and 13–36 % of step readings have no pulse
  of their own (mostly 1–2-shot gaps, up to 48). `run_data.load_run` carries the last reading forward.
  - It used to interpolate. That filled shot n partly from shot n+1: a future leak into every laser and step
    feature, and fake ramps after jumps.
  - Fixing it cost the forecasts 0.001–0.003 nC. All numbers below are after the fix.
- **Correlations that come from dropouts.** Notebook 07_02 found the klystron amplitude (PKI15A) at r = 0.44 with the
  charge jitter in run 2. That is the RF trip at shot ~12 100: charge and klystron collapse together, and a 51-shot
  detrend does not remove a one-shot collapse. With all shots r = 0.40; without the 76 trip shots r = −0.03. On the
  pulse grid, klystron, reflected power and gun field have |r| ≤ 0.06 on every run. Drop dropouts before correlating.

## Results: run 2026-10-09_07-46-07, next shot (h = 1), test region = last 1070 shots (9630–10699)

| Forecast | MAE (nC) | vs baseline |
|---|---|---|
| mean of last 9 good shots (baseline) | 0.3430 | |
| mean of last 20 good shots (window picked on train) | 0.3328 | −3.0 % |
| mean of last 3 good shots | 0.3390 | −1.2 % |
| ridge: past charge + laser meters + beam-loading step | **0.3312** | −3.4 % |

- **The test region is ~10 nC stable beam, apart from its last 7 shots.** The clean span ends at chunk 106, just as the
  operators change the laser: the charge goes 10 → 5 → 4 → 1 → 18 nC over shots 10692–10699. The beam-loading step
  confirms these readings.
  - These shots carry 10 % of the baseline's summed error, and 5 % of the hybrid's.
- **On the 1063 stable test shots:** 9-shot mean 0.3096, 20-shot mean 0.3023, ridge about the same.
- **Longer horizons:** the ridge gains 3–4 % up to h = 10 and nothing from h = 20 on.
- Plots: `run_overview.png` (the whole run, discarded chunks red), `next_shot.png` (predicted vs actual next shot),
  `residuals_stable.png` (residuals on stable shots, below).

### Next-shot residuals on stable beam (`residuals_stable.png`, clean run, 6083 shots)

A shot counts as stable if all of these hold:
- it is not a dropout and the BCM is not stuck;
- it is not within 20 shots of a level change;
- the local level moves less than 10 % over ~100 shots;
- there is no bad shot in the previous 10.

The ridge here is out-of-fold over 10 contiguous blocks.

| Forecast | ~5.5 nC: std / MAE (nC) | ~10 nC: std / MAE (nC) | relative std | skew at 5.5 nC | excess kurtosis at 5.5 nC |
|---|---|---|---|---|---|
| 9-shot mean | 0.173 / 0.134 | 0.399 / 0.302 | 3.4 % | −0.16 | +0.9 |
| 20-shot mean | 0.170 / 0.132 | 0.395 / 0.298 | 3.4 % | −0.13 | +0.9 |
| ridge | 0.171 / 0.133 | 0.401 / 0.305 | 3.4 % | −0.17 | +0.9 |

- **On stable beam all three are within ~2 %.** The best window is 20–50 shots.
- **The jitter is close to Gaussian at 5.5 nC,** with slightly heavier tails and a small negative skew. The 10 nC
  part has much heavier tails (excess kurtosis ~15) from the few low shots before the final level change.
- **The jitter grows with charge:** 3.2 % rms of the level at 5.5 nC and 3.9 % at 10 nC.

## Hybrid next-shot forecaster (`hybrid_forecast.py`, `hybrid.txt`, `hybrid_next_shot.png`)

The forecast is the mean of the good shots since the last level change, capped at `wmax` shots.
- **Level change:** the charge deviates from the current mean by more than z × the stable jitter (3.1 %). The mean
  then restarts at the new shot.
- **Skipped shots:** a fall of more than 70 % that does not count as a change is treated as a dropout. A reading
  below half the beam-loading-step charge (trailing 5-shot median) is treated as a stuck BCM.
- **Causal:** the official dropout labels are used only for scoring.
- **Tuning:** z and wmax are tuned on the first 90 %.

| Next-shot forecast, run 1 | test MAE (nC) | vs baseline | test, stable | whole run, stable ~5.5 / ~10 nC |
|---|---|---|---|---|
| 9-shot mean (baseline) | 0.3430 | | 0.3096 | 0.134 / 0.302 |
| 3-shot mean | 0.3390 | −1.2 % | 0.3330 | 0.141 / 0.321 |
| 20-shot mean | 0.3328 | −3.0 % | 0.3023 | 0.132 / 0.298 |
| **hybrid, charge only (z = 6, wmax = 30)** | **0.3127** | **−8.9 %** | 0.2994 | 0.132 / 0.295 |
| hybrid, laser must confirm (z = 6, wmax = 30) | 0.3202 | −6.7 % | 0.2994 | 0.132 / 0.295 |
| hybrid, two shots in a row (z = 4, wmax = 30) | 0.3354 | −2.2 % | 0.2999 | 0.132 / 0.295 |

- **The hybrid matches the 20–50-shot mean on stable beam and reacts within a shot at the final level change.**
  Most of its gain over the 20-shot mean is those last 7 shots.
- **Robust to tuning:** test MAE stays between 0.312 and 0.318 nC for every z ≥ 4 and wmax from 20 to 100.
- **The laser does not help.** At shot 10693 the charge fell to 5 nC while the lab reading stayed at 9.7 uJ, so
  requiring the laser to confirm misses the change.
- **No method can predict the first shot at a new level.** Nothing before that shot shows the change, so the
  hybrid catches up one shot later.

## Second run: 2026-10-08_21-11-47 (`results/2026-10-08_21-11-47/`)

- **What the run looks like:** 20 000 shots, 5–6 nC throughout, with no stuck BCM and no laser steps. The lab laser
  drifts in slow sawtooth cycles every ~1500–3000 shots (9.5 → 12.5 uJ), and the charge follows it. One RF trip at
  shot ~12100 gives 76 dropouts. All chunks are kept.
- **Test region:** the last 2000 shots are all stable, with no level change.

| Next-shot forecast | run 1 (09_07) test MAE | run 2 (08_21) test MAE |
|---|---|---|
| 9-shot mean (baseline) | 0.3430 | 0.144 |
| 3-shot mean | 0.3390 (−1.2 %) | 0.155 (+7 %) |
| 20-shot mean | 0.3328 (−3.0 %) | 0.141 (−2 %) |
| ridge | 0.3312 (−3.4 %) | **0.1395 (−3.2 %)** |
| hybrid, tuned on its own run | 0.3127 (−8.9 %) (z = 6, wmax = 30) | 0.141 (−2 %) (z = 6, wmax = 20) |
| hybrid, z = 6, wmax = 20 on both | **0.3156 (−8.0 %)** | 0.141 (−2 %) |

- **One hybrid setting (z = 6, wmax = 20) works on both runs.** On calm run 2 it is the 20-shot mean, about 2 %
  better than the baseline.
- **The ridge does best on calm beam.** It uses the long means and the latest lab reading.
- **The stable-beam residuals have the same shape as in run 1:** 3.35 % rms relative, skew −0.11, excess kurtosis +1.0.

## Third run: 2026-10-08_00-08-19 (`results/2026-10-08_00-08-19/`)

- **What the run looks like:** the clean span is shots 0–19 399, at 4–6 nC, following slow lab-laser drifts
  (9–12 uJ). 5 dropouts and 5 BCM-wrong shots.
  - The rest of the run (0.3 nC, laser at 4 uJ, with short steps up to 5.4 nC) is discarded by the cuts.
- **Test region (shots 17 460–19 399):** 4.4–5.3 nC; 1924 of the 1939 scored shots are stable, with no level change.

| Next-shot forecast, run 3 test region | MAE (nC) |
|---|---|
| 9-shot mean (baseline) | 0.1414 |
| last shot | 0.1770 (+25 %) |
| 3-shot mean | 0.1497 (+5.9 %) |
| 20-shot mean (window picked on train) | 0.1388 (−1.9 %) |
| **ridge (`forecast_run.py`)** | **0.1353 (−4.3 %)** |
| hybrid, z = 6, wmax = 20 (also what run 3's own tuning picks) | 0.1392 (−1.6 %) |

- **Run 3 now looks like run 2:** calm 5 nC beam, lag-1 autocorrelation of the jitter +0.13, and the ridge is best.
- **Before the cuts** the test region was the 0.3 nC part. There the last shot was best (lag-1 +0.26), and the
  nC ridge failed (+59 %) because one fit spanned 5 nC and 0.3 nC. The combined forecaster's relative correction was
  built to avoid that.

## Runs 4 and 5: 2026-10-07_21-43-26 and 2026-10-08_15-42-27 (low charge)

Everything was tested on these runs frozen: settings and features chosen on runs 1–3.
- **Run 4 (07_21-43-26):** the clean span is shots 1100–7126 (chunks 11–71), a quiet ~0.2 nC.
  - The test region (603 shots) is all stable, with no level change.
  - The jitter is 7.7 % of the level (3–4 % at 5–10 nC).
- **Run 5 (08_15-42-27):** 12 100 shots at 0.05–0.2 nC, with steps and slow swings; 300 shots discarded,
  363 dropouts.
  - The lab laser stays flat (3.6–3.7 uJ) while the charge swings. The gun field moves from 1566 to 1577 in the second
    half, so here the charge may follow the RF rather than the laser.
  - Only 480 of the 1210 test shots count as stable; the rest drift. The jitter is 11.9 % of the level.
- **At these charges the beam-loading step is useless** (it reads ~0 nC), so the stuck-BCM check never fires.

| Next-shot test MAE (nC) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|
| 9-shot mean (baseline) | 0.0195 | 0.0122 |
| 20-shot mean | 0.0187 (−4.0 %) | 0.0119 (−2.5 %) |
| 50-shot mean (window picked on train) | 0.0182 (−6.7 %) | 0.0119 (−2.5 %) |
| **ridge (`forecast_run.py`)** | **0.0179 (−8.1 %)** | **0.0118 (−3.5 %)** |
| hybrid, z = 6, wmax = 20 | 0.0192 (−1.7 %) | 0.0119 (−2.5 %) |
| hybrid, tuned on its own run | 0.0182 (−6.7 %) (z = 6, wmax = 50) | 0.0119 (−2.6 %) (z = 8, wmax = 30) |

- **On quiet low-charge beam a longer mean helps.** At 8–12 % jitter, 20 shots is too short; run 4's own tuning
  picks wmax = 50.
- **The ridge is best on both.**

## Combined forecaster: hybrid + stable-beam correction (`combined_forecast.py`, `results/combined/`)

```bash
python3 -I scripts/combined_forecast.py data/tables results/combined 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 \
    2026-10-07_21-43-26 2026-10-08_15-42-27
```

**How it works:** forecast = hybrid level × (1 + ridge correction).
- The hybrid runs with z = 6 and wmax = 20 on every run.
- The correction predicts the hybrid's relative error on the next shot.
- Its features are computed only inside the current segment, so a level change resets them:
  - charge means over 1/3/50/200 shots, relative to the hybrid level;
  - lab and clex over 1/9 shots, minus their 200-shot segment mean, relative to that mean;
  - the step charge the same way, but divided by the hybrid level (the step sits near 0 nC at low charge).
- It is trained on the stable shots of the first 90 % of the runs, and only applied once a segment is
  ≥ 20 shots old.
- To try more inputs, add columns in `features()`.

| Next-shot test MAE (nC) | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|---|---|---|
| 9-shot mean (baseline) | 0.3430 | 0.1442 | 0.1414 | 0.0195 | 0.0122 |
| hybrid (z = 6, wmax = 20) | 0.3156 (−8.0 %) | 0.1414 (−1.9 %) | 0.1392 (−1.6 %) | 0.0192 (−1.7 %) | 0.0119 (−2.5 %) |
| ridge alone (`forecast_run.py`) | 0.3312 (−3.4 %) | 0.1395 (−3.2 %) | 0.1353 (−4.3 %) | 0.0179 (−8.1 %) | 0.0118 (−3.5 %) |
| combined, correction trained on its own run | 0.3149 (−8.2 %) | 0.1396 (−3.1 %) | 0.1357 (−4.1 %) | 0.0184 (−5.5 %) | 0.0119 (−2.7 %) |
| **combined, correction trained on all 5 runs** | **0.3163 (−7.8 %)** | **0.1399 (−3.0 %)** | **0.1371 (−3.0 %)** | **0.0183 (−5.9 %)** | **0.0119 (−3.0 %)** |
| combined, correction trained on the other 4 runs only | 0.3165 (−7.7 %) | 0.1401 (−2.8 %) | 0.1377 (−2.7 %) | 0.0183 (−6.0 %) | 0.0119 (−2.8 %) |

- **On these stable test regions the plain ridge is as good or better.** It is best on runs 2–4, and level on run 5.
  The combined forecaster only wins on run 1, through the hybrid's response at the final level change.
  - The ridge in `forecast_run.py` is fitted on its own run (in nC, with ~25 features). The combined correction
    transfers: trained on the other runs only, it loses at most 0.0006 nC against training on the run itself.
- **On stable test shots** the correction cuts the hybrid's error by 1–2 % on runs 2 and 3 (0.1399 vs 0.1414;
  0.1370 vs 0.1392) and by 5 % on run 4 (0.0183 vs 0.0192). It does nothing on run 1 (0.3030 vs 0.3023).
- **What the correction uses most:** the 200- and 50-shot segment means, the latest clex reading and the 9-shot lab
  mean.
- **Two things were needed** (found on the full runs, before the cuts):
  - Train the correction on stable shots only. Trained on all shots, it learns to chase the last few shots from
    run 1's laser scan, and run 2 got worse.
  - Switch it off for the first 20 shots of a segment.
- **A bug that run 3's 0.3 nC part caught** (before the cuts). At first the step features were relative to their own
  200-shot mean. At 0.3 nC that mean is ~0, so the features reached ~1000. Dividing by the charge level fixed it.

## GRU sequence model (`gru_forecast.py`, `results/gru/`)

```bash
python3 -I scripts/gru_forecast.py data/tables results/gru 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 \
    2026-10-07_21-43-26 2026-10-08_15-42-27
python3 -I scripts/gru_window_sweep.py data/tables results/gru/window_sweep 2026-10-09_07-46-07 2026-10-08_21-11-47 \
    2026-10-08_00-08-19                                 # the context-window sweep below
python3 -I scripts/screen_variables.py data/tables      # the notebook-variable screen below (runs 1-3)
```

**How it works:** like the ridge in the combined forecaster, the GRU corrects the hybrid level on stable beam.
- **Input:** the last 64 shots of the current segment (since the last level change), with six channels:
  - charge (0 where the hybrid skipped the shot, plus a skip flag; steps before the segment count as skipped);
  - hybrid level;
  - lab and clex laser meters;
  - beam-loading-step charge.
- **Scaling:** charges are divided by the hybrid level at the newest shot, and the meters by their window mean.
  Each channel is then divided by its training std and clipped at ±5.
- **Output:** the next shot relative to the hybrid level. The output layer starts at zero, so the untrained GRU *is*
  the hybrid.
- **Model:** one GRU layer with 32 hidden units, then a linear layer.
- **Training:**
  - L1 loss, because the metric is MAE.
  - Trained on the stable shots of the first 90 % of the runs, early-stopped on the last 10 % of those.
  - The untrained (= hybrid) state is an early-stopping candidate too, so a fit that never beats the hybrid stays
    the hybrid.
  - 3 seeds per variant, with their forecasts averaged.
- **When it applies:** once the segment is ≥ 20 shots old and the newest shot was used by the hybrid.
- **Run time:** ~1–3 min per fit on 4 CPU threads (12 threads is 9× slower), ~40 min in total for 5 runs.
- **To try more inputs, add channels in `windows()`.**

| Next-shot test MAE (nC) | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|---|---|---|
| 9-shot mean (baseline) | 0.3430 | 0.1442 | 0.1414 | 0.0195 | 0.0122 |
| hybrid (z = 6, wmax = 20) | 0.3156 (−8.0 %) | 0.1414 (−1.9 %) | 0.1392 (−1.6 %) | 0.0192 (−1.7 %) | 0.0119 (−2.5 %) |
| combined (hybrid + ridge), trained on all 5 runs | 0.3163 (−7.8 %) | 0.1399 (−3.0 %) | 0.1371 (−3.0 %) | 0.0183 (−5.9 %) | 0.0119 (−3.0 %) |
| ridge alone (`forecast_run.py`, own run) | 0.3312 (−3.4 %) | 0.1395 (−3.2 %) | 0.1353 (−4.3 %) | 0.0179 (−8.1 %) | 0.0118 (−3.5 %) |
| **GRU, pooled runs** | **0.3116 (−9.2 %)** | **0.1399 (−3.0 %)** | **0.1359 (−3.9 %)** | **0.0186 (−4.6 %)** | **0.0118 (−3.3 %)** |
| GRU, pooled runs, charge only | 0.3096 (−9.8 %) | 0.1407 (−2.4 %) | 0.1364 (−3.5 %) | 0.0186 (−4.5 %) | 0.0120 (−1.6 %) |
| GRU, other runs only (frozen test) | 0.3129 (−8.8 %) | 0.1408 (−2.3 %) | 0.1370 (−3.1 %) | 0.0186 (−4.8 %) | 0.0118 (−3.6 %) |

Seed-to-seed spread of the test MAE: ≤ 0.0017 nC. Every fit beats the hybrid on validation (e.g. 4.09 % against
4.15 % L1).

- **The GRU beats the 9-shot mean on all five runs, also trained without the run it is tested on.** It satisfies the
  Hard-tier rule: a sequence model that beats the 9-shot mean on a chronological split. The gain is significant on
  every run (bootstrap below).
- **It is the best cross-run model on runs 1, 3 and 5;** the combined forecaster is better on run 4. The own-run
  ridge is as good on runs 2, 3 and 5 and better on run 4.
- **On run 1's stable test shots** the GRU gains 3.6 % (0.2983 vs 0.3096); the rest of its 9 % is the final level
  change.
- **The laser channels add little.** Without them the GRU is slightly better on run 1 and worse on the others.

### Context window (`gru_window_sweep.py`, `results/gru/window_sweep.txt`; runs 1–3, pooled, 3 seeds)

| window L (shots) | 8 | 16 | 32 | 64 (used) | 128 | 256 |
|---|---|---|---|---|---|---|
| validation L1 (hybrid 2.87 %) | 2.83 % | 2.82 % | 2.81 % | 2.82 % | 2.82 % | 2.83 % |
| test MAE, run 1 | 0.3125 | 0.3120 | 0.3121 | 0.3130 | 0.3123 | 0.3133 |
| test MAE, run 2 | 0.1404 | 0.1405 | 0.1404 | 0.1403 | 0.1402 | 0.1403 |
| test MAE, run 3 | 0.1365 | 0.1361 | 0.1359 | 0.1358 | 0.1361 | 0.1359 |

- **The window length makes no real difference:** the test MAE moves by ≤ 0.0013 nC on run 1 and ≤ 0.0007 on the
  others, within ~2 seed spreads. The same held before the cuts.
- This fits white jitter on stable beam: the hybrid level already holds the useful history, and the GRU only adds a
  small correction from the last few shots. L = 64 is kept. L = 256 costs ~6–9 min per fit.
- **What made the difference** (full runs 1–3, before the cuts and the interpolation fix). Each fix came from
  looking at where the GRU lost:

| Step | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| first GRU | 0.366 | 0.1411 | 0.0543 |
| + clip the standardised inputs (laser steps saturated the GRU) | 0.358 | 0.1412 | 0.0501 |
| + train on stable shots only, apply after 20 shots in a segment | 0.344 | 0.1400 | 0.0504 |
| + window limited to the current segment | 0.3415 | 0.1401 | 0.0503 |
| + off when the newest shot was skipped; zero-initialised output | 0.3411 | 0.1401 | 0.0481 |

  The last fix came from run 3's 1.67 nC BCM plateau (shots 23394–23445, now discarded). There the hybrid's stuck-BCM
  check skipped every shot, and the GRU, given a window of skipped shots it had never seen in training, predicted
  13 % low.

### Do the notebook variables help? (`screen_variables.py`, `results/gru/screen_variables.txt`)

- **Same shot, they correlate as the notebooks found.** Against the charge jitter: camera intensity 0.46–0.89
  (run 3 / run 1), lab 0.61–0.62, clex 0.55–0.60, laser scope 0.38–0.45, step 0.28–0.42. Klystron, gun field, magnets,
  temperatures and humidities are all |r| < 0.08.
- **For the next shot, none of them helps, the camera included.** Added to the correction, each group changes the
  stable-beam MAE on an unseen run by −0.1 to +0.7 %. All of them together make it 0.1–0.9 % worse.
  - The jitter is laser-driven, and the next pulse's laser energy is not predictable from earlier pulses.
  - The slow variables (temperature, humidity, magnets) do not lead the charge drift by enough to matter.
- **A trap:** the laser scope shifted one pulse early correlates 0.22–0.30 with the next-shot error, more than twice
  the lab meter. That is because it is the next shot's laser reading, which is future data. Its same-shot r peaks at
  the stamp alignment, which confirms the stamp alignment. Any new variable that suddenly predicts well should get
  this check.
- The screen used to interpolate the variables across gaps, which fills shot n partly from shot n+1, the same leak
  as in `run_data`. It now carries the last reading forward. The conclusions did not change.

### Where the error is

- After the cuts the test regions are stable beam, except run 1's last 7 shots (5 % of the hybrid's summed error).
- On stable beam the remaining error is white, laser-driven jitter: lag-1 autocorrelation +0.02 to +0.13 about the
  9-shot mean.
- Before the cuts, run 1's test region had 20 % of the error in the 20 shots after a change, and run 3's 41 %, nearly
  all in the first 1–3 shots of a new level. Changes come without warning in the data.

## Are the gains real? Prediction intervals (`forecast_uncertainty.py`, `results/uncertainty/`)

```
python3 -I scripts/forecast_uncertainty.py data/tables results/gru/gru_forecasts.npz results/uncertainty \
    2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 2026-10-07_21-43-26 2026-10-08_15-42-27
```

It reads the next-shot forecasts that `gru_forecast.py` saves (`results/gru/gru_forecasts.npz`), so nothing is
retrained. It runs in ~20 s.

### Bootstrap intervals for the gains

A circular block bootstrap of the test shots (50-shot blocks, 4000 resamples) gives a 95 % interval for each MAE
reduction.
- The blocks keep the error autocorrelation.
- Wrapping round gives the last shots the same weight as the rest. With plain moving blocks, run 1's final level
  change was under-sampled, and the point estimate fell outside its own interval.
- Where the 9-shot mean has no good shot in its window (after a beam trip), it holds its last value, so that every
  forecaster is scored on the same shots.

| % MAE reduction [95 % interval] | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|---|---|---|
| hybrid vs 9-shot mean | +8.0 [+1.1, +16.9] | +1.9 [+0.3, +3.5] | +1.6 [+0.1, +3.0] | +1.7 [−1.2, +4.2] | +2.5 [+1.0, +3.7] |
| combined vs 9-shot mean | +7.8 [+0.3, +16.3] | +3.0 [+1.4, +4.7] | +3.0 [+1.3, +4.7] | +5.9 [+3.4, +8.5] | +3.0 [+1.5, +4.5] |
| GRU pooled vs 9-shot mean | +9.2 [+2.0, +17.5] | +3.0 [+1.4, +4.6] | +3.9 [+2.3, +5.5] | +4.6 [+1.8, +7.0] | +3.3 [+1.9, +4.7] |
| GRU other runs only vs 9-shot mean | +8.8 [+1.6, +17.5] | +2.3 [+0.7, +3.9] | +3.1 [+1.4, +4.7] | +4.8 [+2.0, +7.1] | +3.6 [+2.3, +4.7] |
| combined vs hybrid | −0.2 [−2.3, +2.0] | +1.1 [−0.2, +2.3] | +1.5 [−0.0, +3.0] | +4.3 [+2.8, +5.7] | +0.6 [−0.9, +2.0] |
| GRU pooled vs hybrid | +1.3 [−0.0, +2.7] | +1.1 [−0.1, +2.2] | +2.3 [+1.0, +3.8] | +2.9 [+1.5, +4.4] | +0.8 [−0.2, +1.9] |

- **The GRU beats the 9-shot mean significantly on all five runs,** also when trained without the test run. This is
  the Hard-tier claim. The combined forecaster does too, though only just on run 1.
- **Run 1's interval is wide** because most of its gain is the 7 shots of the final level change.
- **The correction on top of the hybrid is 0–4 %.** It is significant on run 4 for both models and on run 3 for the
  GRU; elsewhere it cannot be told apart from 0. The combined forecaster and the GRU cannot be told apart anywhere.
- **With 200-shot blocks the conclusions are the same.** The hybrid on run 4 stays not significant, and the GRU vs
  hybrid on run 1 becomes just significant.

### 90 % prediction intervals

The interval is forecast × (1 + [5 %, 95 %] quantiles of the relative residual), so it scales with the charge. Two
calibrations, both causal:
- **rolling:** the last 500 good residuals of the run, each used only once its shot has been measured. It follows the
  jitter when the regime changes.
- **fixed:** the quantiles of the run's own training part.

Forecasts made fewer than 5 shots into a new segment use quantiles pooled over such forecasts in all runs' training
parts. After the cuts there are only 71 of them, and their quantiles (−15 % / +387 %) are too unstable to use; they
only matter for run 1's last few shots.

| combined forecaster, test region | run 1 | run 2 | run 3 | run 4 | run 5 |
|---|---|---|---|---|---|
| coverage, rolling (nominal 90 %) | 90.0 % | 89.8 % | 89.4 % | 87.2 % | 91.9 % |
| mean width, rolling (% of level) | 14.2 | 10.1 | 12.9 | 32.2 | 54.2 |
| coverage, fixed | 83.6 % | 91.1 % | 87.6 % | 82.6 % | 83.9 % |

- **The rolling calibration holds 87–92 % on all runs.** The fixed one drops to 83–84 % on runs 1, 4 and 5, where
  the jitter in the test region differs from the training part.
- **The width follows the relative jitter:** ±5–7 % at 5–10 nC, ±16 % at 0.2 nC (run 4) and ±27 % at 0.1 nC (run 5).
- **The first shot of a level change falls outside any interval of sensible width.** Before the cuts, coverage in
  the 20 shots after changes was ~80 %.

## Same-shot virtual BCM (`virtual_bcm.py`, `results/virtual_bcm/`)

```bash
python3 -I scripts/camera_features.py data/h5 results/datasets data/tables 2026-10-09_07-46-07 2026-10-08_21-11-47 \
    2026-10-08_00-08-19 2026-10-07_21-43-26 2026-10-08_15-42-27      # optional: the camera inputs below
python3 -I scripts/virtual_bcm.py data/tables results/virtual_bcm 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 \
    2026-10-07_21-43-26 2026-10-08_15-42-27
```

**What it is:** an estimate of shot n's charge from shot n's own laser and camera readings, plus the BCM history up
to shot n−1.
It is not a forecast, because it needs shot n's instruments. Uses: a cross-check on the BCM, or a stand-in while
the BCM reads wrong.

**How it works:** log(Q[n] / hybrid level after shot n−1) = a ridge on
- the combined forecaster's history features;
- for lab, clex and the laser scope: log(reading[n] / its segment mean over the last 20 and 200 shots);
- the same for the camera features, if `<run>_camera.npz` exists: screen intensity, peak amplitude, spot sizes along
  the spot's axes, and the 6 largest singular values of the binned image.

A missing reading gives a feature of 0. The ridge is trained on all good shots of the first 90 %, not only stable
ones, because level changes are where it should help. There are two ridges: one for when the hybrid level is above
1 nC, one for below it.

| MAE against the BCM, test region (nC) | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|---|---|---|
| hybrid next-shot forecast (for scale) | 0.3156 | 0.1414 | 0.1392 | 0.0192 | 0.0119 |
| virtual BCM, lab only, pooled runs | 0.258 (−18 %) | 0.1178 (−17 %) | 0.1336 (−4 %) | 0.0167 (−13 %) | 0.0116 (−3 %) |
| **virtual BCM, lab + clex + scope, pooled runs** | **0.252 (−20 %)** | **0.1165 (−18 %)** | **0.1315 (−5 %)** | **0.0121 (−37 %)** | **0.0103 (−13 %)** |
| virtual BCM, lab + clex + scope, other runs only | 0.251 (−20 %) | 0.1169 (−17 %) | 0.1320 (−5 %) | 0.0121 (−37 %) | 0.0107 (−10 %) |
| virtual BCM, camera intensity only, pooled runs | 0.199 (−37 %) | 0.0967 (−32 %) | **0.0889 (−36 %)** | 0.0115 (−40 %) | 0.0093 (−22 %) |
| virtual BCM, lab + clex + scope + camera intensity, pooled runs | 0.194 (−39 %) | 0.0935 (−34 %) | 0.1013 (−27 %) | 0.0105 (−45 %) | 0.0091 (−24 %) |
| **virtual BCM, lab + clex + scope + all camera features, pooled runs** | **0.177 (−44 %)** | **0.0903 (−36 %)** | 0.0913 (−34 %) | **0.0099 (−48 %)** | **0.0087 (−27 %)** |
| virtual BCM, lab + clex + scope + all camera features, other runs only | 0.178 (−44 %) | 0.0925 (−35 %) | 0.0831 (−40 %) | 0.0098 (−49 %) | 0.0095 (−20 %) |
| BCM lost at the cut: laser × frozen charge/laser ratio | 0.232 | 0.1655 | 0.1701 | 0.0199 | 0.0234 |
| BCM lost at the cut: camera × frozen charge/intensity ratio | 0.261 | 0.1692 | 0.1018 | 0.0121 | 0.0103 |
| BCM lost at the cut: step − frozen offset | 0.344 | 0.355 | 0.363 | 0.347 | 0.364 |

- **On stable beam, seeing the shot's own laser cuts the error by 18–21 % on runs 1 and 2:** run 1 0.238 vs 0.302 nC,
  run 2 0.117 vs 0.141. This is the laser-driven jitter that no forecast can see coming.
- **On run 3 it gains only 5 %,** and on the low-charge runs 37 % (run 4) and 13 % (run 5).
- **It transfers.** Trained on the other runs only, it is within 0.0005 nC of the pooled model on runs 1–4.
- **Why two ridges** (found before the cuts). The laser→charge relation differs below 1 nC; one ridge for all charges
  made runs 1–2 worse.
  - Below 1 nC the clex meter carries most of the signal: clex vs its 20-shot mean +3.4 % per std, against lab
    features at ~1 %. The lab meter sits before the attenuation used for low charge.
  - Above 1 nC it is the lab meter: lab vs its 20-shot mean +1.5 %, then the 200-shot charge mean +0.7 % and clex vs
    its 200-shot mean +0.5 %.
- **The beam-loading step is not used.** Its noise is roughly fixed in nC, so at 0.3 nC it is 8× noisier relative
  to the level than at 5 nC. Trained on the high-charge runs, the model trusted it too much on run 3's 0.3 nC part
  (before the cuts: 0.0511 nC with it, 0.0472 without). On runs 1 and 2 it added < 0.3 %.
- **With no BCM at all after the cut,** both stand-ins are calibrated once, on the last 200 good training shots.
  - The laser ratio holds where the test region stays in the same regime as the end of training: on run 1 it is
    better than the hybrid (0.232 vs 0.316), and on run 4 it matches it.
  - It fails on runs 2, 3 and 5 (17–96 % worse than the hybrid), because the charge per uJ drifts.
  - The step stand-in fails everywhere.
- **The camera roughly doubles the gain.** With all camera features the error falls by 27–48 % against the hybrid,
  against 5–37 % from the laser alone. On stable beam: run 1 0.168 vs 0.302 nC, run 2 0.090 vs 0.141, run 3 0.091
  vs 0.139.
  - The screen sees the beam itself, so it is a second charge measurement rather than a laser proxy. Its intensity
    follows the charge with a slope of 0.5–1.2 in log, and with r = 0.78–0.93 in the test regions.
  - It transfers: trained on the other runs only, it is as good on runs 1 and 4, better on run 3, and worse on runs 2
    and 5 by 0.001–0.002 nC.
  - The spot shape and singular values add 3–9 % beyond the intensity on runs 1, 2, 4 and 5. On run 3 the intensity
    alone is best, and adding the laser makes it worse there.
- **Camera alignment** (`camera_features.py`, `results/virtual_bcm/camera_features.txt`):
  - The camera's own cycle stamp names its pulse. For 0.3–21 % of rows (by run) it is one or two pulses after the
    row's trigger, so the row is not the image's pulse.
  - At the stamp alignment the intensity's r with the charge jitter is 0.35–0.90; one pulse either side gives ≤ 0.30.
  - `extract_features.py` leaves the camera on its row, which matches the stamp for most rows.
- **Dark frames.** 0–1 % of images read 1–20 % of the usual intensity while the BCM and the laser are normal. Each one
  gives a huge log-ratio feature, and in the training parts of runs 3–5 they cut the camera's r from ~0.85 to 0.14–0.20,
  so the ridge learned to ignore the camera. `camera_features.py` drops a frame below 30 % of the median of the
  previous 50 when the lab laser has not dropped. Milder outliers remain: Spearman r is still above Pearson r there.
- **With no BCM, the camera is the better stand-in on runs 3–5** (17–37 % better than the hybrid). Its intensity per
  nC drifts too: on run 2 it is 20 % worse than the hybrid. The laser ratio is still better on run 1.
- **How the interpolation leak was found.** The first version overshot after big jumps up (30 nC predicted against
  18 nC). The interpolated meters ramped linearly across gaps after a jump, so shot n's "reading" was partly shot n+1's.

## Same-shot charge from the reduced feature set, per regime (`same_shot_regression.py`, `results/same_shot/`)

```bash
python3 -I scripts/same_shot_regression.py data/tables results/same_shot 2026-10-09_07-46-07 2026-10-08_21-11-47 \
    2026-10-08_00-08-19 2026-10-07_21-43-26 2026-10-08_15-42-27            # needs <run>_camera.npz (camera_features.py)
```

This is the first step of the model comparison: a baseline for how well the shot's own readings give its charge,
with no BCM history and no forecasting.

**Inputs:** the reduced feature set of `extract_features.py` on our pulse grid. The windowed beam-loading step stands
in for `PGLA_drop`. A second set adds clex and the laser scope.
- Positive readings enter as logs, because the charge scales with laser energy and the screen light with the charge.
- A missing reading takes the segment's last one, plus a missing flag per instrument.

**Model:** log Q = a ridge per regime. The regularisation is chosen by 5-fold CV over contiguous blocks.

**Regimes (clean segments of `results/datasets`):**
- low charge: 0, 2, 3, 4;
- high, repetitive: 5;
- high, non-repetitive: 1, 6, 7.

**Split:** the first 90 % of each segment trains, the last 10 % tests.

**Four ways to use a regime's segments:**
- **absolute:** one intercept for the regime;
- **per-segment calibration:** each segment's own intercept, from its training part, with shared slopes (fixed
  effects);
- **new segment:** a segment never seen; slopes and intercept come from the regime's other segments;
- **new segment, calibrated:** slopes from the other segments, the intercept from this segment's training part.

| MAE in % of the level, test regions | seg 0 (low) | seg 3 (low) | seg 4 (low) | seg 5 (rep.) | seg 1 (non-rep.) | seg 6 (non-rep.) | seg 7 (non-rep.) |
|---|---|---|---|---|---|---|---|
| level (nC) | 0.25 | 0.13 | 0.11 | 5.9 | 4.8 | 5.7 | 10.4 |
| constant: the segment's training mean | 11.7 | 12.5 | 19.8 | 7.8 | 5.1 | 3.4 | 4.6 |
| lab only, per-segment calibration | 9.9 | 11.7 | 16.7 | 2.3 | 3.2 | **1.4** | 4.6 |
| reduced set, absolute | 19.2 | 16.0 | 14.5 | 1.5 | 3.3 | 2.3 | 4.3 |
| **reduced set, per-segment calibration** | **5.7** | 7.4 | 8.5 | **1.5** | 2.4 | 1.6 | 3.4 |
| reduced set + clex, scope, per-segment calibration | 5.9 | 7.4 | **8.0** | **1.5** | 2.5 | 1.7 | **3.2** |
| reduced set + clex, scope, new segment | 60 | 25 | 29 | – | 12.8 | 12.0 | 13.7 |
| reduced set + clex, scope, new segment, calibrated | 5.9 | 7.4 | **8.0** | – | **1.7** | 1.8 | **3.2** |

Segment 2 has only 20 test shots, so it is left out of the table (it is in `same_shot_regression.txt`). The r of the
detrended jitter is in the text file too.

- **Calibration is the main problem, not the model.**
  - With per-segment calibration the shot's own readings cut the error by a factor of 1.4–5 against the
    segment's mean.
  - Without it, a never-seen segment is off by 12–60 %, although the prediction has the right shape
    (`same_shot_regression.png`). The camera's intensity per nC differs ~10× between runs, and the lab meter sits
    before the low-charge attenuator.
  - Fitting only the intercept on the new segment's own BCM fixes nearly all of it. Slopes learned on other segments
    transfer.
- **The regimes behave as expected:**
  - **Repetitive (seg 5):** easiest. 1.5 %, jitter r 0.79, mostly the camera intensity and the lab meter.
  - **Low charge:** 6–8 %, against a relative jitter of 8–12 %. The signal is in the camera and clex. The lab meter
    alone does little there.
  - **Non-repetitive:** 1.6–3.4 %.
    - On seg 6 the lab meter alone is best: 1.4 %, against 1.6 % with the camera.
    - On seg 1, slopes from 09_07 beat the ones trained with seg 1's own training part (1.7 % against 2.4 %). The
      camera is noisier in that training part.
- **Fixed effects also give the right slopes.** With one intercept per regime, the slopes partly explain the level
  differences between segments, and the jitter r drops (seg 0: 0.48 against 0.86).
- **The non-repetitive camera coefficients are suspect:** SVD s4 +11 % and SVD s6 −11 % per std, opposite in sign.
  That is the collinear block that PLS or a group penalty should tame (next step).
- **Seg 7's final jump to 18 nC is overshot** (~24 nC predicted). The log-linear map extrapolates beyond the
  training range.

- **Where the error is** (the error split at the end of `same_shot_regression.txt`). Each residual log(pred / BCM)
  is split into a slow part (its 51-shot centred median) and the fast rest.
  - With per-segment calibration, the slow part is 2–16 % of the mean-square error on every segment but seg 0
    (51 %, a level offset). The model explains 27–76 % of the BCM's own shot-to-shot jitter: 63–76 % on segs 5 and
    7, 32–43 % on segs 1 and 6, 27–28 % on segs 3 and 4, and 70 % on seg 0.
  - For a never-seen, uncalibrated segment, the slow part is 56–100 %.
  - So once a segment is calibrated, what is left is jitter. A plain-language write-up and the slow/fast plan are in
    `../docs/charge_models_explained.md`.

## Shot-to-shot jitter with sequence models (`jitter_sequence.py`, `results/jitter_sequence/`)

```bash
python3 -I scripts/jitter_sequence.py data/tables results/jitter_sequence 2026-10-09_07-46-07 2026-10-08_21-11-47 \
    2026-10-08_00-08-19 2026-10-07_21-43-26 2026-10-08_15-42-27            # needs PyTorch and <run>_camera.npz; ~30 min
```

This is the fast-model step of the slow/fast plan in `../docs/charge_models_explained.md`. A notebook version on the team
datasets (`results/datasets/`), laid out like `08_simple_models.ipynb`, is `../notebooks/08_01_jitter_models.ipynb`. It is still same-shot:
nothing is forecast.

**Causal slow/fast split**, within each clean segment:
- slow = the mean of the previous 50 readings (of the logs; linear for step, mean_x and tilt);
- fast = the reading − slow.

**Target:** log Q[n] − the mean log Q of the previous 50 good shots, i.e. the jitter against past BCM readings.
- The level is that trailing mean, so the total charge = exp(level + predicted jitter).
- This needs the BCM up to the previous shot. The `same_shot_regression.py` numbers do not, so the rel % below are
  not comparable with them.

**Inputs:** the fast parts of the reduced set + clex and scope, 0 where missing, plus a missing flag per instrument.
The sequence models see the current shot and the L − 1 before it, within the segment.

**Models,** one per regime, split as in `same_shot_regression.py`:
- ridge on the current shot only, or with the last 16 BCM jitters added (past BCM);
- `torch.nn.RNN` and `torch.nn.GRU` (32 hidden), set up as `gru_forecast.py`: L1 loss, zero-initialised output,
  early stopping on the last 10 % of each segment's training part, 3 seeds averaged.

**Jitter explained** = 1 − mean square of (target − prediction) / mean square of the target.

| Test regions | seg 0 (low) | seg 3 (low) | seg 4 (low) | seg 5 (rep.) | seg 1 (non-rep.) | seg 6 (non-rep.) | seg 7 (non-rep.) |
|---|---|---|---|---|---|---|---|
| target rms (% of level) | 9.5 | 13.5 | 14.1 | 3.2 | 3.9 | 2.8 | 9.1 |
| jitter explained: ridge, current shot | 68 | 28 | 39 | 74 | 63 | 79 | **89** |
| jitter explained: RNN, L = 1 | **74** | **30** | 46 | **89** | 83 | 92 | 39 |
| jitter explained: GRU, L = 1 | **74** | 29 | **49** | **89** | 83 | 92 | 42 |
| jitter explained: GRU, L = 16 | 73 | 28 | 47 | **89** | 83 | **94** | 44 |
| jitter explained: GRU, L = 64 | **74** | 28 | 48 | 88 | 83 | **94** | 45 |
| MAE, % of level: trailing level only | 7.4 | 9.4 | 10.5 | 2.5 | 3.0 | 2.2 | 4.1 |
| MAE, % of level: ridge, current shot | 4.1 | 7.1 | 7.9 | 1.2 | 1.8 | 1.0 | **1.7** |
| MAE, % of level: GRU, L = 1 | 3.7 | 7.0 | **7.0** | **0.7** | **1.0** | 0.6 | 2.3 |
| MAE, % of level: GRU, L = 16 | 3.7 | 7.0 | 7.3 | **0.7** | **1.0** | **0.55** | 2.2 |

The RNN rows for L = 16 and 64 are within a point of the GRU's (`jitter_sequence.txt`). Seg 2 (20 test shots) is
only in the text file.

- **The gain is the nonlinearity, not the memory.** A one-shot RNN or GRU (L = 1) beats the ridge on every segment
  but seg 7: 83 % against 63 % on seg 1, 89 % against 74 % on seg 5, and 92 % against 79 % on seg 6. Windows of 16
  or 64 shots add at most 2 points (seg 6), as the forecasting work found.
- **RNN ≈ GRU.** With a single shot the gates have nothing to do, and with longer windows the plain RNN is within
  1–8 points. It is 2–4× faster to train.
- **Seg 7 is lost only on its last 5 shots,** the jump to 18 nC (+55 % jitter). The networks saturate there (+15 %),
  because their inputs are clipped at 5 training std and the units are tanh. The ridge extrapolates (+42 %). Before
  the jump the networks match the ridge (`jitter_sequence.png`). An RNN on the ridge's residual, or a wider clip,
  would keep both.
- **Low charge stays hard:** 28–49 % on segs 3 and 4. Part of that 13–14 % jitter is probably the BCM's own noise.

**Ridge + GRU, and past BCM.** Two more variants:
- **Ridge + GRU:** the GRU learns the ridge's residual, so the ridge keeps extrapolating where the GRU saturates.
- **Past BCM:** each window step also carries the previous good shot's BCM jitter, never the current one's. The
  ridge gets the last 16.

| MAE (pC), test regions | seg 0 | seg 3 | seg 4 | seg 5 | seg 1 | seg 6 | seg 7 |
|---|---|---|---|---|---|---|---|
| trailing level only | 18.2 | 12.4 | 11.7 | 145.5 | 143.3 | 122.2 | 428.1 |
| ridge, current shot | 10.0 | 9.4 | 8.9 | 69.5 | 84.6 | 57.0 | 174.4 |
| ridge, + past BCM | 10.0 | 9.4 | 8.9 | 69.7 | 84.4 | 56.7 | 177.0 |
| GRU, L = 16 | 9.1 | 9.2 | 8.1 | 42.1 | 47.2 | 31.2 | 226.3 |
| GRU, L = 16, past BCM | 9.3 | 9.2 | 8.1 | 41.1 | 46.1 | **30.5** | 223.8 |
| ridge + GRU, L = 1 | **9.0** | 9.2 | **7.9** | 41.6 | 46.6 | 35.7 | 158.2 |
| ridge + GRU, L = 16 | 9.1 | 9.2 | 8.1 | 42.7 | 46.9 | 32.8 | 150.2 |
| **ridge, + past BCM + GRU, L = 64, past BCM** | 9.2 | **9.1** | 8.1 | **40.8** | **45.7** | 31.0 | **147.0** |

- **Ridge + GRU fixes seg 7:** 147–158 pC against 226 for the GRU alone and 174 for the ridge (91–92 % of the jitter
  explained, against 44 % and 89 %). Elsewhere it matches the GRU alone.
- **Past BCM adds almost nothing:** at most 1.4 pC (3 %) on segs 1 and 5, and no change for the ridge. Against the
  trailing level, the BCM jitter has no shot-to-shot memory that the instruments do not already carry. These
  differences are not tested against seed noise.
- **Best overall:** the last row, within 0.5 pC of the best on every segment. It leaves the gaps to the BCM noise
  floor (next section) almost unchanged: residual rms 1.06 % against ~0.7 % on seg 5, 1.52 % against ~1.2 % on seg 1,
  and 11.3 % against ~9.3 % on seg 3.

## How low can the jitter error go? The BCM's own noise (`bcm_noise_floor.py`, `results/noise_floor/`)

```bash
python3 -I scripts/bcm_noise_floor.py data/tables 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 \
    2026-10-07_21-43-26 2026-10-08_15-42-27
```

No model can predict the BCM's own measurement noise, so that noise is the floor for the residual of any jitter model.
It is estimated with a three-cornered hat:
- **Readings:** the BCM, the camera intensity and a third reading of the same shot. Each jitter (as in
  `jitter_sequence.py`) = gain × the true charge jitter + its own noise.
- **Formula:** with independent noises, noise variance of a = var(a) − cov(a, b) cov(a, c) / cov(b, c). The gains
  drop out.
- **The third reading** need only correlate with the charge. The beam-loading step, clex and lab are each tried, and
  their agreement is the check.
  - The step is useless at low charge (r ≈ 0.02 with the others).
  - clex and lab agree to within 0.6 points at low charge, and to within 0.35 points at high charge.
- **Spikes:** any reading beyond 5 MAD within its segment (0.36 % of shots) is removed.
- **Intervals:** 95 % block bootstrap.

| rms in % of the level | seg 0 (low) | seg 3 (low) | seg 4 (low) | seg 5 (rep.) | seg 1 (non-rep.) | seg 6 (non-rep.) |
|---|---|---|---|---|---|---|
| BCM jitter (target rms, test region) | 9.5 | 13.5 | 14.1 | 3.2 | 3.9 | 2.8 |
| **BCM noise, test region** (clex / lab as third) | 4.8 / 4.8 | 9.6 / 9.0 | 11.5 / 10.6 (whole seg.) | 0.77 / 0.68 | 1.26 / 1.19 | 0.95 / 0.61 |
| residual, ridge (`jitter_sequence.py`) | 5.4 | 11.4 | 11.1 | 1.6 | 2.4 | 1.3 |
| residual, GRU L = 16 | **4.9** | 11.4 | **10.2** | 1.1 | 1.6 | **0.70** |

The residual rms = target rms × √(1 − jitter explained). Seg 4's test region has too few shots with all three readings,
so its whole-segment value is shown. Seg 7 (1.2–1.4 % over the whole segment) is dominated by its final jump.

- **At low charge about half of the BCM jitter is the BCM's own noise:** 40 % on seg 0, 50–53 % on segs 3 and 4, by
  variance.
- **At high charge it is small:** 0.5–1.1 %, i.e. 2–11 % of the jitter variance. The three third readings
  disagree by up to 2× there (step 0.35–0.6 %, clex 0.84–1.06 %), so the one-factor model is only approximate.
- **The GRU is at the floor on segs 0, 4 and 6.** Nothing on those segments can get much lower.
- **Room is left on segs 5, 1 and 3:**
  - seg 5: residual 1.1 % vs a floor of ~0.7 %;
  - seg 1: 1.6 % vs ~1.2 %;
  - seg 3: 11.4 % vs ~9.3 %.

  Removing all the model error would cut the residual rms by about a third on seg 5, a quarter on seg 1 and a fifth
  on seg 3: roughly 42 → 28 pC, 47 → 36 pC and 9.3 → 7.6 pC MAE, if the MAE scales with the rms.
- **What "noise" means here:** the part of the BCM jitter that the camera does not share. Real charge the BCM sees but
  the screen does not, such as losses between the gun and the screen, counts as noise. So this is the floor for
  models built on these instruments.

## More inputs for the jitter? Laser shape and RF (`jitter_input_screen.py`, `results/jitter_input_screen/`)

```bash
python3 -I scripts/jitter_input_screen.py data/tables 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 \
    2026-10-07_21-43-26 2026-10-08_15-42-27
```

**Laser features of `extract_features.py`:** there is nothing new to add.
- Its laser outputs are `lab_laser_power` (our lab) and `laser_amp`.
- `laser_amp` is the same matched filter on the laser scope as our `scope` input: same baseline and window. It differs
  in three ways:
  - its template is per segment rather than per run, a constant factor that the fast part removes;
  - saturated and not-acquired pulses are not masked;
  - it uses the team's row + 1 alignment.

The screen tests two groups of readings that are not used yet. Each is added to the current-shot ridge of
`jitter_sequence.py`, with the same target, fast parts, split and regimes:
- **laser shape:** the scope pulse beyond its amplitude: 5 principal components of pulse / amplitude, centroid and
  width;
- **RF:** flat-top means before the beam, and their slopes: gun incident amplitude and phase, gun reflected, klystron
  phase and reflected.

| Jitter explained %, test regions | seg 0 | seg 3 | seg 4 | seg 5 | seg 1 | seg 6 | seg 7 |
|---|---|---|---|---|---|---|---|
| ridge | 68.1 | 28.3 | 38.5 | 74.1 | 63.3 | 79.0 | 89.0 |
| ridge + laser shape | 68.1 | 28.2 | 38.5 | 74.1 | 63.5 | 79.0 | 88.8 |
| ridge + RF | 68.2 | 28.4 | 38.3 | 74.1 | 63.3 | 79.1 | 89.0 |

- **Neither group helps.** No feature correlates with the ridge's residual by more than |r| = 0.055, and the jitter
  explained moves by less than 1.2 points.
- **Laser:** lab, clex and the scope amplitude carry all the laser information that matters here. The pulse shape on
  the scope probably shows the photodiode's response more than the laser's.
- **RF:** the gun and klystron flat tops do not vary from shot to shot in a way the charge feels.
- **What is left:** the remaining gaps to the BCM noise floor on segs 5, 1 and 3 must come from the camera, from
  nonlinear structure (the GRU already takes some), or from a floor that is higher than estimated. The high-charge
  floor is uncertain by ~2×.

## Online correction: refit after every shot (`jitter_online.py`, `results/jitter_online/`)

```bash
python3 -I scripts/jitter_online.py data/tables results/jitter_sequence 2026-10-07_21-43-26 2026-10-08_00-08-19 \
    2026-10-08_15-42-27 2026-10-08_21-11-47 2026-10-09_07-46-07
```

`jitter_sequence.py` now saves every model's predicted jitter to `results/jitter_sequence/jitter_predictions.parquet`.
The trailing level already assumes the BCM is read up to the previous shot, so the offline model's recent errors are
known too. After every shot, an exponentially weighted ridge on the residual is refitted on the same inputs (+ a
constant), and the next shot's prediction is corrected:

    w_n = argmin_w  sum_{m<n} LAM^(n-1-m) (r_m - w.z_m)^2 + RHO |w|^2,   r = target - offline prediction

- Each segment starts from the offline model (w = 0), and a shot only uses the shots before it.
- LAM (memory 1000–10000 shots) and RHO (10–1000) are picked per regime on the early-stopping rows, and so is "no
  correction". The test regions are never used for the choice.

| MAE pC, test regions | seg 0 | seg 3 | seg 4 | seg 5 | seg 1 | seg 6 | seg 7 |
|---|---|---|---|---|---|---|---|
| ridge, current shot | 10.0 | 9.4 | 8.9 | 69.5 | 84.6 | 57.0 | 174.4 |
| ridge + online | 9.3 | 9.4 | 8.6 | 60.4 | 65.8 | 32.4 | 131.4 |
| best offline: ridge + past BCM + GRU, L=64 | 9.2 | 9.1 | 8.1 | 40.8 | 45.7 | 31.0 | 147.0 |
| best offline + online | 9.1 | 9.2 | 8.1 | 40.7 | 45.8 | **27.8** | **133.2** |

- **On the ridge, online helps a lot:** −22 % on seg 1, −43 % on seg 6 and −25 % on seg 7. The ridge's coefficients
  drift within a run, and the refit follows them.
- **On the best model, online helps only on segs 6 and 7:** 31.0 → 27.8 pC and 147 → 133 pC (seg 7 has 10 nC, a level
  the training never saw). Elsewhere the GRU has already taken what a linear refit can find: on the low-charge and
  repetitive regimes, the grid's best choice is within 1 % of no correction.
- **Recommendation:** use the best offline model + online correction. It costs one 25 × 25 solve per shot and never
  does worse than the offline model on these test regions by more than 0.1 pC.

## Is the BCM noise Gaussian? (`jitter_noise_plot.py`, `results/noise_floor/jitter_noise.png`, `jitter_noise.txt`)

```bash
python3 -I scripts/jitter_noise_plot.py data/tables results/jitter_sequence results/noise_floor/jitter_noise.png \
    2026-10-07_21-43-26 2026-10-08_00-08-19 2026-10-08_15-42-27 2026-10-08_21-11-47 2026-10-09_07-46-07
```

The three-cornered hat gives the noise's width, not the noise of each shot, so its shape is checked two ways on the
test regions (log-y histograms in the plot):
- **the best model's residual:** where the model is at the floor (segs 0, 4, 6), the residual is mostly noise;
- **d = BCM − k camera,** with k = cov(BCM, lab) / cov(camera, lab): this cancels the charge and leaves the BCM noise
  plus the camera noise, with no model involved.

| test regions | seg 0 | seg 3 | seg 4 | seg 5 | seg 1 | seg 6 | seg 7 |
|---|---|---|---|---|---|---|---|
| BCM noise, % of level (hat) | 4.78 | 8.96 | 9.80 | 0.68 | 1.19 | 0.61 | 0.71 |
| BCM noise, pC | 11.7 | 11.8 | 11.0 | 40 | 57 | 34 | 73 |
| residual excess kurtosis | 2.0 | 14.4 | 5.6 | 21.6 | 9.7 | 2.4 | 34.8 |
| residual beyond 4σ, % (Gaussian 0.006) | 0.50 | 0.57 | 0.22 | 0.95 | 1.19 | 0.24 | 1.50 |
| residual lag-1 autocorrelation | 0.02 | 0.24 | 0.07 | 0.04 | 0.14 | 0.02 | 0.22 |
| d excess kurtosis | 1.5 | 12.8 | 0.3 | 2.1 | 106 | 2.9 | 0.6 |

- **The core is Gaussian:** the residual follows the hat's Gaussian down to a few % of the peak on segs 0, 4, 5 and 6.
- **The tails are heavier:** 0.2–1.5 % of shots lie beyond 4σ, against 0.006 % for a Gaussian. The large kurtosis
  values come from a handful of shots (seg 1's d: a few camera glitches). At low charge the tails lean negative (skew
  −0.5 to −2): additive noise in pC becomes skewed in log, and some shots lose charge.
- **The noise is white:** lag-1 autocorrelation ≤ 0.07 where the model is at the floor. Segs 1, 3 and 7 (0.14–0.24)
  have correlated residual: that is model error, not noise, and it matches their gap to the floor.
- **The noise is a fixed ~11.5 pC plus ~0.6–0.7 % of the charge:** the low-charge segments have the same noise in pC
  (11.0–11.8 pC at 0.11–0.25 nC), so it is additive electronic noise. At high charge it is 34–73 pC, about
  sqrt(11.5² + (0.006 Q)²); seg 1 is ~2× higher.

**For a Gaussian process:** a Gaussian likelihood fits the core. Use a white-noise term with its variance fixed (or
given a tight prior) from the hat, σ_n(Q)² ≈ (11.5 pC)² + (0.0065 Q)², rather than learned: a learned noise term absorbs
real jitter. No correlated noise term is needed. For the ~1 % tail shots, use a Student-t likelihood or clip the shots
beyond 5 MAD before fitting.

## Squeezing the noise estimate (`bcm_noise_model.py`, `results/noise_floor/bcm_noise_model.txt`)

```bash
python3 -I scripts/bcm_noise_model.py data/tables 2026-10-07_21-43-26 2026-10-08_00-08-19 2026-10-08_15-42-27 \
    2026-10-08_21-11-47 2026-10-09_07-46-07
```

Training parts only, so a model may use the results. Four additions to the three-cornered hat:

1. **A stronger third reading.** The hat works for any third reading X whose noise is independent of the BCM's and the
   camera's, whatever part of the charge X sees. So X can be a combination: lab, clex, scope and step, fitted to the
   BCM jitter. The fit is cross-fitted (weights from the other half of the segment), so it cannot pick up this half's
   BCM noise.
   - At low charge, r(BCM, X) rises from 0.16–0.32 (lab) to 0.31–0.68, and the intervals narrow by up to 2×. For
     example, seg 0 goes from 5.12 % [4.89, 5.32] to 4.96 % [4.81, 5.08].
   - At high charge, lab alone is already as good.
2. **Noise against charge:** σ_n(Q)² = a² + (b Q)², fitted across segments.
   - **a = 10.7 pC [10.5, 11.0] and b = 0.72 % [0.66, 0.78]** (95 % bootstrap).
   - The fit matches every segment within 10 % except seg 7: 99 pC against 74 pC, from only 1381 shots with an interval
     of 68–119 pC.
   - Seg 1's training part gives 39 pC. The 57 pC in the table above came from its test region.
3. **Is the noise Gaussian? (no model involved)** The same algebra one order up gives the noise's own 3rd and 4th
   cumulants:

       k3(n_B) = k3(B) − cum(BBC)·cum(BBX)/cum(BCX),   k4(n_B) = k4(B) − cum(BBBC)·cum(BBBX)/cum(BBCX)

   This only works if the true charge jitter is itself non-Gaussian, and it is: its excess kurtosis is 0.6–1.5.
   A synthetic check (Gaussian noise on an exponential charge) recovers zero.

   | X = comb, spikes removed | seg 0 | seg 3 | seg 4 | high charge (1, 5, 6, 7) |
   |---|---|---|---|---|
   | noise skewness | −0.33 [−0.63, 0.10] | −0.62 [−0.78, −0.46] | −0.6 [−2.3, 1.8] | not identified |
   | noise excess kurtosis | 1.5 [1.1, 1.9] | 2.9 [2.3, 3.4] | 3.3 [−5, 9] | not identified (intervals ±50 or more) |

   - **At low charge, the noise has heavier tails than a Gaussian** (excess kurtosis 1.5–3), and seg 3 is slightly
     skewed towards low readings.
   - Taking the log of additive Gaussian noise of 5–9 % would give a skew of −0.15 to −0.28 and an excess kurtosis of
     0.06–0.19. About half of seg 3's skew (−0.62) is the log; the kurtosis is far too large to come from it.
   - A Student-t with ν = 4 + 6 / kurtosis ≈ 6–8 has these tails.
   - At high charge the noise is only ~0.5 % of the jitter variance, so its shape cannot be measured. The residual
     histograms (above) suggest heavy tails there too.
4. **Is the noise steady?** Hat per 1000-shot block, in pC:
   - **At low charge it is constant:** seg 3 stays at 10.3–10.9 pC and seg 0 at 9.6–13.4 pC. So the excess kurtosis
     is the noise's own shape, not a drifting width.
   - At high charge the blocks scatter (seg 1: 20–81 pC; seg 5: 11–118 pC). That is within the noise of a
     1000-shot estimate there, so a drift can be neither confirmed nor excluded.

## Minimising the largest error (`jitter_maxloss.py`, `results/jitter_maxloss/`)

```bash
python3 -I scripts/jitter_maxloss.py data/tables results/jitter_sequence 2026-10-07_21-43-26 2026-10-08_00-08-19 \
    2026-10-08_15-42-27 2026-10-08_21-11-47 2026-10-09_07-46-07
```

Linear models of the jitter (current-shot inputs, one per regime) are trained on the mean of |error|^p for p = 1, 2,
4 and 8, and on the maximum |error| (p = ∞, Chebyshev regression, a linear program). Training excludes 5-MAD target
outliers (BCM-lost shots).

| \|error\| pC, test regions | MAE seg 5 | MAE seg 1 | max seg 0 | max seg 3 | max seg 4 | max seg 5 | max seg 1 | max seg 6 | max seg 7 |
|---|---|---|---|---|---|---|---|---|---|
| linear, p = 1 | 51.5 | 62.0 | 50.3 | 77.6 | 43.3 | 671 | 1066 | 260 | 5357 |
| linear, p = 2 | 69.5 | 81.9 | 50.9 | 76.3 | 43.8 | 619 | 495 | 345 | 2423 |
| linear, p = 8 | 89.8 | 105.1 | 65.9 | 72.3 | 43.1 | 615 | 551 | 353 | 4493 |
| linear, p = ∞ (minimax) | 133.3 | 111.2 | 93.8 | 70.7 | 46.5 | 689 | 546 | 501 | 4592 |
| best model (L1-trained GRU) | 40.8 | 45.7 | 50.1 | 74.3 | 40.0 | 777 | 477 | 208 | 2955 |

- **Minimax training does not lower the test maximum.** On the training set it does: at high charge the maximum
  falls from 30–40 % (L2) to 11 %. But the worst training shots are not like the worst test shots, so on the test set
  the maximum is about the same or higher, and the MAE is 1.5–4× worse.
- **What sets the largest errors.** Looking at the five worst test shots per segment:
  - **Segs 1 and 5: shots without a camera frame** (dark or frozen, ~18 % of test shots). There the model has only the
    laser, which misses the jitter that the laser does not drive:

    | best model, \|error\| pC | seg 1, camera present | seg 1, camera missing | seg 5, camera present | seg 5, camera missing |
    |---|---|---|---|---|
    | MAE | 32 | 106.5 | 32 | 82.5 |
    | 99th percentile | 120 | 430 | 106 | 371 |
    | max | 161 | 477 | 160 | 777 |

    With the camera, the worst error at ~5 nC is 160 pC (3 %).
  - **Seg 7:** the last 8 shots of the run, where the beam jumps by ±60 %. The model follows the direction but not the
    full size. The GRU clips its inputs at 5 training standard deviations.
  - **Seg 6:** the camera over-swings (−25 % against the BCM's −9 %).
- **A dedicated no-camera model does not help:** on camera-missing shots, ridges on the non-camera inputs (trained on
  all shots, or on camera-missing shots only, ± past BCM) do no better than the GRU (seg 5: 83–86 against 82.5 pC).
  That is the information limit of the laser readings.
- **So the maximum can only come down by recovering camera frames.** Otherwise, report an error bar that is ~3× wider
  when the camera frame is missing, instead of one worst case.

## Decomposing the target by timescale (`jitter_spectrum.py`, `results/jitter_spectrum/`)

```bash
python3 -I scripts/jitter_spectrum.py data/tables results/jitter_sequence results/jitter_spectrum/jitter_spectrum.png \
    2026-10-07_21-43-26 2026-10-08_00-08-19 2026-10-08_15-42-27 2026-10-08_21-11-47 2026-10-09_07-46-07
```

These are Welch power spectra on each test region, of the jitter target, the current-shot ridge's residual and the
best model's residual. The white BCM noise floor uses σ_n(Q) from `bcm_noise_model.py`. The script also prints the
variance per band of period: 2–4, 4–10, 10–50 and > 50 shots.

- **No lines.** There is no periodic part (mains beat, feedback cycle) to split off and model on its own.
- **The target is nearly white,** with a little extra power at periods over 50 shots at high charge: drift that the
  trailing level lags behind.
- **The best residual is flat too**: 2–4× the floor in every band at high charge (segs 1, 5, 7), and at the floor on
  segs 0, 4 and 6. What is left is the same-shot mapping, not a timescale that a separate model could take.
- **Seg 3 is the exception.** Its residual at periods over 50 shots is 12.9 %² against a floor of 2.7 %², where the
  floor holds everywhere else. That is a slow drift the 50-shot trailing level misses at low charge. A level that
  also follows the instruments' slow parts could close it. That band is about a third of seg 3's gap to the floor;
  the rest is spread over the other bands.


## Pulse-aligned datasets (`extract_features_aligned.py`, `compare_datasets.py`, `results/datasets_aligned_check/`)

A copy of `extract_features.py` (output `results/datasets_aligned`) that puts every reading on the 1.2 s pulse grid
with `pulses.py` instead of shifting whole rows by one. Laser meters take `meter_shift(delay)`, RF traces `rf_shift`,
the camera its own stamp. A reading whose pulse was not recorded becomes NaN, so it no longer lands on a neighbour's
shot. Late in run 2026-10-08_21-11-47 about 40 % of rows are 2 pulses apart, so fewer rows stay valid.
`FA_DATA_ROOT=<h5 dir> FA_JOBS=6 python -I extract_features_aligned.py` takes about 65 min.

`compare_datasets.py <dir> <dir>` compares the two datasets segment by segment: the correlation of each input's
shot-to-shot part with the charge, and a ridge test R². On rows both datasets keep, R² goes from 0.65 to 0.89 on
seg 5 and from 0.19 to 0.25 on seg 3. It drops on segs 1 and 4 (0.44 → 0.38, 0.54 → 0.41), and seg 7's test region
has only about 20 rows.

## Filtering the slow level (`jitter_kalman.py`, `results/jitter_kalman/`)

Compares levels other than the 50-shot trailing mean: a trailing median, a robust EMA (half-life 25, 4σ gate), and a
local-level Kalman filter with the measured BCM noise as R (4σ gate, reset after 5 rejections). Each level is scored
alone and with a ridge refitted on top. Tuned on the level alone, the Kalman filter is the best level (seg 5:
140.5 vs 145.5 pC), but it follows the jitter the ridge would predict. With the ridge on top, nothing beats the
trailing mean (seg 5: 69.5 pC trailing, 71.2 median, 72.5 EMA, 72.1 Kalman). Keep the trailing mean.

## Results slides (`make_results_deck.py`, `results_slides.py`, `results/slides/figs/`)

`python -I make_results_deck.py Forecasting.pptx Forecasting_results.pptx results_slides.py` writes a copy of the
kickoff deck with results slides after each tier slide. The new slides reuse the deck's own shapes and renumber the
pages. `results_slides.py` holds the text, tables and figures, with the source of each. Notebook figures were decoded
to `/tmp/slides_figs`, and copies are kept in `results/slides/figs`.
