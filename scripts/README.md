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
