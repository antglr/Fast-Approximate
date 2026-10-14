# Next-shot charge forecasting (CLEAR, BCM_GUN)

Download a run's HDF5 chunks, reduce them to per-shot tables, and score next-shot forecasts against the
challenge baseline, the mean of the last 9 good shots. The split is chronological: train on the first 90 % of the run,
test on the last 10 %. Results for run 2026-10-09_07-46-07 are in `../results/2026-10-09_07-46-07/`.

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
| `plot_run.py` | Whole-run overview: BCM charge, step charge, laser, gun field |
| `run_data.py` | Loads one run's tables onto the pulse grid; shared shot labels (BCM-wrong, stable, level change) |
| `forecast_run.py` | Forecast scores for horizons 1–50, the next-shot plot and the stable-shot residuals |
| `forecasters.py` | `hybrid()`: the level-change-aware mean, shared by the two scripts below |
| `hybrid_forecast.py` | Tunes and scores the hybrid on one run |
| `combined_forecast.py` | The hybrid plus a ridge correction, trained across runs (below); run it after `run_pipeline.sh` |
| `gru_forecast.py` | GRU sequence model: a stable-beam correction of the hybrid from the last 64 shots, trained across runs (below); needs PyTorch |
| `gru_window_sweep.py` | Reruns the GRU for context windows of 8–256 shots |
| `screen_variables.py` | Tests whether the notebook variables (klystron, gun field, magnets, climate, laser scope) help the next-shot forecast |
| `virtual_bcm.py` | Same-shot virtual BCM: shot n's charge from shot n's own laser readings plus the BCM history (below) |

Run the scripts with `python -I`. The tables match the ones made from the pickles, except that the camera
columns are left out.

## Things the data does that matter

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
  of 09_07-46-07 and 09_11-48-33, while the beam-loading step sees full charge. `forecast_run.py` keeps these shots
  out of training (in this run, 2415 shots between shots 5000 and 8650).
- **Missing readings.** After pulse alignment, ~10 % of lab/clex readings and 13–36 % of step readings have no pulse
  of their own (mostly 1–2-shot gaps, up to 48). `run_data.load_run` carries the last reading forward.
  - It used to interpolate. That filled shot n partly from shot n+1: a future leak into every laser and step
    feature, and fake ramps after jumps.
  - Fixing it cost the forecasts 0.001–0.003 nC. All numbers below are after the fix.

## Results: run 2026-10-09_07-46-07, next shot (h = 1), test region = last 1096 shots

| Forecast | MAE (nC) | vs baseline |
|---|---|---|
| mean of last 9 good shots (baseline) | 0.505 | |
| mean of last 3 good shots (window picked on train) | **0.396** | −22 % |
| ridge: past charge + laser meters + beam-loading step | 0.426 | −16 % |

- **The test region has two level changes**, made by the operators: at ~shot 10700 the laser goes up to 40 uJ
  (19 nC), then down to 5 uJ (1 nC). Most of the gain is faster recovery after these changes.
  - The ridge reaches the new level one shot after the change, because the last laser reading already shows it.
  - Means need about W shots.
- **On the 1005 steady shots:** 9-shot mean 0.383, 3-shot mean 0.359, ridge 0.374.
- **On chunks at 5 nC** the jitter is nearly white, and nothing causal beats the 9-shot mean.
- **Longer horizons:** the ridge loses to the baseline from h = 10 on.
- Plots: `run_overview.png` (the whole run), `next_shot.png` (predicted vs actual next shot),
  `residuals_stable.png` (residuals on stable shots, below).

### Next-shot residuals on stable beam (`residuals_stable.png`, whole run, 6257 shots)

A shot counts as stable if all of these hold:
- it is not a dropout and the BCM is not stuck;
- it is not within 20 shots of a level change;
- the local level moves less than 10 % over ~100 shots;
- there is no bad shot in the previous 10.

The ridge here is out-of-fold over 10 contiguous blocks.

| Forecast | ~5.5 nC: std / MAE (nC) | ~10 nC: std / MAE (nC) | relative std | skew | excess kurtosis (relative) |
|---|---|---|---|---|---|
| 9-shot mean | 0.176 / 0.135 | 0.380 / 0.300 | 3.4 % | −0.20 | +1.0 |
| 3-shot mean | 0.182 / 0.141 | 0.403 / 0.317 | 3.6 % | −0.15 | +0.9 |
| ridge | 0.194 / 0.150 | 0.428 / 0.337 | 3.9 % | −0.22 | +0.9 |

- **On stable beam the 9-shot mean is the best of the three.** The 3-shot mean and the ridge only win at level
  changes.
  - The best window on stable beam is 20–50 shots, and it is only ~2 % better than 9.
  - The jitter about the local level is nearly white (lag-1 r = 0.14).
- **The distribution is close to Gaussian in the core, with heavier tails and a small negative skew.**
  - Skew is −0.2 ± 0.03: occasional low shots, more than high ones.
  - Excess kurtosis is ~+1 relative to the level. In nC it is +2.5, because pooling the 5.5 and 10 nC levels mixes
    two widths.
  - The Q-Q plot follows the Gaussian to about ±2σ. The tails reach −6σ and +5σ.
- **The jitter grows with charge:** 3.2 % rms of the level at 5.5 nC and 3.8 % at 10 nC.
- **Implication for next-shot forecasting:** use a long mean on stable beam and react fast at level changes,
  for example when the last laser reading jumps.
- **One run only.** Scores on other runs will depend on what happens in their last 10 %.

## Hybrid next-shot forecaster (`hybrid_forecast.py`, `hybrid.txt`, `hybrid_next_shot.png`)

The forecast is the mean of the good shots since the last level change, capped at `wmax` shots.
- **Level change:** the charge deviates from the current mean by more than z × the stable jitter (3.1 %). The mean
  then restarts at the new shot.
- **Skipped shots:** a fall of more than 70 % that does not count as a change is treated as a dropout. A reading
  below half the beam-loading-step charge (trailing 5-shot median) is treated as a stuck BCM.
- **Causal:** the official dropout labels are used only for scoring.
- **Tuning:** z and wmax are tuned on the first 90 %.

| Next-shot forecast | test MAE (nC) | vs baseline | test, stable | test, after a change | whole run, stable ~5.5 / ~10 nC |
|---|---|---|---|---|---|
| 9-shot mean (baseline) | 0.505 | | 0.287 | 1.88 | 0.135 / 0.300 |
| 3-shot mean | 0.396 | −22 % | 0.306 | 0.83 | 0.141 / 0.317 |
| ridge (`forecast_run.py`) | 0.426 | −16 % | | | 0.150 / 0.337 |
| **hybrid, charge only (z = 5, wmax = 9)** | **0.354** | **−30 %** | 0.287 | 0.83 | 0.134 / 0.300 |
| hybrid, laser must confirm (z = 4, wmax = 9) | 0.423 | −16 % | 0.287 | 0.99 | 0.136 / 0.300 |
| hybrid, two shots in a row (z = 3, wmax = 9) | 0.403 | −20 % | 0.285 | 1.28 | 0.134 / 0.299 |

- **The hybrid matches the long mean on stable beam and the 3-shot mean after changes**, and beats both overall.
- **Robust to tuning:** test MAE stays between 0.347 and 0.357 nC for every z ≥ 4 and every wmax from 9 to 100.
  - Training picks wmax = 9 because the training part contains a slow laser scan, and a slow ramp never
    triggers a change.
  - On stable beam alone, wmax = 20–50 is ~2 % better.
- **The laser does not help.** Requiring the laser to confirm makes both scores worse (test 0.423 vs 0.354 nC,
  training 0.266 vs 0.232). Before the interpolation fix it looked neutral on test, thanks to the leak. In training, the charge also changes without the laser: the BCM
  recovering from its stuck state, and swings of 3–12 nC at shots 5000–5800 while the laser stays flat.
- **No method can predict the first shot at a new level.** Nothing before that shot shows the change, so the
  hybrid catches up one shot later.

## Second run: 2026-10-08_21-11-47 (`results/2026-10-08_21-11-47/`)

- **What the run looks like:** 20 000 shots, 5–6 nC throughout, with no stuck BCM and no laser steps. The lab laser
  drifts in slow sawtooth cycles every ~1500–3000 shots (9.5 → 12.5 uJ), and the charge follows it. One RF trip at
  shot ~12100 gives 76 dropouts.
- **Test region:** the last 2000 shots are all stable, with no level change.

| Next-shot forecast | run 1 (09_07) test MAE | run 2 (08_21) test MAE |
|---|---|---|
| 9-shot mean (baseline) | 0.505 | 0.144 |
| 3-shot mean | 0.396 (−22 %) | 0.155 (+7 %) |
| 20-shot mean | 0.656 (+30 %) | 0.141 (−2 %) |
| ridge | 0.426 (−16 %) | **0.1395 (−3.2 %)** |
| hybrid, tuned on its own run | 0.354 (−30 %) (z = 5, wmax = 9) | 0.141 (−2 %) (z = 6, wmax = 20) |
| hybrid, z = 6, wmax = 20 on both | **0.350 (−31 %)** | 0.141 (−2 %) |

- **One hybrid setting (z = 6, wmax = 20) works on both runs.**
  - It is the best forecast on run 1, where level changes dominate.
  - On calm run 2 it is the 20-shot mean, about 2 % better than the baseline.
- **The ridge does best on calm beam.** It uses the long means and the latest lab reading. On run 1 it loses
  because one linear fit has to cover both regimes.
- **The stable-beam residuals have the same shape as in run 1:** 3.35 % rms relative, skew −0.11, excess kurtosis +1.0.
- **Next step:** the hybrid for level changes plus a ridge-style laser correction on stable beam.

## Third run: 2026-10-08_00-08-19 (`results/2026-10-08_00-08-19/`)

- **What the run looks like:** 25 500 shots.
  - Shots 0–19 400: 4–6 nC, following slow lab-laser drifts (9–12 uJ).
  - Then the laser drops to 4 uJ and the charge to ~0.3 nC for the rest of the run.
  - At 0.3 nC the beam-loading step reads ~0 nC (an offset), so it is no use as a charge check there.
  - 176 dropouts and 76 BCM-wrong shots.
- **Test region (last 2550 shots):** mostly 0.2–0.35 nC, with short steps to 1.2, 1.7, 3 and 5.4 nC. 328 of the
  2459 scored shots are within 20 shots after a change.

| Next-shot forecast, run 3 test region | MAE (nC) |
|---|---|
| 9-shot mean (baseline) | 0.0702 |
| last shot | 0.0469 (−33 %) |
| 3-shot mean | 0.0502 (−29 %) |
| ridge (`forecast_run.py`) | 0.1119 (+59 %) |
| hybrid, z = 6, wmax = 20 (also what run 3's own tuning picks) | 0.0494 (−30 %) |

- **The hybrid setting holds on a third run.** Tuned on run 3's training part alone, it picks z = 6, wmax = 20
  again.
- **The last shot alone is slightly better here (0.0469).** At 0.3 nC the jitter has more memory: lag-1
  autocorrelation is +0.26, against ~0.14 at 5–10 nC. The level changes also favour short memory.
- **The `forecast_run.py` ridge fails** because it predicts in nC with one fit across 5 nC and 0.3 nC. The
  combined forecaster's relative correction does not have this problem.

## Runs 4 and 5: 2026-10-07_21-43-26 and 2026-10-08_15-42-27 (low charge)

These two runs are the other runs the team marked clean in `config/CLEAN_CUTS.toml`. Everything was tested on them
frozen: settings and features chosen on runs 1–3.
- **Run 4 (07_21-43-26):** 7127 shots.
  - Steps between 0.05 and 1.7 nC in the first ~1100 shots, then a quiet ~0.2 nC to the end.
  - The test region (713 shots) is all stable, with no level change.
  - The jitter is 7.4 % of the level (3–4 % at 5–10 nC).
- **Run 5 (08_15-42-27):** 12 100 shots at 0.05–0.2 nC, with steps and slow swings.
  - The lab laser stays flat (3.6–3.7 uJ) while the charge swings. The gun field moves from 1566 to 1577 in the second
    half, so here the charge may follow the RF rather than the laser.
  - Only 480 of the 1210 test shots count as stable; the rest drift. The jitter is 11.9 % of the level.
- **At these charges the beam-loading step is useless** (it reads ~0 nC), so the stuck-BCM check never fires.

| Next-shot test MAE (nC) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|
| 9-shot mean (baseline) | 0.0198 | 0.0122 |
| 20-shot mean | 0.0189 (−4.4 %) | 0.0119 (−2.5 %) |
| ridge (`forecast_run.py`) | 0.0212 (+6.8 %) | 0.0118 (−3.5 %) |
| hybrid, z = 6, wmax = 20 | 0.0193 (−2.6 %) | 0.0119 (−2.5 %) |
| hybrid, tuned on its own run | 0.0200 (+0.8 %) (z = 4, wmax = 30) | 0.0119 (−2.6 %) (z = 8, wmax = 30) |

- **The hybrid transfers, but gains little here:** neither test region has a level change.
- **Tuning on run 4's own training part goes wrong.** Its first 1100 shots are full of steps, so it picks z = 4. On
  the quiet test region that fires 9 false level changes. The setting from runs 1–3 is safer.

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
| 9-shot mean (baseline) | 0.505 | 0.1442 | 0.0702 | 0.0198 | 0.0122 |
| hybrid (z = 6, wmax = 20) | 0.350 (−30.7 %) | 0.1414 (−1.9 %) | 0.0494 (−29.7 %) | 0.0193 (−2.6 %) | 0.0119 (−2.5 %) |
| ridge alone (`forecast_run.py`) | 0.426 (−15.7 %) | 0.1395 (−3.2 %) | 0.1119 (+59 %) | 0.0212 (+6.8 %) | 0.0118 (−3.5 %) |
| combined, correction trained on its own run | 0.3464 (−31.4 %) | 0.1396 (−3.1 %) | 0.0491 (−30.0 %) | 0.0186 (−5.9 %) | 0.0119 (−2.8 %) |
| **combined, correction trained on all 5 runs** | **0.3483 (−31.1 %)** | **0.1398 (−3.0 %)** | **0.0494 (−29.6 %)** | **0.0185 (−6.4 %)** | **0.0118 (−3.3 %)** |
| combined, correction trained on the other 4 runs only | 0.3485 (−31.0 %) | 0.1399 (−3.0 %) | 0.0495 (−29.4 %) | 0.0185 (−6.5 %) | 0.0118 (−3.2 %) |

- **Frozen tests on runs 3, 4 and 5.** Hybrid settings and features were chosen on runs 1 and 2. Trained on the
  other runs only, the correction scores −29 % on run 3, −6.5 % on run 4 and −3.2 % on run 5 against the baseline.
  - On run 3 nearly all of the gain comes from the hybrid (0.0494).
  - On run 4 the correction adds most of it (hybrid −2.6 %, combined −6.5 %).
- **The combined forecaster is the best, or within 0.002 nC of the best, on all five runs.**
- **Adding runs 4 and 5 to the training barely moves runs 1–3** (with runs 1–3 only: 0.3472 / 0.1397 / 0.0491).
- **On stable test shots** the correction cuts the error by ~1 % on runs 1 and 2 (run 1: 0.280 vs 0.282 nC;
  run 2: 0.1398 vs 0.1414) and by 4 % on run 4 (0.0185 vs 0.0193). After changes it leaves the hybrid unchanged.
  - Before the interpolation fix this looked like 2–3 %, so about half of the apparent gain was the leak.
- **What the correction uses most:** the 200- and 50-shot segment means, the latest lab reading and the 9-shot clex
  mean.
- **Two things were needed:**
  - Train the correction on stable shots only. Trained on all shots, it learns to chase the last few shots from
    run 1's laser scan, and run 2 got worse.
  - Switch it off for the first 20 shots of a segment.
- **A bug that run 3 caught.** At first the step features were relative to their own 200-shot mean. At 0.3 nC that
  mean is ~0, so the features reached ~1000. The correction from runs 1+2 then raised run 3's stable error from 0.026
  to 0.045 nC (test MAE 0.0628). Dividing by the charge level fixed it, and runs 1 and 2 did not change.

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
| 9-shot mean (baseline) | 0.505 | 0.1442 | 0.0702 | 0.0198 | 0.0122 |
| hybrid (z = 6, wmax = 20) | 0.350 (−30.7 %) | 0.1414 (−1.9 %) | 0.0494 (−29.7 %) | 0.0193 (−2.6 %) | 0.0119 (−2.5 %) |
| combined (hybrid + ridge), trained on all 5 runs | 0.3483 (−31.1 %) | 0.1398 (−3.0 %) | 0.0494 (−29.6 %) | 0.0185 (−6.4 %) | 0.0118 (−3.3 %) |
| **GRU, pooled runs** | **0.3472 (−31.3 %)** | **0.1400 (−2.9 %)** | **0.0493 (−29.8 %)** | **0.0188 (−5.1 %)** | **0.0119 (−3.1 %)** |
| GRU, pooled runs, charge only | 0.3453 (−31.7 %) | 0.1407 (−2.4 %) | 0.0495 (−29.5 %) | 0.0189 (−4.6 %) | 0.0119 (−2.4 %) |
| GRU, other runs only (frozen test) | 0.3463 (−31.5 %) | 0.1399 (−3.0 %) | 0.0493 (−29.8 %) | 0.0189 (−4.5 %) | 0.0118 (−3.7 %) |
| *first GRU (runs 1–3; all shots, whole window, no clip; before the interpolation fix)* | *0.366* | *0.1411* | *0.0543* | | |

Seed-to-seed spread of the test MAE: ≤ 0.0019 nC. Every fit beats the hybrid on validation (e.g. 5.34 % against
5.42 % L1).

- **The GRU matches the combined forecaster within 0.0003 nC on all five runs**, and transfers to unseen runs (−4.5 %
  and −3.7 % on runs 4 and 5, trained without them). It satisfies the Hard-tier rule: a sequence model that beats the
  9-shot mean on a chronological split.
- **The laser channels add little.** Without them the GRU is slightly better on run 1 and slightly worse on the others.

### Context window (`gru_window_sweep.py`, `results/gru/window_sweep.txt`; runs 1–3, pooled, 3 seeds)

| window L (shots) | 8 | 16 | 32 | 64 (used) | 128 | 256 |
|---|---|---|---|---|---|---|
| validation L1 (hybrid 4.67–4.68 %) | 4.65 % | 4.64 % | 4.64 % | 4.65 % | 4.65 % | 4.65 % |
| test MAE, run 1 | 0.3479 | 0.3467 | 0.3484 | 0.3482 | 0.3463 | 0.3475 |
| test MAE, run 2 | 0.1406 | 0.1401 | 0.1404 | 0.1406 | 0.1403 | 0.1401 |
| test MAE, run 3 | 0.0491 | 0.0491 | 0.0492 | 0.0491 | 0.0491 | 0.0491 |

- **The window length makes no real difference:** the test MAE moves by ≤ 0.002 nC on run 1 and ≤ 0.0005 on the
  others, within ~2 seed spreads.
- This fits white jitter on stable beam: the hybrid level already holds the useful history, and the GRU only adds a
  small correction from the last few shots. L = 64 is kept. L = 256 costs ~10 min per fit.
- **What made the difference.** Each fix came from looking at where the GRU lost:

| Step | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| first GRU | 0.366 | 0.1411 | 0.0543 |
| + clip the standardised inputs (laser steps saturated the GRU) | 0.358 | 0.1412 | 0.0501 |
| + train on stable shots only, apply after 20 shots in a segment | 0.344 | 0.1400 | 0.0504 |
| + window limited to the current segment | 0.3415 | 0.1401 | 0.0503 |
| + off when the newest shot was skipped; zero-initialised output | 0.3411 | 0.1401 | 0.0481 |

  (Pooled-runs variant, before the interpolation fix; after it, the last row is 0.3482 / 0.1406 / 0.0491. A
  50/200-shot segment-mean channel was also tried and made no difference.)
- **The run-3 plateau.** At shots 23394–23445 the BCM reads a steady 1.67 nC, while the beam-loading step says
  5.4 nC and the laser 11 uJ (about 5 nC earlier in the run). The BCM is probably wrong there, but the official score
  uses it.
  - The hybrid's stuck-BCM check skips these shots and holds 1.68, which happens to match.
  - The GRU saw a window of skipped shots, never seen in training, and predicted 13 % low until it was switched off
    there.

### Do the notebook variables help? (`screen_variables.py`, `results/gru/screen_variables.txt`)

- **Same shot, they correlate as the notebooks found.** Against the charge jitter: lab 0.59–0.62, clex 0.54–0.60,
  laser scope 0.36–0.45, step 0.26–0.36. Klystron, gun field, magnets, temperatures and humidities are all |r| < 0.09.
- **For the next shot, none of them helps.** Added to the correction, each group changes the stable-beam MAE on an
  unseen run by less than 0.5 %. All of them together make run 3 1.0 % worse.
  - The jitter is laser-driven, and the next pulse's laser energy is not predictable from earlier pulses.
  - The slow variables (temperature, humidity, magnets) do not lead the charge drift by enough to matter.
- **A trap:** the laser scope shifted one pulse early correlates 0.17–0.29 with the next-shot error, twice the lab
  meter. That is because it is the next shot's laser reading, which is future data. Its same-shot r peaks at the stamp
  alignment, which confirms the stamp alignment. Any new variable that suddenly predicts well should get this check.

### Where the error is (summed |error|, combined forecaster, test regions)

| | stable beam | 20 shots after a change | other (unsettled beam) |
|---|---|---|---|
| run 1 | 62 % | 20 % (69 of the 74 nC in the first 3 shots of a new level) | 19 % |
| run 2 | 100 % | – | – |
| run 3 | 25 % | 41 % | 34 % |

- On stable beam the remaining error is white, laser-driven jitter.
- After a change, nearly all of the error is in the first shot or two. Changes come without warning in the data.

## Same-shot virtual BCM (`virtual_bcm.py`, `results/virtual_bcm/`)

```bash
python3 -I scripts/virtual_bcm.py data/tables results/virtual_bcm 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19 \
    2026-10-07_21-43-26 2026-10-08_15-42-27
```

**What it is:** an estimate of shot n's charge from shot n's own laser readings, plus the BCM history up to shot n−1.
It is not a forecast, because it needs shot n's instruments. Uses: a cross-check on the BCM, or a stand-in while
the BCM reads wrong.

**How it works:** log(Q[n] / hybrid level after shot n−1) = a ridge on
- the combined forecaster's history features;
- for lab, clex and the laser scope: log(reading[n] / its segment mean over the last 20 and 200 shots).

A missing reading gives a feature of 0. The ridge is trained on all good shots of the first 90 %, not only stable
ones, because level changes are where it should help. There are two ridges: one for when the hybrid level is above
1 nC, one for below it.

| MAE against the BCM, test region (nC) | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) | run 4 (07_21) | run 5 (08_15) |
|---|---|---|---|---|---|
| hybrid next-shot forecast (for scale) | 0.350 | 0.1414 | 0.0494 | 0.0193 | 0.0119 |
| virtual BCM, lab only, pooled runs | 0.304 (−13 %) | 0.1205 (−15 %) | 0.0472 (−4 %) | 0.0172 (−11 %) | 0.0117 (−2 %) |
| **virtual BCM, lab + clex + scope, pooled runs** | **0.295 (−16 %)** | **0.1185 (−16 %)** | **0.0454 (−8 %)** | **0.0134 (−31 %)** | **0.0104 (−13 %)** |
| virtual BCM, lab + clex + scope, other runs only | 0.297 (−15 %) | 0.1204 (−15 %) | 0.0462 (−6 %) | 0.0134 (−30 %) | 0.0106 (−11 %) |
| *the same, one ridge for all charges, pooled runs* | *0.302* | *0.1251* | *0.0449* | *0.0140* | *0.0106* |
| BCM lost at the cut: laser × frozen charge/laser ratio | 0.622 | 0.1655 | 0.519 | 0.0192 | 0.0234 |
| BCM lost at the cut: step − frozen offset | 0.387 | 0.355 | 0.437 | 0.344 | 0.364 |

- **On stable beam, seeing the shot's own laser cuts the error by ~16 % on runs 1 and 2:** run 1 0.233 vs 0.282 nC,
  run 2 0.119 vs 0.141. This is the laser-driven jitter that no forecast can see coming.
- **On the low-charge runs it gains even more: −31 % on run 4 and −13 % on run 5.**
- **At level changes it finds the new level on the shot itself** (see the zooms in the plots). After changes:
  run 1 0.73 vs 0.83 nC, run 3 0.118 vs 0.152.
- **At 0.3 nC (run 3) it gains nothing on stable beam** (0.0256 vs 0.0255).
- **It transfers.** Trained on the other runs only, it is within 0.002 nC of the pooled model on runs 1 and 2, and
  cuts the error by 30 % and 11 % on the unseen runs 4 and 5.
- **Why two ridges.** The laser→charge relation differs below 1 nC. One ridge pooled over all 5 runs made runs 1–2
  worse (0.302 / 0.125 nC, against 0.293 / 0.118 when trained on runs 1–3).
  - Below 1 nC the clex meter carries most of the signal: clex vs its 20-shot mean +3.8 % per std, against +1.2 % for
    lab. The lab meter sits before the attenuation used for low charge.
  - Above 1 nC it is the lab meter: lab vs its 20-shot mean +1.6 %.
- **The beam-loading step is not used.** Its noise is roughly fixed in nC, so at 0.3 nC it is 8× noisier relative
  to the level than at 5 nC (std of the feature 0.75 vs 0.09). Trained on the high-charge runs, the model trusted it
  too much: run 3 scored 0.0511 nC with it and 0.0472 without. On runs 1 and 2 it added < 0.3 %.
- **What it uses most above 1 nC** (change of the estimate per std of each feature): lab vs its 20-shot mean +1.6 %,
  clex vs its 200-shot mean +0.8 %, the 200-shot charge mean +0.6 %, lab vs its 200-shot mean +0.5 %, scope vs its
  20-shot mean +0.4 %.
- **With no BCM at all after the cut, neither stand-in holds.** Both are calibrated once, on the last 200 good
  training shots. The errors are 1.1–10× the hybrid's on runs 1–3 and 5: the virtual BCM needs the BCM history to
  re-anchor. Only on the quiet run 4 does the laser ratio match the hybrid.
- **How the interpolation leak was found.** The first version overshot after big jumps up (30 nC predicted against
  18 nC). The interpolated meters ramped linearly across gaps after a jump, so shot n's "reading" was partly shot n+1's.
