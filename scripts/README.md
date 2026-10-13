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
| `gru_forecast.py` | GRU sequence model on the last 64 shots, trained across runs (below); needs PyTorch |

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
  out of training (in this run, 2416 shots between shots 5000 and 8650).

## Results: run 2026-10-09_07-46-07, next shot (h = 1), test region = last 1096 shots

| Forecast | MAE (nC) | vs baseline |
|---|---|---|
| mean of last 9 good shots (baseline) | 0.505 | |
| mean of last 3 good shots (window picked on train) | **0.396** | −22 % |
| ridge: past charge + laser meters + beam-loading step | 0.405 | −20 % |

- **The test region has two level changes**, made by the operators: at ~shot 10700 the laser goes up to 40 uJ
  (19 nC), then down to 5 uJ (1 nC). Most of the gain is faster recovery after these changes.
  - The ridge reaches the new level one shot after the change, because the last laser reading already shows it.
  - Means need about W shots.
- **On the 1005 steady shots:** 9-shot mean 0.383, 3-shot mean 0.359, ridge 0.358.
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
| ridge | 0.199 / 0.153 | 0.420 / 0.330 | 3.9 % | −0.20 | +1.0 |

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
| ridge (`forecast_run.py`) | 0.405 | −20 % | | | 0.153 / 0.330 |
| **hybrid, charge only (z = 5, wmax = 9)** | **0.352** | **−30 %** | 0.287 | 0.80 | 0.134 / 0.300 |
| hybrid, laser must confirm (z = 4, wmax = 9) | 0.352 | −30 % | 0.287 | 0.80 | 0.136 / 0.300 |
| hybrid, two shots in a row (z = 3, wmax = 9) | 0.391 | −23 % | 0.285 | 1.12 | 0.134 / 0.299 |

- **The hybrid matches the long mean on stable beam and the 3-shot mean after changes**, and beats both overall.
- **Robust to tuning:** test MAE stays between 0.345 and 0.355 nC for every z ≥ 4 and every wmax from 9 to 100.
  - Training picks wmax = 9 because the training part contains a slow laser scan, and a slow ramp never
    triggers a change.
  - On stable beam alone, wmax = 20–50 is ~2 % better.
- **The laser adds nothing here.** Requiring the laser to confirm leaves the test score unchanged and makes the
  training score worse (0.267 vs 0.233 nC). In training, the charge also changes without the laser: the BCM
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
| ridge | 0.405 (−20 %) | **0.139 (−3.6 %)** |
| hybrid, tuned on its own run | **0.352 (−30 %)** (z = 5, wmax = 9) | 0.141 (−2 %) (z = 6, wmax = 20) |
| hybrid, z = 6, wmax = 20 on both | 0.348 (−31 %) | 0.141 (−2 %) |

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
  - 176 dropouts and 72 BCM-wrong shots.
- **Test region (last 2550 shots):** mostly 0.2–0.35 nC, with short steps to 1.2, 1.7, 3 and 5.4 nC. 328 of the
  2459 scored shots are within 20 shots after a change.

| Next-shot forecast, run 3 test region | MAE (nC) |
|---|---|
| 9-shot mean (baseline) | 0.0702 |
| last shot | 0.0469 (−33 %) |
| 3-shot mean | 0.0507 (−28 %) |
| ridge (`forecast_run.py`) | 0.1210 (+72 %) |
| hybrid, z = 6, wmax = 20 (also what run 3's own tuning picks) | 0.0482 (−31 %) |

- **The hybrid setting holds on a third run.** Tuned on run 3's training part alone, it picks z = 6, wmax = 20
  again.
- **The last shot alone is slightly better here (0.0469).** At 0.3 nC the jitter has more memory: lag-1
  autocorrelation is +0.26, against ~0.14 at 5–10 nC. The level changes also favour short memory.
- **The `forecast_run.py` ridge fails** because it predicts in nC with one fit across 5 nC and 0.3 nC. The
  combined forecaster's relative correction does not have this problem.

## Combined forecaster: hybrid + stable-beam correction (`combined_forecast.py`, `results/combined/`)

```bash
python3 -I scripts/combined_forecast.py data/tables results/combined 2026-10-09_07-46-07 2026-10-08_21-11-47 \
    2026-10-08_00-08-19
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

| Next-shot test MAE (nC) | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) |
|---|---|---|---|
| 9-shot mean (baseline) | 0.505 | 0.1442 | 0.0702 |
| hybrid (z = 6, wmax = 20) | 0.348 (−31.2 %) | 0.1414 (−1.9 %) | 0.0482 (−31.3 %) |
| ridge alone (`forecast_run.py`) | 0.405 (−19.9 %) | 0.1390 (−3.6 %) | 0.1210 (+72 %) |
| combined, correction trained on its own run | 0.3413 (−32.4 %) | 0.1392 (−3.4 %) | 0.0480 (−31.6 %) |
| **combined, correction trained on all 3 runs** | **0.3411 (−32.5 %)** | **0.1396 (−3.2 %)** | **0.0480 (−31.6 %)** |
| combined, correction trained on the other 2 runs only | 0.3411 (−32.5 %) | 0.1396 (−3.2 %) | 0.0484 (−31.1 %) |

- **Frozen test on run 3.** Everything was chosen on runs 1 and 2: hybrid settings, features, and correction trained
  on those runs only. On run 3 that scores 0.0484 nC, −31 % against the baseline.
  - All of the gain comes from the hybrid (0.0482).
  - At 0.3 nC the correction neither helps nor hurts.
- **The combined forecaster is the best, or within 0.0004 nC of the best, on all three runs.**
- **On stable test shots** the correction cuts the error by ~2–3 % on runs 1 and 2 (run 1: 0.275 vs 0.282 nC;
  run 2: 0.140 vs 0.141). After changes it leaves the hybrid unchanged.
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
python3 -I scripts/gru_forecast.py data/tables results/gru 2026-10-09_07-46-07 2026-10-08_21-11-47 2026-10-08_00-08-19
```

**How it works:**
- **Input:** the last 64 shots, each with six channels:
  - charge (0 where the hybrid skipped the shot, plus a skip flag);
  - hybrid level;
  - lab and clex laser meters;
  - beam-loading-step charge.
- **Scaling:** charges are divided by the hybrid level at the newest shot, and the laser meters by their window mean.
  Each channel is then divided by its training std.
- **Output:** the next shot relative to the hybrid level, so 0 means "same as the hybrid".
- **Model:** one GRU layer with 32 hidden units, then a linear layer.
- **Training:**
  - L1 loss, because the metric is MAE.
  - Trained on every good next shot of the first 90 % of the runs, early-stopped on the last 10 % of those.
  - 3 seeds per variant, with their forecasts averaged.
- **Run time:** about 1 min per fit on 4 CPU threads (12 threads is 9× slower), about 15 min in total.
- **To try more inputs, add channels in `windows()`.**

| Next-shot test MAE (nC) | run 1 (09_07) | run 2 (08_21) | run 3 (08_00) |
|---|---|---|---|
| 9-shot mean (baseline) | 0.505 | 0.1442 | 0.0702 |
| hybrid (z = 6, wmax = 20) | 0.348 (−31 %) | 0.1414 (−2 %) | 0.0482 (−31 %) |
| **combined (hybrid + ridge)** | **0.341 (−33 %)** | **0.1396 (−3 %)** | **0.0480 (−32 %)** |
| GRU, pooled runs, charge only | 0.350 (−31 %) | 0.1416 (−2 %) | 0.0506 (−28 %) |
| GRU, pooled runs, all inputs | 0.366 (−28 %) | 0.1411 (−2 %) | 0.0543 (−23 %) |
| GRU, other runs only, all inputs | 0.410 (−19 %) | 0.1414 (−2 %) | 0.0575 (−18 %) |
| GRU, all inputs, not standardised (first try) | 0.351 (−31 %) | 0.1413 (−2 %) | 0.0491 (−30 %) |

Seed-to-seed spread of the test MAE: 0.0002–0.011 nC, and 0.09 nC for "other runs only" on run 1.

- **The GRU satisfies the Hard-tier rule** (a sequence model that beats the 9-shot mean on a chronological split).
  It does not beat the hybrid it starts from.
- **On validation it only matches the hybrid**, in every fit (5.27–5.28 % L1 against 5.27 % for the hybrid
  alone). There is little sequence structure left to learn:
  - the stable-beam jitter is nearly white (lag-1 autocorrelation ~0.14);
  - level changes have no warning in the laser data;
  - the slow laser-driven drift is the ~2–3 % that the ridge correction already takes.
- **Where it loses is just after step-ups, where it overshoots.** After 0.25 → 1.67 nC it predicts 2.2 nC and decays
  back over ~20 shots (`gru_<run>.png`, zooms).
  - The laser channels drive it: a 4 → 11 uJ jump, divided by a training std of ~0.03, saturates the GRU.
  - The charge-only variant, and the first try without standardisation, overshoot much less.
- **"Other runs only" is weak on run 1.** Runs 2 and 3 contain no 10 → 19 nC jumps to learn from.
- **Things to try:**
  - clip the standardised inputs (at ±5 std);
  - train on stable shots only, as the ridge correction does;
  - a longer window (the ridge's strongest feature is a 200-shot mean);
  - use the GRU as a correction on top of the combined forecaster.

