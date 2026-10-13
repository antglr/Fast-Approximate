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
| `hybrid_forecast.py` | The hybrid next-shot forecaster (below) |

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
