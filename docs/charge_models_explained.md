# Predicting the CLEAR gun charge: what we are doing, in plain words

## 1. The problem

The machine fires an electron bunch every 1.2 s. A current monitor, the **BCM**, measures each bunch's charge in
nanocoulombs (nC). We want to know that charge **from the other instruments**:
- the laser energy meters (lab, clex), and the laser photodiode (scope);
- the camera that looks at the beam on a screen (BTV);
- the RF pickup in the gun (its dip when the bunch passes, the "step").

If we can do that, we can check the BCM, stand in for it when it fails, and later forecast the next shot.

For now we do **not** forecast. We estimate the charge of a shot from that same shot's readings.

## 2. The charge has two parts

Look at any run and you see two things on top of each other:

```
charge  =  slow level        x  (1 + fast jitter)
           drifts over          changes from one shot
           minutes to hours     to the next, a few %
```

- **The slow level** wanders: the laser drifts, the photocathode ages, and operators change settings.
- **The fast jitter** is the shot-to-shot scatter around the level: 3 % at 5–10 nC, 8–13 % at 0.1–0.25 nC. Most of
  it comes from the laser energy changing from pulse to pulse.

They need different tools, and that is the whole story of this document.

## 3. The three regimes in the clean data

| Regime | Segments | What it looks like |
|---|---|---|
| low charge | 0, 2, 3, 4 (runs 07_21, 08_15) | 0.1–0.25 nC, large relative jitter, weak signals |
| high charge, repetitive | 5 (run 08_21) | ~5.5 nC, slow sawtooth swings that follow the laser |
| high charge, non-repetitive | 1, 6, 7 (runs 08_00, 09_07) | 5–10 nC, drifts that do not simply follow the laser |

A **segment** is a stretch of clean data with no gap. For every test, the model learns on the first 90 % of each
segment and is scored on the last 10 %. It is never scored on data it has seen.

## 4. The first model: a straight line through logs ("log-ridge")

**What it does:** log(charge) = a + b1 × log(laser) + b2 × log(camera brightness) + ...
- **Logs**, because the physics multiplies: twice the laser energy gives about twice the charge, and twice the
  charge gives a screen about twice as bright. In logs those become sums, and a straight line fits.
- **Ridge** is ordinary least squares with a small penalty on large coefficients. It keeps the fit stable when
  several inputs say almost the same thing, like the six camera "SVD" numbers.

**The key choice is the intercept "a", which sets the overall level.** We tried four ways:

| Way | What it means | Error (% of the charge) |
|---|---|---|
| one level for the whole regime | one "a" for all segments of a regime | 1.5–23 % |
| **each segment calibrated** | each segment gets its own "a" from its first 90 % | **1.5–8 %** |
| brand-new segment | everything learned from the other segments | 12–60 % |
| brand-new segment, calibrated | slopes from the other segments, only "a" from this one | 1.7–8 % |

**What that tells us:**
- **Getting the level right is most of the battle.** The camera's brightness per nC differs ~10× between runs, so
  a model that has never seen a run gets the level wrong, even though the shape of its prediction is right.
- **The slopes travel well between segments.** Fixing only the level with a few BCM readings repairs almost
  everything.

## 5. Where the remaining error is

We split each model's error the same way as the charge:
- the **slow part** is the error smoothed over ~50 shots;
- the **fast part** is what is left.

| Model | Slow part of the remaining error | Jitter it explains |
|---|---|---|
| segment's average only (no instruments) | 6–89 % | 0 % (by definition) |
| best model, each segment calibrated | **2–16 %** (seg 0: 51 %) | **27–76 %** |
| best model, brand-new segment, not calibrated | **56–100 %** (seg 2: 8 %) | – |

**Jitter it explains** is the fraction of the BCM's own shot-to-shot scatter the model reproduces:
- high-charge segments 5 and 7: 63–76 %;
- high-charge segments 1 and 6: 32–43 %;
- low-charge segments 3 and 4: 27–28 %;
- low-charge segment 0: 70 %.

**In one sentence:** once a segment is calibrated, the slow drift is handled well, and nearly all the remaining
error is shot-to-shot jitter. That is your observation, now measured.

Two things to keep in mind:
- **Some jitter can never be explained.** Part of the BCM's scatter is the BCM's own measurement noise, and no
  instrument can predict that. At low charge this noise may be a large share. Comparing the three charge-sensitive
  instruments against each other (BCM, camera, RF step) can estimate how big it is.
- **The laser meter alone can make the fast part worse** (negative "jitter explained" on several segments). It
  follows the drift but adds its own noise from shot to shot.

## 6. The plan: split the signal, and give each part its own model

1. **Split every signal into slow and fast** with a frequency decomposition.
   - The simplest version is a moving average for the slow part, with fast = signal − slow. Fancier versions are
     exponential smoothing, low-pass filters and wavelets.
   - **It must only look backwards** (a causal filter). A centred average or an FFT over the whole run uses future
     shots, which the model would not have in real use.
   - Use relative terms: fast = log(signal / its slow part). Then 3 % jitter means the same at 0.1 nC and at 10 nC.
2. **Slow model:** predicts the level, from the BCM history, slow features, or a calibration that tracks drift.
3. **Fast model:** predicts the jitter, from the fast parts of the instruments.
   - It is trained with a loss on the jitter only. The big level differences then no longer swamp the small
     shot-to-shot signal.
4. **Combine:** charge = slow prediction × (1 + fast prediction). Score each part on its own, and the total at the
   end.

**Next experiment: sequence models for the fast part.**
- **RNN** (recurrent neural network): reads the recent shots one by one and keeps a running memory.
- **GRU**: an RNN with gates that decide what to remember and what to forget. It is usually easier to train.
- Both take the fast parts of the last few dozen shots' readings, together with the current shot's, and predict
  the current shot's jitter.
- They are compared against the log-ridge on the same split, per regime.
- Earlier forecasting work found that the history length matters little for the jitter. So the question is whether
  a sequence model finds nonlinear or delayed structure that the straight line misses.

**Result** (`scripts/jitter_sequence.py`; table in `scripts/README.md`). The jitter is measured against the average
of the previous 50 BCM readings, and every instrument's jitter against its own previous 50 readings.
- **A small neural network beats the straight line:** it explains 83–94 % of the jitter on high-charge segments 1, 5
  and 6, against 63–79 % for the ridge. It already does this when it sees only the current shot.
- **Memory hardly helps:** 16 or 64 past shots add at most 2 points, and the plain RNN is about as good as the GRU.
  So the gain comes from a curved (nonlinear) map, not from remembering the past.
- **A big jump beyond anything seen in training** (seg 7's last shots, +55 %) is where the networks fail and the
  straight line does better. The networks cap their output.
- **Low charge stays hard:** 28–49 % of the jitter on segments 3 and 4.

**How low can it go?** (`scripts/bcm_noise_floor.py`). Three instruments read each shot: the BCM, the camera and a
laser meter. Their jitters are compared: whatever is shared is real charge, and what only the BCM has is its own
noise. Nothing can predict that noise.
- **At low charge, about half of the BCM jitter is BCM noise** (5–11 % of the charge).
- **At high charge, the BCM noise is only 0.5–1 %.**
- **The GRU is already at that floor on segments 0, 4 and 6.** Some room is left on segments 5, 1 and 3, enough to
  cut the remaining error by a fifth to a third.

**Two fixes tried:**
- **Straight line plus network** (the network learns what the line misses): the big jump on segment 7 is now
  handled (147 pC average error, against 226 for the network alone), and nothing else gets worse.
- **Giving the model the previous BCM readings:** gains at most 3 %. The past BCM jitter tells it almost nothing that
  the instruments do not already say.
- **More laser and RF readings** (the laser pulse's shape on the scope; the gun and klystron power and phase): add
  nothing (`scripts/jitter_input_screen.py`). The laser energy meters already carry what the laser contributes.

## 7. Words used

| Word | Meaning |
|---|---|
| BCM | the charge monitor: the "truth" we predict |
| level | the slowly drifting average charge |
| jitter | the shot-to-shot scatter around the level |
| calibration | fixing the level of a model with a few known BCM readings |
| causal | uses only the past and the present, never the future |
| ridge | least squares with a penalty that keeps coefficients from blowing up |
| MAE | mean absolute error: the average size of the miss |
| segment | an unbroken stretch of clean data |
| RNN, GRU | neural networks that read a sequence of shots and keep a memory; a GRU has gates that choose what to keep |
