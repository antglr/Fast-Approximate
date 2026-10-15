# Plan: LSTM with attention, and autoencoder embeddings, for forecasting (branch adnan_01)

**Question.** Does attention over the input window, or a learned low-dimensional embedding of the shots
(autoencoder), let a recurrent model forecast the charge h shots ahead better than the GRU and the ridge of
`feature_forecast.py`? And what do the attention weights say about which past shots and which inputs matter?

## Setting (same as `feature_forecast.py`, so the numbers compare)

- Data: `results/datasets_aligned/` (readings on their own pulse), robust preparation (`regimes.load(robust=5)`).
- Pooled per regime (low / high repetitive / high non-repetitive), one model per (regime, horizon).
- Horizons h = 1, 5, 10, 20, 50. Inputs: everything known after shot t − h, as a window of the last L shots
  (L = 64 by default): BCM history relative to the 9-shot mean, instrument fast and slow parts, missing flags (29 inputs).
- Target: log Q[t] − log(mean of the last 9 good shots up to t − h). The output layer starts at zero, so an
  untrained model *is* the challenge baseline.
- Split: fit / validation / test per segment as everywhere (test = last 10 %). L1 loss, early stopping on
  validation, 3 seeds averaged.
- References: 9-shot mean, ridge (all inputs), GRU L = 32 (`results/aligned/feature_forecast/`).

## Variants (`scripts/attn_forecast.py`)

| Variant | Encoder | Readout |
|---|---|---|
| `lstm` | LSTM (64) | last hidden state |
| `lstm_additive` | LSTM | Bahdanau additive attention over the time steps, query = last state |
| `lstm_general` | LSTM | Luong "general" (bilinear) attention |
| `lstm_mha` | LSTM + learned positional embedding | 4-head attention, query = last state |
| `lstm_self` | LSTM + one transformer block (self-attention) over its outputs | last position |
| `lstm_vsn_additive` | per-step variable-selection softmax over the inputs (TFT-like), then LSTM | additive attention |
| `ae_frozen` | per-shot MLP autoencoder (29 → 8), pretrained on the fit part, frozen; LSTM on the codes | additive attention |
| `ae_joint` | same autoencoder, trained jointly: L1(forecast) + 0.5 · MSE(reconstruction) | additive attention |
| `seq_ae` | LSTM sequence autoencoder pretrained to reconstruct the window, then fine-tuned with a forecast head on its code | linear head |

## Tests and benchmarks

1. **Sanity:** an untrained model reproduces the 9-shot mean exactly; windows never reach past t − h (checked in code).
2. **Benchmark:** test MAE per segment and horizon, % against the 9-shot mean, against the ridge and the GRU.
3. **Significance:** circular block bootstrap (50-shot blocks) of the best attention / AE variant against the GRU
   and the ridge.
4. **Context length:** the best variant with L = 16, 64, 256 at h = 1 and 10.
5. **Interpretation:** mean attention weight per lag (does the model look at the last few shots or far back?),
   variable-selection weights per regime, autoencoder reconstruction error and code size.
6. **Cost:** training time per fit.

## Expectations (to be confirmed or refuted)

The earlier work found the shot-to-shot jitter unpredictable ahead of time and the window length nearly irrelevant
for the GRU. So attention and embeddings are expected to change little at short horizons; any gain is more likely at
h = 20–50, where the slow parts and the level dynamics matter. The attention maps should show it either way.

## Deliverables

`scripts/attn_forecast.py`, `results/attention/`, `notebooks/adnan/16_attention_autoencoder_forecast.ipynb`.
