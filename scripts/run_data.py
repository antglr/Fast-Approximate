# One contiguous run's tables, put on the pulse grid, plus the shot labels the forecast scripts share.
import os
import numpy as np
import pandas as pd
from beam_loading_step import step
from nbfuncs import TARGET, W, dropouts
from pulses import align, meter_shift, pulse_ids, read_delay, rf_shift

PER_NC = 0.89          # beam-loading step, a.u. per nC


def load_run(tables, run):
    # -> df (the run table), x (step charge in nC, lab and clex meters in uJ, per shot), bcm_bad, good
    p = os.path.join(tables, run)
    df = pd.read_parquet(p + '.parquet')
    d, L = np.load(p + '_rf.npz'), np.load(p + '_laser.npz')
    assert np.array_equal(d['shot'], df.index) and np.array_equal(L['shot'], df.index)
    t, ch = d['t'], d['chunk']
    pid = pulse_ids(t, ch)
    delay = read_delay(df['pc_time'], t)
    x = pd.DataFrame({'step': step(align(d['PGLA'], pid, rf_shift(d['PGLA_off']))) / PER_NC,
                      'lab': align(L['lab'], pid, meter_shift(delay)) * 1e6,
                      'clex': align(L['clex'], pid, meter_shift(delay)) * 1e6}, index=df.index)
    x = x.replace([np.inf, -np.inf], np.nan)
    for k in ('lab', 'clex', 'step'):
        x[k + '_ok'] = x[k].notna()          # this pulse's own reading exists (~10 % of meter, 13-36 % of step readings don't)
    # fill gaps with the last reading: interpolating fills shot n from later shots, a leak into the causal features
    # (bfill only fills the first shots of a run, before any reading)
    x = x.ffill().bfill()
    # shots where the BCM itself is wrong (stuck low while the beam-loading step sees full charge): never train on them
    qstep = x.step.rolling(41, center=True, min_periods=10).median()
    bcm_bad = (qstep > 1) & (df[TARGET] < 0.5 * qstep)
    return df, x, bcm_bad, ~dropouts(df) & ~bcm_bad


def level_labels(q, good):
    # Analysis labels, made with the future (never use them as features):
    #   med:    the local level, 41-shot centred median of the good charge q
    #   jump:   within 20 shots after the level moved by > 20 %
    #   stable: good, not after a jump, no bad shot in the last W+1 (the official baseline keeps averaging stuck
    #           BCM readings), and the level moves < 10 % over ~100 shots (removes scans)
    med = q.rolling(41, center=True, min_periods=10).median()
    jump = (med / med.shift(20) - 1).abs().rolling(20, min_periods=1).max() > 0.2
    span = med.rolling(101, center=True, min_periods=20)
    recent_bad = (~good).astype(float).rolling(W + 1, min_periods=1).max().astype(bool)
    stable = good & ~jump & ~recent_bad & ((span.max() - span.min()) / med < 0.10) & q.notna()
    return med, jump, stable
