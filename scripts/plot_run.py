# Whole-run overview: BCM charge, beam-loading-step charge and lab laser energy per shot, test region shaded,
# plus a per-10-chunk summary table.
# Usage: python -I plot_run.py <tables dir> <run> <out.png>
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beam_loading_step import step
from nbfuncs import TARGET, cut_shot, dropouts
from run_data import clean_run, kept_chunks
from pulses import align, meter_shift, pulse_ids, read_delay, rf_shift

p = os.path.join(sys.argv[1], sys.argv[2])
df = pd.read_parquet(p + '.parquet')
d, L = np.load(p + '_rf.npz'), np.load(p + '_laser.npz')
t, ch = d['t'], d['chunk']
pid = pulse_ids(t, ch)
P = align(d['PGLA'], pid, rf_shift(d['PGLA_off']))
s = pd.Series(step(P) / 0.89, df.index).rolling(41, center=True, min_periods=10).median()
lab = pd.Series(align(L['lab'], pid, meter_shift(read_delay(df['pc_time'], t))) * 1e6, df.index)
field = P[:, 400:600].mean(1) - P[:, :40].mean(1)
Q, cut, drop = df[TARGET], cut_shot(clean_run(df, sys.argv[2])), dropouts(df)
kept = kept_chunks(df, sys.argv[2])

fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
ax[0].plot(df.index, Q, ',', color='k', label='BCM_GUN')
ax[0].plot(df.index[drop], Q[drop], ',', color='r', label='dropout')
ax[0].plot(df.index, s, color='C1', lw=1, label='beam-loading step / 0.89 (41-shot median)')
ax[0].set_ylabel('charge (nC)'); ax[0].legend(markerscale=20, loc='upper left')
ax[1].plot(df.index, lab, ',', color='C0'); ax[1].set_ylabel('lab laser (uJ)')
ax[2].plot(df.index, field, ',', color='C2'); ax[2].set_ylabel('gun field (PGLA a.u.)'); ax[2].set_xlabel('shot')
for a in ax:
    a.axvspan(cut, df.index[kept][-1], color='0.85', zorder=0)
    a.fill_between(df.index, 0, 1, where=~kept, transform=a.get_xaxis_transform(), color='r', alpha=0.12, lw=0)
ax[0].set_title(f'{sys.argv[2]}: grey = test region (last 10 % of the clean span), red = discarded (CLEAN_CUTS.toml)')
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110)

g = pd.DataFrame({'Q': Q, 'Qstep': s, 'lab': lab, 'field': field, 'drop': drop}).groupby(ch // 10 * 10)
print(g.agg(Q=('Q', 'mean'), Q_std=('Q', 'std'), Qstep=('Qstep', 'mean'), lab=('lab', 'mean'), lab_std=('lab', 'std'),
            field=('field', 'mean'), drops=('drop', 'sum')).round(2).to_string())
print('->', sys.argv[3])
