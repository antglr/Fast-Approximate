# Beam-loading step on the gun pickup crest (found in Fast-Approximate notebooks/02_extract_waveforms):
# the beam passes at PGLA sample ~632 and the field drops by ~0.9 a.u. per nC until the RF falls (~666).
# step = mean(620-631) - mean(637-655). Absent on beam-off shots, tracks drift and regime changes.
# Per shot it is noisy (~0.4 nC-equivalent), so a trailing mean over the previous rows (no BCM needed) gives the
# level, and the lab meter's deviation from its own trailing mean gives the shot-to-shot part.
# Scored as a virtual diagnostic: absolute charge, leave-one-chunk-out.
# Usage: python -I beam_loading_step.py <rf.npz> <laser.npz> <table.parquet>
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pulses import align, meter_shift, pulse_ids, read_delay, rf_shift

STABLE = [17, 58, 88, 109, 132, 160, 178]
BEFORE, AFTER = slice(620, 632), slice(637, 656)     # PGLA samples either side of the beam (~632)


def step(pgla):
    return pgla[:, BEFORE].mean(1) - pgla[:, AFTER].mean(1)


def trailing(s, group, n):
    # mean over the last n rows of the same chunk, including this one
    return s.groupby(group).transform(lambda x: x.rolling(n, min_periods=max(2, n // 3)).mean())


if __name__ == '__main__':
    x = np.full((2, 700), 10.)
    x[1, 632:] = 9.                                      # a 1 a.u. drop at the beam
    assert np.allclose(step(x), [0, 1])

    d, L = np.load(sys.argv[1]), np.load(sys.argv[2])
    Q, ch, t = d['Q'], d['chunk'], d['t']
    assert np.array_equal(d['shot'], L['shot'])
    pid = pulse_ids(t, ch)
    delay = read_delay(pd.read_parquet(sys.argv[3]).reindex(L['shot'])['pc_time'], t)
    P = align(d['PGLA'], pid, rf_shift(d['PGLA_off']))
    D = pd.DataFrame({'Q': Q, 'c': ch, 'step': step(P), 'lab': align(L['lab'], pid, meter_shift(delay)) * 1e6})

    i = np.where((ch == 244) & np.isfinite(D.step))[0]
    print('chunk 244 shutter blocks: mean step by charge level')
    for lv in (0, 1.67, 3):
        k = i[np.abs(Q[i] - lv) < 0.3]
        print(f'  Q ~ {lv:4.2f} nC: step {D.step[k].mean():+.2f} ± {D.step[k].std() / np.sqrt(len(k)):.2f}  (n={len(k)})')
    cm = D.groupby('c')[['Q', 'step']].mean()
    print('chunk means:  ' + '  '.join(f'{c}: {r.Q:.2f} nC / {r.step:+.2f}' for c, r in cm.iterrows()))

    for n in (10, 20, 40):
        D[f'step{n}'] = trailing(D.step, D.c, n)
        D[f'lab_dev{n}'] = D.lab - trailing(D.lab, D.c, n)
    sets = [['lab'], ['step'], ['step20'], ['step40'], ['step20', 'lab_dev20'], ['step40', 'lab_dev20'],
            ['step40', 'lab_dev10'], ['step40', 'lab_dev40']]
    for chunks, tag in ((STABLE, '7 stable'), (sorted(np.unique(ch)), 'all 9')):
        G = D[D.c.isin(chunks)].dropna()
        print(f'\nvirtual diagnostic, leave-one-chunk-out over {tag} chunks ({len(G)} shots with every feature):')
        for cols in sets:
            p = np.empty(len(G))
            for c in chunks:
                m = (G.c == c).to_numpy()
                if m.any():
                    p[m] = LinearRegression().fit(G[cols][~m], G.Q[~m]).predict(G[cols][m])
            e = G.Q.to_numpy() - p
            bias = pd.Series(e).groupby(G.c.to_numpy()).mean()
            print(f'  {" + ".join(cols):22s} MAE {np.abs(e).mean():.3f} nC   chunk bias rms {np.sqrt((bias ** 2).mean()):.3f} nC')
