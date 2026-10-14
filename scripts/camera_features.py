# Put the team's camera features (extract_features.py -> results/datasets/segment_XX.npz: BTV_* and SVD_* columns, per
# raw row, not shifted) on our pulse grid. The camera's own cycle stamp (BTV0125/cycleStampGlob) names its pulse; the
# extra pulse shift is picked where the same-shot r of BTV_intensity with the charge jitter peaks, and the r of each
# shift is printed. Repeated images (btv_frozen) are dropped. Output: <tables>/<run>_camera.npz (shot + one array per
# feature, NaN where the pulse has no image of its own).
# Usage: python -I camera_features.py <h5 root> <datasets dir> <tables dir> <run> [<run> ...]
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py
import numpy as np
import pandas as pd
from nbfuncs import TARGET
from pulses import align, pulse_ids, stamp_shift
from run_data import kept_chunks, load_run

SHIFTS = (-1, 0, 1)
DARK = 0.3            # a frame below this fraction of the recent intensity is a camera glitch, not beam

h5root, dsdir, tables, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
man = json.load(open(os.path.join(dsdir, 'manifest.json')))
cam = [i for i, f in enumerate(man['features']) if f.startswith(('BTV_', 'SVD_'))]
names = [man['features'][i] for i in cam]


def jit(v, seg):  # analysis only: centred 51-shot median removed within each clean segment
    return v - v.groupby(seg).transform(lambda s: s.rolling(51, center=True, min_periods=10).median())


for run in runs:
    df, x, bcm_bad, good = load_run(tables, run)
    full = pd.read_parquet(os.path.join(tables, run + '.parquet'))
    rows = []
    for s in man['segments']:
        if s['run'] != 'acquisition_' + run:
            continue
        d = np.load(os.path.join(dsdir, s['file']))
        rows.append(pd.DataFrame(np.c_[d['X'][:, cam], d['y'], d['btv_frozen']], columns=names + ['y', 'frozen'])
                    .assign(chunk=d['chunk'], sic=d['shot_in_chunk']))
    C = pd.concat(rows, ignore_index=True)
    # shot number, trigger and camera stamps of each row, from the HDF5 chunks
    S = []
    for c in np.unique(C.chunk):
        with h5py.File(os.path.join(h5root, f'acquisition_{run}', f'chunk_{c:05d}.h5'), 'r') as f:
            S.append(pd.DataFrame({'chunk': c, 'sic': np.arange(int(f.attrs['n_shots'])), 'shot': f['shot'][()],
                                   'trig': f['trigger_stamp'][()], 'cam': f['BTV0125/cycleStampGlob'][()] / 1e9}))
    C = C.merge(pd.concat(S), on=['chunk', 'sic'], validate='1:1').set_index('shot').reindex(full.index)
    ok = C.y.notna() & full[TARGET].notna()
    assert np.allclose(C.y[ok], full[TARGET][ok]), 'segment rows do not match our shots'
    off = stamp_shift(C.cam - C.trig)
    print(f'{run}: camera stamp - trigger, in pulses:', pd.Series(off).value_counts().sort_index().to_dict())
    rf = np.load(os.path.join(tables, run + '_rf.npz'))
    pid = pulse_ids(rf['t'], rf['chunk'])
    V = C[names].where(C.frozen != 1)
    kept = kept_chunks(full, run)
    seg = (kept != kept.shift()).cumsum().loc[df.index]                  # contiguous clean chunks
    y = jit(df[TARGET].where(good), seg)
    rs = {}
    for k in SHIFTS:
        a = pd.Series(align(V['BTV_intensity'].values, pid, off + k), full.index).loc[df.index]
        j = jit(a, seg)
        m = j.notna() & y.notna()
        rs[k] = np.corrcoef(j[m], y[m])[0, 1]
    k = max(rs, key=rs.get)
    print(f'  BTV_intensity r with the charge jitter, per extra shift:', {s: round(v, 3) for s, v in rs.items()}, '-> use', k)
    out = {n: align(V[n].values, pid, off + k)[np.searchsorted(full.index, df.index)] for n in names}
    # dark frames: a few % of images read 1-20 % of the usual intensity while the BCM and the laser are normal. Causal:
    # against the median of the previous 50 frames, and only where the lab laser has not dropped too
    trail = lambda v: v / v.shift(1).rolling(50, min_periods=5).median()
    dark = (trail(pd.Series(out['BTV_intensity'], df.index)) < DARK) & (trail(x.lab) > 0.7)
    print(f'  dark frames masked: {dark.mean():.3%}')
    out = {n: np.where(dark, np.nan, v) for n, v in out.items()}
    for n in names:
        j = jit(pd.Series(out[n], df.index), seg)
        m = j.notna() & y.notna()
        print(f'    {n:16s} r {np.corrcoef(j[m], y[m])[0, 1]:+.3f}  ({m.sum()} shots)')
    np.savez(os.path.join(tables, run + '_camera.npz'), shot=df.index.values, **out)
    print('  ->', os.path.join(tables, run + '_camera.npz'), flush=True)
