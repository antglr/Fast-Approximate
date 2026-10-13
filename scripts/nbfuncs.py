# Functions copied verbatim from forecasting/kickoff_notebook.ipynb (cells: rules, safe loader, table reduction).
import pickle, numpy as np, pandas as pd
TARGET = 'BCM_GUN'    # the charge at the gun, in nC
TEST_FRAC = 0.10      # the last 10 % of a run's shots are the test region
DROP_FRAC = 0.3       # dropout: charge below 30 % of the local (201-shot) median
W = 9                 # window of the rolling-mean baseline, in shots


def cut_shot(df):
    # first shot of the test region
    return df.index[int(len(df) * (1 - TEST_FRAC))]


def dropouts(df):
    # True for shots whose charge collapsed or is missing
    q = df[TARGET]
    local_median = q.rolling(201, center=True, min_periods=20).median()
    return (q < DROP_FRAC * local_median) | q.isna()


def scored_shots(df):
    # test shots whose charge is a real measurement
    return df.index[(df.index >= cut_shot(df)) & ~dropouts(df)]


def rolling_baseline(df):
    # at each shot: the mean of the last W good shots (the naive forecast)
    return df[TARGET].where(~dropouts(df)).rolling(W, min_periods=1).mean()


def metrics(y, p):
    err = y - p
    return {'MAE': float(np.mean(np.abs(err))),
            'R2': float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}


def score_vd(df, pred):
    # Virtual diagnostic. pred: a Series of predictions indexed by shot.
    # Returns MAE, R2, and the fraction of evaluated shots that got a prediction (coverage).
    shots = scored_shots(df)
    pred = pd.Series(pred).reindex(shots)
    ok = pred.notna()
    res = metrics(df.loc[shots[ok], TARGET].to_numpy(float), pred[ok].to_numpy(float))
    res['coverage'] = float(ok.mean())
    return res


def score_forecast(df, pred):
    # Forecasting. pred: a DataFrame with columns shot, h, pred, where `shot` is the shot
    # being predicted, from information up to shot - h.
    # Returns one row per horizon h, with the baseline at the same origins and
    # dMAE = MAE(baseline) - MAE(model).
    shots = scored_shots(df)
    base = rolling_baseline(df)
    rows = []
    for h, g in pd.DataFrame(pred).groupby('h'):
        h = int(h)
        p = g.set_index('shot')['pred'].reindex(shots)
        origin_ok = (shots - h) >= df.index[0]
        ok = p.notna() & origin_ok
        y = df.loc[shots[ok], TARGET].to_numpy(float)
        model = metrics(y, p[ok].to_numpy(float))
        b = metrics(y, base.reindex(shots[ok] - h).to_numpy(float))
        rows.append({'h': h, 'MAE': model['MAE'], 'R2': model['R2'],
                     'baseline_MAE': b['MAE'], 'dMAE': b['MAE'] - model['MAE'],
                     'coverage': float(ok.sum() / origin_ok.sum())})
    return pd.DataFrame(rows).set_index('h')

SAFE = {   # the only types a chunk may rebuild
    'builtins': {'set', 'frozenset', 'dict', 'list', 'tuple', 'bytearray', 'bytes',
                 'complex', 'int', 'float', 'str', 'bool', 'slice', 'range', 'object'},
    'datetime': {'datetime', 'date', 'time', 'timedelta', 'timezone'},
    'collections': {'OrderedDict', 'defaultdict'},
    'copyreg': {'_reconstructor'},
    '_codecs': {'encode'},
    'numpy': {'ndarray', 'dtype', 'bool_'},
    'numpy.core.multiarray': {'_reconstruct', 'scalar'},
    'numpy._core.multiarray': {'_reconstruct', 'scalar'},
    'numpy.core.numeric': {'_frombuffer'},
    'numpy._core.numeric': {'_frombuffer'},
}


class Opaque:
    # placeholder for any object outside SAFE; keeps its arguments and state for inspection
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state

    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__}>'


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in SAFE.get(module, ()):
            if module.startswith('numpy.core') and hasattr(np, '_core'):
                module = 'numpy._core' + module[len('numpy.core'):]   # renamed in numpy 2
            return super().find_class(module, name)
        return type(name, (Opaque,), {'__module__': module})


def load_chunk(path):
    # a raw chunk file -> list of shot dictionaries
    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()


def data_array(v):
    # The measurement inside a value: a number, a waveform (1-D array) or an image (2-D).
    # Whole device properties (dicts) keep it under 'image2D' (camera), 'signal' (scope)
    # or 'samples' (RF loop, klystron); their other keys are metadata.
    if isinstance(v, dict):
        v = next((v[k] for k in ('image2D', 'signal', 'samples') if k in v), None)
    if v is None or isinstance(v, (int, float)):
        return v
    a = np.squeeze(np.asarray(v))
    return a if a.dtype.kind in 'iufb' else None

N_WINDOWS = 48    # each waveform is also kept as 48 window averages (used in section 3d)
NOISE_CUT = 3     # camera pixels below 3 x the noise level count as background
SKIP = {'shot', 'pc_time', 'trigger_stamp', 'stale_params'}
STATUS = {        # device fields kept as status columns
    'BTV0125': ('isGoodExposure', 'acqImageNumberDiffFromLast'),
    'laser_scope': ('saturated', 'acquired', 'vertOffset_V', 'vertGain_VperLSB', 'horizOffset_ns'),
}
DEVICE_STAMPS = {'BTV0125': ('cycleStampGlob', 'imageTimeStamp')}   # the camera's own timing


def to_seconds(stamp):
    # a stamp (datetime, or a number in ns, us, ms or s) -> seconds since 1970
    if stamp is None:
        return np.nan
    if hasattr(stamp, 'timestamp'):
        return stamp.timestamp()
    try:
        x = float(stamp)
    except (TypeError, ValueError):
        return np.nan
    for limit, scale in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3)):
        if x > limit:
            return x / scale
    return x


def flag_value(v):
    # A status field as a number. CERN Boolean and enum objects arrive as Opaque
    # placeholders that keep their value in their state.
    if isinstance(v, Opaque):
        state = getattr(v, 'state', None)
        v = state.get('value') if isinstance(state, dict) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def waveform_features(a):
    a = np.asarray(a, float)
    baseline = a[:max(5, len(a) // 20)].mean()       # the start of the trace, before the pulse
    b = a - baseline
    i = int(np.argmax(np.abs(b)))
    feats = {'peak': b[i], 'peak_pos': float(i), 'integral': b.sum(), 'baseline': baseline}
    for k, part in enumerate(np.array_split(a, N_WINDOWS)):
        feats[f'w{k:02d}'] = part.mean()
    return feats


def image_features(img):
    img = np.asarray(img, float)
    background = np.median(img)
    noise = 1.4826 * np.median(np.abs(img - background))     # robust noise level
    img = img - background
    img[img < NOISE_CUT * max(noise, 1e-12)] = 0              # keep only the laser spot
    total = img.sum()
    if total <= 0:
        return dict.fromkeys(('intensity', 'cx', 'cy', 'sx', 'sy'), np.nan)
    py, px = img.sum(1), img.sum(0)
    x, y = np.arange(len(px)), np.arange(len(py))
    cx, cy = (px * x).sum() / total, (py * y).sum() / total
    return {'intensity': total, 'cx': cx, 'cy': cy,
            'sx': np.sqrt((px * (x - cx) ** 2).sum() / total),
            'sy': np.sqrt((py * (y - cy) ** 2).sum() / total)}


def device_status(name, d, t):
    out = {f: flag_value(d[f]) for f in STATUS.get(name, ()) if f in d}
    for f in DEVICE_STAMPS.get(name, ()):
        if f in d:
            offset = to_seconds(d[f]) - t
            if abs(offset) < 86400:                  # a real time stamp, not a counter
                out[f'{f}_offset'] = offset
    if 'hotSpotsSamples' in d:                       # RF loop and klystron: two reference samples
        for k, x in enumerate(np.ravel(np.asarray(d['hotSpotsSamples'], float))[:2]):
            out[f'hotspot{k}'] = x
    return out


def reduce_shot(shot):
    # one raw shot -> one flat row of numbers
    t = to_seconds(shot['trigger_stamp'])
    row = {'shot': shot['shot'], 't': t, 'pc_time': shot.get('pc_time')}
    for name, v in shot.items():
        if name in SKIP or name.endswith('_stamp'):
            continue
        row[f'{name}.stamp_offset'] = to_seconds(shot.get(name + '_stamp')) - t
        a = data_array(v)
        if a is not None and np.ndim(a) == 0:
            row[name] = float(a)
        elif a is not None and a.ndim == 1:
            row.update({f'{name}.{k}': x for k, x in waveform_features(a).items()})
        elif a is not None and a.ndim == 2:
            row.update({f'{name}.{k}': x for k, x in image_features(a).items()})
        if isinstance(v, dict):
            row.update({f'{name}.{k}': x for k, x in device_status(name, v, t).items()})
    return row


def is_window(c):
    # True for the waveform window columns <NAME>.w00 ... <NAME>.w47
    tail = c.rsplit('.', 1)[-1]
    return tail[0] == 'w' and tail[1:].isdigit()


def reduce_run(files):
    rows = []
    for i, path in enumerate(files):
        print(f'\r  reducing chunk {i + 1}/{len(files)}', end='', flush=True)
        rows += [reduce_shot(s) for s in load_chunk(path)]
    print()
    table = pd.DataFrame(rows).set_index('shot').sort_index()
    windows = [c for c in table.columns if is_window(c)]
    table[windows] = table[windows].astype(np.float32)        # half the memory, enough precision
    return table