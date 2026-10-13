# Read a converted HDF5 chunk (measured_data/acquisition_<run>/h5/chunk_XXXXX.h5) as the same list of shot dicts
# that nbfuncs.load_chunk gives for the pickles, so the extract scripts work on either.
# The camera group is skipped (images are 470 MB per chunk and handled elsewhere).
import ast, datetime, os, zoneinfo
import h5py
import numpy as np
from nbfuncs import load_chunk

SKIP = ('BTV0125', 'parameters')        # parameters is an empty group in the h5 files
CERN = zoneinfo.ZoneInfo('Europe/Zurich')


def _value(x):
    # h5 strings are bytes; pyda Booleans were stored as their repr, e.g. b"pyda.data.Boolean(state={'value': True, ...
    if isinstance(x, bytes):
        x = x.decode()
        if x.startswith('pyda.data.Boolean'):
            return ast.literal_eval(x.split("'value': ")[1].split(',')[0])
        return x
    return x.item() if isinstance(x, np.generic) else x


def load_h5(path):
    with h5py.File(path, 'r') as f:
        n = int(f.attrs['n_shots'])
        shots = [{} for _ in range(n)]
        for name, o in f.items():
            if name in SKIP:
                continue
            if isinstance(o, h5py.Dataset):
                a = o[()]
                for i in range(n):
                    shots[i][name] = _value(a[i])
            else:                                   # a device property: one dict per shot
                cols = {k: v[()] for k, v in o.items() if isinstance(v, h5py.Dataset)}
                for i in range(n):
                    shots[i][name] = {k: _value(v[i]) for k, v in cols.items()}
    for s in shots:                                 # pickles keep pc_time as a naive CERN local datetime
        s['pc_time'] = datetime.datetime.fromtimestamp(s['pc_time'], CERN).replace(tzinfo=None)
    return shots


def load_any(path):
    return load_h5(path) if path.endswith('.h5') else load_chunk(path)


def chunk_files(folder):
    # chunk_*.pkl or chunk_*.h5, sorted by chunk number
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.startswith('chunk_') and f.endswith(('.pkl', '.h5')))


assert _value(b"pyda.data.Boolean(state={'value': False, 'meaning': pyda.data.Meaning('NONE')})") is False
