# Keep the full 960-sample RF traces (the table keeps only 48 window means), with each trace's stamp offset.
# Usage: python -I extract_rf.py <chunk dir> <out.npz>
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbfuncs import *
from h5chunk import chunk_files, load_any

RF = ['PGIA', 'PGRA', 'PGLA', 'PGIP', 'PKI15A', 'PKR15A', 'PKI15P']
out = {k: [] for k in RF + [k + '_off' for k in RF] + ['Q', 'shot', 'chunk', 't']}
for f in chunk_files(sys.argv[1]):
    for s in load_any(f):
        t = to_seconds(s['trigger_stamp'])
        out['Q'].append(s['BCM_GUN'])
        out['shot'].append(s['shot'])
        out['t'].append(t)
        out['chunk'].append(int(os.path.basename(f)[6:11]))
        for k in RF:
            out[k].append(np.asarray(data_array(s[k]), float))
            out[k + '_off'].append(to_seconds(s.get(k + '_stamp')) - t)
    print(os.path.basename(f), flush=True)
np.savez_compressed(sys.argv[2], **{k: np.array(v) for k, v in out.items()})
print('->', sys.argv[2])
