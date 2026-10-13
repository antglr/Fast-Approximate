# Keep the laser monitors per shot: both energy meters, the full photodiode (scope) trace and its flags,
# each with its stamp offset. The camera image is not kept.
# Usage: python -I extract_laser.py <chunk dir> <out.npz>
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbfuncs import *
from h5chunk import chunk_files, load_any

out = {k: [] for k in ('Q', 'shot', 'chunk', 't', 'lab', 'clex', 'scope', 'scope_times', 'scope_saturated',
                       'scope_acquired', 'lab_off', 'clex_off', 'scope_off')}
for f in chunk_files(sys.argv[1]):
    for s in load_any(f):
        t = to_seconds(s['trigger_stamp'])
        sc = s['laser_scope']
        out['Q'].append(s['BCM_GUN'])
        out['shot'].append(s['shot'])
        out['t'].append(t)
        out['chunk'].append(int(os.path.basename(f)[6:11]))
        out['lab'].append(s['lab_laser_power'])
        out['clex'].append(s['clex_laser_power'])
        out['scope'].append(np.asarray(sc['signal'], np.float32).ravel())
        out['scope_times'].append(np.asarray(sc['times'], np.float32).ravel())
        out['scope_saturated'].append(flag_value(sc.get('saturated')))
        out['scope_acquired'].append(flag_value(sc.get('acquired')))
        for k, n in (('lab_off', 'lab_laser_power'), ('clex_off', 'clex_laser_power'), ('scope_off', 'laser_scope')):
            out[k].append(to_seconds(s.get(n + '_stamp')) - t)
    print(os.path.basename(f), flush=True)
np.savez_compressed(sys.argv[2], **{k: np.array(v) for k, v in out.items()})
print('->', sys.argv[2])
