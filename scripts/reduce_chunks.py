# Reduce raw chunks to one row per shot with the kickoff notebook's functions, one chunk in memory at a time.
# Usage: python -I reduce_chunks.py <chunk dir> <out.parquet>
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbfuncs import *
from h5chunk import chunk_files, load_any

src, out = sys.argv[1], sys.argv[2]
rows = []
for f in chunk_files(src):
    name = os.path.basename(f)
    rows += [dict(reduce_shot(s), chunk=name) for s in load_any(f)]
    print(name, flush=True)
t = pd.DataFrame(rows).set_index('shot').sort_index()
w = [c for c in t.columns if is_window(c)]
t[w] = t[w].astype(np.float32)
t.to_parquet(out)
print(t.shape, '->', out)
