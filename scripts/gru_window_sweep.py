# Context-window sweep for the GRU (Hard tier: "tune the context window and show its effect"): reruns gru_forecast.py
# (pooled-runs variant, same seeds) for each window length L. Pick L on the validation L1 that gru_forecast prints
# (the end of the training part), not on the test MAE.
# Usage: python -I gru_window_sweep.py <tables dir> <out dir> <run> [<run> ...]   -> <out dir>/L<L>/ per window length
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gru_forecast as G

LS = (8, 16, 32, 64, 128, 256)

tables, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
G.VARIANTS = ('pooled runs',)
for L in LS:
    G.L = L                                       # windows(), fit() and main() read the module global
    d = os.path.join(out, f'L{L}')
    os.makedirs(d, exist_ok=True)
    print(f'=== L = {L}', flush=True)
    sys.argv = [sys.argv[0], tables, d, *runs]
    G.main()
