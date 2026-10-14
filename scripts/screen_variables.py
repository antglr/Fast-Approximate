# Screen the notebook variables (klystron, gun field, magnets, temperatures, humidities, laser scope) for the next shot:
# (1) same-shot r with the charge jitter (alignment check, replicates the notebooks),
# (2) r of their causal deviations (1/9/50-shot segment mean minus the 200-shot one) with the hybrid's next-shot
#     relative error on stable shots,
# (3) leave-one-run-out ridge: stable test MAE with the combined features plus each group.
# The scope is tried at its stamp alignment and one pulse either side: at -1 it is the NEXT shot's laser (a leak).
# Usage: python -I screen_variables.py <tables dir>
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from combined_forecast import prepare, seg_mean, ALPHAS
from nbfuncs import TARGET, scored_shots
from pulses import align, pulse_ids, rf_shift, stamp_shift

T = sys.argv[1]
RUNS = ['2026-10-09_07-46-07', '2026-10-08_21-11-47', '2026-10-08_00-08-19']
R = {}
for run in RUNS:
    r = prepare(T, run); df = r['df']
    p = os.path.join(T, run)
    d, L = np.load(p + '_rf.npz'), np.load(p + '_laser.npz')
    pid = pulse_ids(d['t'], d['chunk'])
    V = {}
    rf = lambda k: align(d[k], pid, rf_shift(d[k + '_off']))
    for k in ('PKI15A', 'PKI15P', 'PKR15A'):
        V[k] = rf(k)[:, 300:640].mean(1)
    P = rf('PGLA'); V['gun field'] = P[:, 400:600].mean(1) - P[:, :40].mean(1)
    sc, t = L['scope'].astype(float), L['scope_times'][0]
    clean = sc - sc[:, t < 5].mean(1, keepdims=True)
    tmpl = np.nanmean(clean, 0) * ((t > 5.6) & (t < 39.8))
    amp = clean @ tmpl / (tmpl @ tmpl)
    bad = np.r_[False, np.all(sc[1:] == sc[:-1], 1)] | ~L['scope_acquired'].astype(bool) | L['scope_saturated'].astype(bool)
    amp[bad] = np.nan
    for k in (-1, 0, 1):
        V[f'scope amp (stamp{k:+d})'] = align(amp, pid, stamp_shift(L['scope_off']) + k)
    for k in ('DHG0130', 'SNH_110', 'SNI_120', 'TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'HUMID1', 'HUMID2', 'HUMID3', 'HUMID4'):
        V[k] = df[k].values.astype(float)
    V['lab'], V['clex'], V['step'] = r['x'].lab.values, r['x'].clex.values, r['x'].step.values
    pos = np.searchsorted(d['shot'], df.index)          # the npz arrays cover the whole run, df only the clean span
    V = {k: v if len(v) == len(df) else v[pos] for k, v in V.items()}
    r['V'] = pd.DataFrame(V, df.index).replace([np.inf, -np.inf], np.nan)
    R[run] = r
    print(run, 'scope bad', bad.mean().round(3), flush=True)

def det(v):  # analysis-only detrend (centred), for the same-shot check
    return v - v.rolling(51, center=True, min_periods=10).mean()

rows = []
for run, r in R.items():
    st = r['stable']
    qj = det(r['q'])
    for k, v in r['V'].items():
        vj = det(v)
        m = st & vj.notna() & qj.notna()
        # next-shot: causal deviation of the variable vs the hybrid's next-shot relative error, on fit rows
        dev = {kk: seg_mean(v.interpolate(limit_direction='both'), r['seg'], kk) - seg_mean(v.interpolate(limit_direction='both'), r['seg'], 200) for kk in (1, 9, 50)}
        f = r['fit'] | (st.shift(-1, fill_value=False).values & r['on'] & ~r['train'])
        y = pd.Series(r['y'], v.index)
        rows.append({'run': run[:13], 'var': k, 'same-shot r': np.corrcoef(vj[m], qj[m])[0, 1] if m.sum() > 50 else np.nan,
                     **{f'next r dev{kk}': (lambda a, b: np.corrcoef(a, b)[0, 1])(*(s[f & np.isfinite(y.values) & dev[kk].notna().values] for s in (dev[kk], y))) for kk in (1, 9, 50)}})
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(pd.DataFrame(rows).pivot_table(index='var', columns='run', values=['same-shot r', 'next r dev1', 'next r dev9', 'next r dev50']).round(3).to_string())

# leave-one-run-out ridge on stable shots: base features + each group's deviations, standardised per run's train std
GROUPS = {'base': [], 'klystron': ['PKI15A', 'PKI15P', 'PKR15A'], 'gun field': ['gun field'], 'scope': ['scope amp (stamp+0)'],
          'magnets': ['DHG0130', 'SNH_110', 'SNI_120'], 'temp': ['TEMP1', 'TEMP2', 'TEMP3', 'TEMP4'],
          'humid': ['HUMID1', 'HUMID2', 'HUMID3', 'HUMID4']}
GROUPS['all'] = sum(GROUPS.values(), [])
def feats(r, names):
    F = r['F'].copy()
    for k in names:
        v = r['V'][k].interpolate(limit_direction='both')
        sd = v[r['train']].diff().std() or 1
        for kk in (1, 9, 50):
            F[f'{k}{kk}'] = ((seg_mean(v, r['seg'], kk) - seg_mean(v, r['seg'], 200)) / sd).clip(-10, 10)
    return F.fillna(0)
out = []
for g, names in GROUPS.items():
    for test in RUNS:
        tr = [u for u in RUNS if u != test]
        m = RidgeCV(alphas=ALPHAS).fit(pd.concat([feats(R[u], names)[R[u]['fit']] for u in tr]), np.concatenate([R[u]['y'][R[u]['fit']] for u in tr]))
        r = R[test]
        sh = scored_shots(r['df']); yq = r['df'].loc[sh, TARGET].values
        pred = pd.Series(r['p'].values * (1 + r['on'] * m.predict(feats(r, names))), r['df'].index).shift(1).reindex(sh).values
        st = (r['stable'] & ~r['train']).reindex(sh).values
        out.append({'group': g, 'test run': test[:13], 'stable MAE': np.nanmean(np.abs(yq - pred)[st])})
O = pd.DataFrame(out).pivot(index='group', columns='test run', values='stable MAE')
print('\nstable test MAE (nC), correction trained on the other two runs\n', O.round(5).to_string())
print('\nchange vs base (%)\n', (100 * (O / O.loc['base'] - 1)).round(2).to_string())
