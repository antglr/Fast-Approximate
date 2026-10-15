# Does the alignment matter? For two extract_features*.py outputs (e.g. results/datasets and results/datasets_aligned):
# per segment, the correlation of each input's fast part with the charge's fast part (fast = log value minus the mean of
# the previous 50 rows; raw value for inputs that can be <= 0), and the test R^2 of a ridge of the charge's fast part on
# all inputs' fast parts (first 90 % of the segment to fit, last 10 % to test; inputs in robust sd, clipped at 5).
# Rows: each dataset's own valid ones, without BCM-lost or doubled shots (charge fast part beyond log 2).
# Usage: python -I compare_datasets.py <dataset dir> <dataset dir>
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


def fast(v):
    v = pd.Series(v)
    lv = np.log(v.where(v > 0)) if (v.dropna() > 0).mean() > 0.99 else v
    return lv - lv.shift(1).rolling(50, min_periods=5).mean()


def scores(path):
    d = np.load(path, allow_pickle=False)
    y = fast(d['y'])
    X = pd.DataFrame({f: fast(d['X'][:, j]) for j, f in enumerate(d['features'])})
    m = d['valid'] & (y.abs() < np.log(2)).values & X.notna().all(axis=1).values
    sd = 1.4826 * (X[m] - X[m].median()).abs().median().replace(0, 1)
    X = X.clip(-5 * sd, 5 * sd, axis=1) / sd
    r = X[m].corrwith(y[m]).to_dict()
    n = int(0.9 * len(y))
    tr, te = m & (np.arange(len(y)) < n), m & (np.arange(len(y)) >= n)
    p = RidgeCV(alphas=np.logspace(-4, 4, 17)).fit(X[tr], y[tr]).predict(X[te])
    r['ridge test R2'] = 1 - ((y[te] - p) ** 2).mean() / ((y[te] - y[te].mean()) ** 2).mean()
    r['valid rows'] = int(m.sum())
    return r


def main():
    rows = {}
    for d in sys.argv[1:3]:
        for f in sorted(os.listdir(d)):
            if f.startswith('segment_') and f.endswith('.npz'):
                rows[(os.path.basename(os.path.normpath(d)), int(f[8:10]))] = scores(os.path.join(d, f))
    D = pd.DataFrame(rows).T
    pd.set_option('display.width', 250)
    print('correlation with the charge fast part, ridge test R2 and valid rows, per dataset and segment')
    print(D.T.round(3).to_string())


if __name__ == '__main__':
    main()
