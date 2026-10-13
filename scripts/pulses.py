# Put readings on the machine-pulse grid. The acquisition skips pulses (rows are 1.2, 2.4 or 3.6 s
# apart), so shifting by rows, as the kickoff notebook's synchronise() does, puts readings on the wrong shot.
import numpy as np
import pandas as pd

T = 1.2   # CLEAR pulse spacing in run 2026-10-08_00-08-19 (0.833 Hz); check it for other runs


def pulse_ids(t, group):
    # unique pulse number per row; counted from each group's first row, because sample chunks are not contiguous
    p = np.zeros(len(t), np.int64)
    for g in np.unique(group):
        m = group == g
        p[m] = np.round((t[m] - t[m][0]) / T)
    return np.asarray(group, np.int64) * 10**6 + p


def stamp_shift(offset):
    # pulses between a reading's own stamp and the shot's stamp
    return np.round(np.asarray(offset, float) / T)


def rf_shift(offset):
    # RF traces hold the pulse BEFORE the one their stamp names: the PGLA beam-loading step matches the charge
    # (and the laser meter) only after this shift. Found on the 9 sample chunks; check on full runs.
    return stamp_shift(offset) - 1


def read_delay(pc_time, t):
    # seconds from the trigger to when acquire.py finished reading the shot; pc_time is local time, t is UTC
    d = pd.to_datetime(pd.Series(pc_time)).astype('int64').to_numpy() / 1e9 - np.asarray(t, float)
    return d - np.round(np.nanmedian(d) / 3600) * 3600


METER_SWITCH = 2.2   # s; found on the 9 sample chunks (1.9-2.6 s is ambiguous), check on full runs


def meter_shift(delay):
    # The energy meters hand over the newest value they have published: read soon after the trigger
    # that is still the previous pulse, read late it is this pulse. Their own stamps do not show this.
    return np.where(np.asarray(delay) < METER_SWITCH, -1, 0)


def align(X, pid, shift):
    # move row i's reading to the row of pulse pid[i] + shift[i]; NaN where that pulse has no reading
    X = np.asarray(X, float)
    shift = np.broadcast_to(np.asarray(shift, float), len(pid))
    row = {k: i for i, k in enumerate(pid)}
    out = np.full_like(X, np.nan)
    for i, s in enumerate(shift):
        j = row.get(pid[i] + int(s)) if np.isfinite(s) else None
        if j is not None:
            out[j] = X[i]
    return out


def jitter(v, t, group):
    # remove each group's linear trend, leaving the shot-to-shot part (1-D, or 2-D column-wise)
    v = np.array(v, float)
    out = np.full_like(v, np.nan)
    for g in np.unique(group):
        m = (group == g) & np.isfinite(v).reshape(len(v), -1).all(1)
        if m.sum() > 3:
            A = np.c_[np.ones(m.sum()), t[m] - t[m].mean()]
            out[m] = v[m] - A @ np.linalg.lstsq(A, v[m], rcond=None)[0]
    return out


def corr(x, y, m):
    m = m & np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 20 and x[m].std() > 0 else np.nan


if __name__ == '__main__':   # self-check: a reading recorded one pulse late, with a skipped pulse in between
    t = np.array([0, 1.2, 3.6, 4.8])                 # pulse 2 (2.4 s) was not recorded
    pid = pulse_ids(t, np.zeros(4, int))
    assert list(pid) == [0, 1, 3, 4]
    x = align([10., 11., 13., 14.], pid, -1)         # each reading belongs to the previous pulse
    assert np.array_equal(x, [11., np.nan, 14., np.nan], equal_nan=True)   # 13 belonged to the unrecorded pulse
    assert np.allclose(jitter(2 * t + 5, t, np.zeros(4)), 0)                 # a pure drift leaves no jitter
    assert list(meter_shift([0.9, 3.0])) == [-1, 0]
    assert list(rf_shift([-0.11, 1.09])) == [-1, 0]
    print('pulses.py self-check OK')
