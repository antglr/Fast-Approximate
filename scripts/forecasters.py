# Next-shot forecasters shared by the forecast scripts.
#
# hybrid(): the mean of the good shots since the last level change, capped at wmax shots. A level change is a charge
# deviation > z * jitter from the current mean. The modes differ in what confirms it:
#   'charge'          the deviation alone
#   'charge+laser'    the lab meter moved the same way by > z * its own jitter
#   'charge x2'       two shots in a row deviate the same way
# Shots are skipped by causal rules only: unconfirmed falls of > 70 % (dropouts), and BCM readings below half the
# trailing beam-loading-step charge (stuck BCM).
from collections import deque
import numpy as np

DROP = -0.7                    # an unconfirmed fall by more than this (relative) is a dropout, not a level change


def hybrid(Q, lab, step_med, mode, z, wmax, sig_q, sig_l):
    # -> forecast for shot i+1 made after shot i (same index as Q), whether shot i started a new segment, and whether
    #    shot i is in the current mean (not skipped as a dropout, stuck BCM or held-back candidate)
    pred, new, used = np.full(len(Q), np.nan), np.zeros(len(Q), bool), np.zeros(len(Q), bool)
    buf, sq, sl, cand = deque(), 0.0, 0.0, []
    last = np.nan
    for i, (qi, li, si) in enumerate(zip(Q, lab, step_med)):
        ok = np.isfinite(qi) and not (si > 1 and qi < 0.5 * si)          # BCM stuck low while the beam is there
        if ok and buf:
            m, ml = sq / len(buf), sl / len(buf)
            rq, rl = qi / m - 1, li / ml - 1
            big = abs(rq) > z * sig_q
            if mode == 'charge':
                change = big
            elif mode == 'charge+laser':
                change = big and abs(rl) > z * sig_l and np.sign(rl) == np.sign(rq)
            else:                                                       # two in a row, same direction
                change = big and bool(cand) and np.sign(cand[-1] / m - 1) == np.sign(rq)
                cand = [qi] if big and not change else []
            if change:
                start = ([cand[-1]] if mode == 'charge x2' and cand else []) + [qi]
                buf, sq, sl = deque(zip(start, [li] * len(start))), sum(start), li * len(start)
                new[i] = used[i] = True
                ok = False                                              # already in the new segment
            elif big and (mode == 'charge x2' or rq < DROP):
                ok = False                                              # held back: a candidate or a dropout
        if ok:
            used[i] = True
            buf.append((qi, li)); sq += qi; sl += li
            if len(buf) > wmax:
                a, b = buf.popleft(); sq -= a; sl -= b
        if buf:
            last = sq / len(buf)
        pred[i] = last
    return pred, new, used


assert np.allclose(hybrid(np.r_[np.ones(5), 3 * np.ones(3)], np.r_[np.ones(5), 3 * np.ones(3)], np.zeros(8),
                          'charge+laser', 4, 9, 0.03, 0.03)[0], [1, 1, 1, 1, 1, 3, 3, 3])
