# Put the laser meters and the RF readings of results/datasets back on their own pulse, shot by shot.
#
# extract_features.py stores group B (lab meter, RF) one row late: row i <- row i + 1, i.e. it assumes the reading on
# row i + 1 always belongs to pulse i. On the full runs the PC reads a shot 0.4-4.8 s after its trigger, and the
# energy meters hand over the newest value they have: read early it is the previous pulse, read late it is already
# this pulse (pulses.meter_shift; their own stamps do not show it). So for every late read the dataset row t holds
# shot t + 1's reading: a leak into forecasting and a scrambled same-shot feature.
#
# Worse, 35-40 % of the rows of the high-charge segments are not one pulse apart (the acquisition skips pulses), so
# "one row later" is often not "one pulse later".
#
# Here every reading is moved on the pulse grid with the rules of pulses.py (as run_data.py does): the lab meter to
# pulse + meter_shift(read delay), the RF to pulse + rf_shift(its own stamp offset). A pulse without a reading of its
# own gets NaN, never a neighbour's. The rows themselves (camera, BCM, flags) are not moved.
# Diagnostic in alignment.json: per read-delay bin, the lab reading's correlation with the camera (BTV_intensity, on
# its own pulse, not the target) of the same and of the previous pulse, on the training part.
# Output: results/datasets_aligned/ (same layout as results/datasets) + alignment.json (the calibration).
# python3 -I scripts/meter_alignment.py results/datasets results/datasets_aligned
import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import FLAT_TOP  # noqa: E402
from pulses import meter_shift, rf_shift  # noqa: E402

DATA = Path("/lustre/work/pdl18409/shared/data/optimisation/measured_data")
METERS = {"lab_laser_power": "meter", "PGLA_drop": "rf", "PKI15A": "rf"}
TRAIN = 0.9            # calibration uses the first 90 % of each segment's rows only
N_BINS = 12            # read-delay bins (quantiles)
WIN = 1.5              # a pulse wins a bin if its r is > WIN x the other's and > R_MIN
R_MIN = 0.2


def raw_run(run, chunks):
    """Raw per-row readings of the chunks, in file order: trigger, read delay, meters, RF scalars as extract_features."""
    rows = []
    for c in sorted(set(chunks)):
        with h5py.File(DATA / run / "h5" / f"chunk_{c:05d}.h5", "r") as f:
            pgla = f["PGLA/samples"][:, 0]
            rows.append(pd.DataFrame({
                "chunk": c, "shot_in_chunk": np.arange(len(f["BCM_GUN"])),
                "t": f["trigger_stamp"][()], "pc": f["pc_time"][()],
                "lab_laser_power": f["lab_laser_power"][()],
                "PGLA_drop": pgla[:, 631] - pgla[:, 636],
                "PKI15A": f["PKI15A/samples"][:, 0, FLAT_TOP].mean(axis=1),
                "PGLA_off": f["PGLA_stamp"][()] - f["trigger_stamp"][()],
                "PKI15A_off": f["PKI15A_stamp"][()] - f["trigger_stamp"][()]}))
    d = pd.concat(rows, ignore_index=True)
    d["delay"] = d.pc - d.t
    d["delay"] -= np.round(d.delay.median() / 3600) * 3600
    return d


def jitter(x):
    lx = np.log(np.where(x > 0, x, np.nan)) if np.nanmin(x) > 0 else np.asarray(x, float)
    s = pd.Series(lx)
    return (s - s.rolling(51, center=True, min_periods=10).median()).to_numpy()


def calibrate(delay, reading, cam_same, cam_prev, train):
    """Per delay bin: r with the camera of the same pulse and of the previous pulse -> shift 0, -1 or NaN."""
    edges = np.unique(np.quantile(delay[train], np.linspace(0, 1, N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    jr = jitter(reading)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = train & (delay >= lo) & (delay < hi) & np.isfinite(jr)
        r = {}
        for k, cam in (("same", cam_same), ("prev", cam_prev)):
            mm = m & np.isfinite(cam)
            r[k] = np.corrcoef(jr[mm], cam[mm])[0, 1] if mm.sum() > 50 else np.nan
        if r["prev"] > R_MIN and r["prev"] > WIN * max(r["same"], 0):
            shift = -1
        elif r["same"] > R_MIN and r["same"] > WIN * max(r["prev"], 0):
            shift = 0
        else:
            shift = np.nan
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "r_same": float(r["same"]),
                     "r_prev": float(r["prev"]), "shift": None if np.isnan(shift) else int(shift)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets")
    ap.add_argument("out")
    a = ap.parse_args()
    src, out = Path(a.datasets), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    names = manifest["features"]
    report = {}
    for info in manifest["segments"]:
        z = dict(np.load(src / info["file"]))
        run, n = info["run"], len(z["y"])
        raw = raw_run(run, z["chunk"])
        key = pd.MultiIndex.from_arrays([z["chunk"], z["shot_in_chunk"]])
        raw = raw.set_index(["chunk", "shot_in_chunk"])
        d = raw.reindex(key)                                   # raw row of each dataset row (before any shift)
        T = np.nanmedian(np.diff(d.t.to_numpy()))
        pid = np.round((d.t.to_numpy() - d.t.to_numpy()[0]) / T).astype(int)
        row_of = {p: i for i, p in enumerate(pid)}
        prev_row = np.array([row_of.get(p - 1, -1) for p in pid])
        cam = jitter(z["X"][:, names.index("BTV_intensity")])
        cam_prev = np.where(prev_row >= 0, cam[np.maximum(prev_row, 0)], np.nan)
        train = np.arange(n) < int(TRAIN * n)
        X = z["X"].copy()
        seg_rep = {"pulse_spacing_s": float(T), "rows_not_one_pulse_apart": float(np.mean(np.diff(pid) != 1))}
        for f, kind in METERS.items():
            if f not in names:
                continue
            reading = d[f].to_numpy()
            delay = d.delay.to_numpy()
            bins = calibrate(delay, reading, cam, cam_prev, train) if kind == "meter" else None
            shift = meter_shift(delay).astype(float) if kind == "meter" else rf_shift(d[f.split("_")[0] + "_off"].to_numpy())
            aligned = np.full(n, np.nan)
            for i in np.flatnonzero(np.isfinite(shift)):          # reading of row i belongs to pulse pid[i] + shift
                j = row_of.get(pid[i] + int(shift[i]))
                if j is not None:
                    aligned[j] = reading[i]
            X[:, names.index(f)] = aligned
            old = z["X"][:, names.index(f)]
            same = np.isfinite(aligned) & np.isfinite(old)
            seg_rep[f] = {"camera_check_bins": bins, "shift_share": {str(k): float(np.mean(shift == k)) for k in (-2, -1, 0, 1)},
                          "rows_with_reading": float(np.isfinite(aligned).mean()),
                          "changed_vs_dataset": float(np.mean(~np.isclose(aligned[same], old[same])))}
        z["X"] = X
        np.savez(out / info["file"], **z)
        report[info["segment"]] = seg_rep
        print(f"segment {info['segment']} ({run}): " + ", ".join(
            f"{f} {seg_rep[f]['rows_with_reading']:.2f} kept, {seg_rep[f]['changed_vs_dataset']:.2f} moved"
            for f in METERS if f in seg_rep), flush=True)
    manifest["aligned_by"] = "scripts/meter_alignment.py"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "alignment.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
