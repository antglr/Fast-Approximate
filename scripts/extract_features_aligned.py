# Per-segment model datasets from the clean chunks (config/CLEAN_CUTS.toml): X = FEATURES, y = BCM_GUN.
# Preprocessing as notebooks 05_correlations_runs / 07_02 / 07_03: laser matched filter with a per-segment template,
# group B one row later within the segment, classic BTV parameters (notebook 04) + spot-shape PCA, per-image SVD
# singular values (NaN on a blank image, as the BTV parameters), and optionally the across-image PCA scores
# (notebook 04_1 / 07_02, fitted on all shots: leaking).
# Rows are never dropped (time congruency): frozen laser / camera shots and the shift's last row are flagged instead.
# Output: OUT/segment_XX.npz per segment + OUT/manifest.json.  Usage: python -I extract_features_aligned.py
#
# ALIGNED COPY of extract_features.py (same features, same files layout, written to results/datasets_aligned).
# extract_features.py moves all of group B one ROW later and leaves the camera on its raw row. Two problems:
#   - the acquisition skips pulses (rows are 1.2, 2.4 or 3.6 s apart), so a row is not always a pulse;
#   - the shift is not the same for every reading: the energy meters hold the previous pulse only when read early
#     (< 2.2 s after the trigger), the RF traces hold the pulse before the one their stamp names, the scope and the
#     camera follow their own stamps (the camera's is one pulse late on 8-20 % of rows).
# Here every reading is put on its pulse with pulses.py (as the forecasting tables): pulse number from the trigger
# stamp, then meters: meter_shift(read delay), RF (PGLA, PKI15A/P, PKR15A): rf_shift(stamp - trigger), scope:
# stamp_shift(stamp - trigger), camera: stamp_shift(cycleStampGlob - trigger). A pulse whose reading was not
# recorded gets NaN (and valid = False) instead of a neighbour's reading. 'complete' = all group-B readings found.
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulses import align, meter_shift, pulse_ids, rf_shift, stamp_shift

try:
    import tomllib                 # Python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib        # Python 3.9 venv on the cluster

# ---------------------------------------------------------------------------
# Configuration (edit here)
# ---------------------------------------------------------------------------
# Union of the per-segment selections of 07_03 (Overall r, R_MIN = 0.25, R_CROSS = 0.5), camera described by the
# spot-shape PCA and the per-image SVD (no fit, no leakage): results/data_labelling/segment_correlations/selected_features.json
FEATURES = ["lab_laser_power", "PGLA_drop", "SVD_s1", "BTV_intensity", "SVD_s5", "SVD_s6", "SVD_s4", "PKI15A",
            "BTV_tilt", "BTV_mean_x"]
TARGET = "BCM_GUN"

REPO = Path(__file__).resolve().parent.parent
CUTS_FILE = REPO / "config" / "CLEAN_CUTS.toml"
OUT = REPO / "results" / "datasets_aligned"
DATA_CANDIDATES = [
    os.environ.get("FA_DATA_ROOT"),
    "/lustre/work/pdl18409/shared/data/optimisation/measured_data",
    "/data/optimisation/measured_data",
]
N_JOBS = int(os.environ.get("FA_JOBS", min(32, os.cpu_count() or 1)))   # shared node: don't take all cores

# preprocessing constants, identical to the notebooks
FLAT_TOP = slice(300, 640)                   # RF flat top (05)
LASER_BASE, LASER_WIN = 5.0, (5.6, 39.8)     # ns: baseline before, template window (03 / 05)
ROI = (slice(350, 1150), slice(600, 1450))   # camera ROI rows, cols (04 / 04_1)
BG_ROWS = slice(0, 150)                      # camera background rows of the full image
THRESHOLD = 0.05                             # beam mask: pixels above 5 % of the peak (04)
BIN = 4                                      # PCA / SVD image binning (04_1 / 07_02)
N_PCA = 6                                    # across-image PCA components (fitted on all shots: leaking)
N_SVD = 6                                    # per-image singular values

GROUP_B = ["laser_amp", "lab_laser_power", "PGLA_drop", "PKI15A", "PKI15P", "PKR15A"]  # not on their own row
STAMPS = {"lab_laser_power": "lab_laser_power_stamp", "PGLA_drop": "PGLA_stamp", "PKI15A": "PKI15A_stamp",
          "PKI15P": "PKI15P_stamp", "PKR15A": "PKR15A_stamp", "laser_amp": "laser_scope_stamp"}
BTV_COLS = ["amplitude", "intensity", "mean_x", "mean_y", "sigma_x", "sigma_y",       # notebook 04
            "sigma_major", "sigma_minor", "tilt", "ellipticity"]                       # spot-shape PCA
PCS = [f"PC{k + 1}" for k in range(N_PCA)]
SVDS = [f"SVD_s{k + 1}" for k in range(N_SVD)]
AVAILABLE = ["laser_amp", "lab_laser_power", "PGLA_drop", "PKI15A", "PKI15P", "PKR15A",
             "DHG0130", "SNH_110", "SNI_120", *[f"BTV_{c}" for c in BTV_COLS], *PCS, *SVDS]


# ---------------------------------------------------------------------------
# Segments of the stitched clean acquisition (same numbering as 07_01)
# ---------------------------------------------------------------------------
def expand(keep):
    """[[0, 10], 12, [15, 20]] -> chunk numbers; a pair is an inclusive range, a single int one chunk."""
    out = []
    for item in keep:
        if isinstance(item, int):
            out.append(item)
        else:
            a, b = item
            out.extend(range(a, b + 1))
    return out


def clean_chunks(cuts_file, root):
    """{run: [chunk paths to use]}; warns about listed chunks that do not exist."""
    with open(cuts_file, "rb") as f:
        cuts = tomllib.load(f)
    out = {}
    for run, spec in cuts.items():
        files = {int(p.stem[6:]): p for p in sorted((root / run).glob("**/chunk_*.h5"))}   # <run>/h5/ or <run>/ (local copy)
        wanted = sorted(files) if spec["keep"] == "all" else expand(spec["keep"])
        absent = [i for i in wanted if i not in files]
        if absent:
            print(f"{run}: listed but not found: chunks {absent}")
        out[run] = [files[i] for i in wanted if i in files]
    return out


def segments_of(clean):
    """[(run, [chunk paths])] per contiguous segment: a new one at every new run and wherever chunks were removed."""
    segs = []
    for run in sorted(clean):
        prev = None
        for p in clean[run]:
            c = int(p.stem[6:])
            if prev is None or c != prev + 1:
                segs.append((run, []))
            segs[-1][1].append(p)
            prev = c
    return segs


# ---------------------------------------------------------------------------
# Per-chunk worker
# ---------------------------------------------------------------------------
def first_non_empty(obj):
    arrays = [obj[k][:] for k in sorted(obj)] if isinstance(obj, h5py.Group) else obj[:]
    return next((a for a in arrays if a.size), None)


def beam_params(img, x_mm, y_mm):
    """
    BTV parameters of one cleaned ROI image, in BTV_COLS order: classic ones as notebook 04 (with the blank-image fix
    of 05_correlations_runs), plus the spot-shape PCA: eigen-decomposition of the intensity-weighted 2x2 covariance
    inside the same beam mask -> RMS sizes along the spot's own axes, tilt of the major axis (deg, in the mm axes
    of the camera) and ellipticity = minor / major.
    """
    smooth = ndimage.gaussian_filter(img, 3)
    peak = smooth.max()
    labels, n_blobs = ndimage.label(smooth > THRESHOLD * peak)
    if peak <= 0 or n_blobs == 0:                    # blank image: no beam, or the camera sent zeros
        return [np.nan] * len(BTV_COLS)
    img = img * (labels == np.bincount(labels.ravel())[1:].argmax() + 1)
    total = img.sum()
    proj_x, proj_y = img.sum(axis=0), img.sum(axis=1)
    mean_x = proj_x @ x_mm / total
    mean_y = proj_y @ y_mm / total
    dx, dy = x_mm - mean_x, y_mm - mean_y
    var_x, var_y = proj_x @ dx**2 / total, proj_y @ dy**2 / total
    cov_xy = dy @ img @ dx / total
    eig = np.linalg.eigvalsh([[var_x, cov_xy], [cov_xy, var_y]])  # ascending
    minor, major = np.sqrt(np.maximum(eig, 0))
    tilt = 0.5 * np.degrees(np.arctan2(2 * cov_xy, var_x - var_y))
    return [peak, total, mean_x, mean_y, np.sqrt(var_x), np.sqrt(var_y), major, minor, tilt, minor / major]


def chunk_features(fp, x_mm, y_mm, need_classic, need_pca, need_svd):
    """One chunk: scalar/RF features, baseline-subtracted laser pulses, frozen flags, camera parameters and images."""
    need_camera = need_classic or need_pca or need_svd
    with h5py.File(fp, "r") as f:
        sig = f["laser_scope/signal"][()]
        t = f["laser_scope/times"][0]
        pgla = f["PGLA/samples"][:, 0]
        feats = {
            TARGET: f[TARGET][()],
            "lab_laser_power": f["lab_laser_power"][()],
            "PGLA_drop": pgla[:, 631] - pgla[:, 636],
            "PKI15A": f["PKI15A/samples"][:, 0, FLAT_TOP].mean(axis=1),
            "PKI15P": f["PKI15P"][:, 0, FLAT_TOP].mean(axis=1),
            "PKR15A": f["PKR15A"][:, 0, FLAT_TOP].mean(axis=1),
            "DHG0130": f["DHG0130"][()],
            "SNH_110": f["SNH_110"][()],
            "SNI_120": f["SNI_120"][()],
        }
        image_number = f["BTV0125/imageNumber"][()]
        stamps = {"trigger": f["trigger_stamp"][()], "pc_time": f["pc_time"][()],
                  "camera": f["BTV0125/cycleStampGlob"][()] / 1e9, **{k: f[v][()] for k, v in STAMPS.items()}}
        if need_camera:
            ds = f["BTV0125/image2D"]
            roi = ds[:, ROI[0], ROI[1]]                                   # only ROI and background rows are read
            bg = np.median(ds[:, BG_ROWS, :].reshape(len(roi), -1), axis=1)
    out = {"feats": feats, "laser": (sig - sig[:, t < LASER_BASE].mean(axis=1, keepdims=True)).astype(np.float32),
           "laser_frozen": np.r_[False, (sig[1:] == sig[:-1]).all(axis=1)], "sig_first": sig[:1], "sig_last": sig[-1:],
           "t": t, "image_number": image_number, "stamps": stamps}
    if need_camera:
        h, w = (s // BIN * BIN for s in roi.shape[1:])
        classic = np.full((len(roi), len(BTV_COLS)), np.nan)
        binned = np.empty((len(roi), h // BIN, w // BIN), np.float32) if need_pca else None
        svd = np.full((len(roi), N_SVD), np.nan)
        for i, (img, b) in enumerate(zip(roi, bg)):
            img = ndimage.median_filter(img, size=3).astype(np.float32) - b   # median of ints is exact: same as on floats
            if need_classic:
                classic[i] = beam_params(img, x_mm, y_mm)
            if need_pca or need_svd:
                small = img[:h, :w].reshape(h // BIN, BIN, w // BIN, BIN).mean(axis=(1, 3))
                if need_pca:
                    binned[i] = small
                # per-image SVD of the binned image: singular values only (the basis differs per image); a blank
                # image (no beam found) keeps NaN: its singular values are background only
                if need_svd and not np.isnan(classic[i, 0]):
                    svd[i] = np.linalg.svd(small, compute_uv=False)[:N_SVD]
        out["classic"], out["binned"], out["svd"] = classic, binned, svd
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def shift_up(a, seg):
    """Row i <- row i + 1 of the same segment; NaN (False for bool) on each segment's last row."""
    out = np.full_like(a, False if a.dtype == bool else np.nan)
    same = seg[1:] == seg[:-1]
    out[:-1][same] = a[1:][same]
    return out


def main():
    unknown = [f for f in FEATURES if f not in AVAILABLE]
    if unknown:
        raise SystemExit(f"unknown features {unknown}; available: {AVAILABLE}")
    need_pca = any(f in PCS for f in FEATURES)
    need_svd = any(f in SVDS for f in FEATURES)
    need_classic = any(f.startswith("BTV_") for f in FEATURES) or need_svd   # the SVD needs the blank-image check

    data_root = next(Path(p) for p in DATA_CANDIDATES if p and Path(p).is_dir())
    segs = segments_of(clean_chunks(CUTS_FILE, data_root))
    files = [p for _, ps in segs for p in ps]
    file_seg = [k for k, (_, ps) in enumerate(segs) for _ in ps]
    file_run = [r for r, ps in segs for _ in ps]
    print(f"DATA_ROOT = {data_root}\n{len(segs)} segments, {len(files)} chunks, features: {FEATURES}")

    # pixel -> mm axes, assumed the same for all shots (as notebook 04); first non-empty one over the chunks
    x_mm = y_mm = None
    if need_classic:
        for p in files:
            with h5py.File(p, "r") as f:
                x, y = first_non_empty(f["BTV0125/imagePositionSet1"]), first_non_empty(f["BTV0125/imagePositionSet2"])
            if x is not None and y is not None:
                x_mm, y_mm = x[ROI[1]], y[ROI[0]]
                break
        if x_mm is None:
            raise SystemExit("no chunk has BTV0125 position axes: cannot compute BTV_* in mm")

    # one parallel pass over all chunks, results assembled in stitched order
    n_per_file = []
    for p in files:
        with h5py.File(p, "r") as f:
            n_per_file.append(int(f.attrs["n_shots"]))
    n = sum(n_per_file)
    starts = np.r_[0, np.cumsum(n_per_file)[:-1]]
    seg = np.repeat(file_seg, n_per_file)
    run = np.repeat(file_run, n_per_file)
    chunk = np.concatenate([[int(p.stem[6:])] * m for p, m in zip(files, n_per_file)])
    shot_in_chunk = np.concatenate([np.arange(m) for m in n_per_file])

    cols = {k: np.empty(n) for k in [TARGET, *GROUP_B[1:], "DHG0130", "SNH_110", "SNI_120"]}
    laser = laser_frozen = image_number = classic = images = svd = None
    stamps = {}
    gen = Parallel(n_jobs=N_JOBS, verbose=2, return_as="generator")(
        delayed(chunk_features)(p, x_mm, y_mm, need_classic, need_pca, need_svd) for p in files)
    for i, (s, m, r) in enumerate(zip(starts, n_per_file, gen)):
        if laser is None:
            t = r["t"]
            laser = np.empty((n, r["laser"].shape[1]), np.float32)
            laser_frozen, image_number = np.zeros(n, bool), np.zeros(n, np.int64)
            classic = np.full((n, len(BTV_COLS)), np.nan) if need_classic else None
            # ~170 kB per shot (float32, 200 x 212 binned): ~11 GB for ~63k shots, as in 07_02
            images = np.empty((n, int(np.prod(r["binned"].shape[1:]))), np.float32) if need_pca else None
            svd = np.full((n, N_SVD), np.nan) if need_svd else None
        for k, v in r["feats"].items():
            cols[k][s:s + m] = v
        laser[s:s + m] = r["laser"]
        laser_frozen[s:s + m] = r["laser_frozen"]
        if i and seg[s] == seg[s - 1]:                                   # frozen across the chunk boundary
            laser_frozen[s] = (r["sig_first"] == prev_last).all()
        prev_last = r["sig_last"]
        image_number[s:s + m] = r["image_number"]
        for k, v in r["stamps"].items():
            stamps.setdefault(k, np.empty(n))[s:s + m] = v
        if need_classic:
            classic[s:s + m] = r["classic"]
        if need_pca:
            images[s:s + m] = r["binned"].reshape(m, -1)
        if need_svd:
            svd[s:s + m] = r["svd"]
    btv_frozen = np.r_[False, (np.diff(image_number) == 0) & (seg[1:] == seg[:-1])]  # camera repeated its image

    # laser matched filter, template = mean pulse of the segment (07_02)
    window = (t > LASER_WIN[0]) & (t < LASER_WIN[1])
    cols["laser_amp"] = np.empty(n)
    for k in range(len(segs)):
        msk = seg == k
        template = laser[msk].mean(axis=0) * window
        cols["laser_amp"][msk] = laser[msk] @ template / (template @ template)
    del laser
    if need_classic:
        for j, c in enumerate(BTV_COLS):
            cols[f"BTV_{c}"] = classic[:, j]
    if need_svd:
        for j, c in enumerate(SVDS):
            cols[c] = svd[:, j]

    pca_info = {}
    if need_pca:
        # NOTE (leakage): one PCA fitted on ALL stitched shots, as in 07_02, so the components "see" every segment,
        # including shots that a forecasting model will be tested on. For a proper evaluation, fit the PCA on the
        # training shots only (or chronologically up to the forecast time) and project the rest onto it.
        fit = ~btv_frozen
        wts = fit.astype(np.float32)
        mean_flat = (wts @ images) / wts.sum()                           # in place / BLAS only: no copies of ~11 GB
        images -= mean_flat
        images[~fit] = 0                                                 # frozen shots do not enter the fit
        total_var = sum(float(np.dot(b, b)) for b in (images[i:i + 1024].ravel() for i in range(0, n, 1024)))
        _, S, Vt = randomized_svd(images, n_components=N_PCA, n_iter=5, random_state=0)
        scores = images @ Vt.T
        # a frozen shot repeats the previous image, so it gets the previous score (rows kept, flagged in btv_frozen)
        for i in np.flatnonzero(~fit):
            scores[i] = scores[i - 1] if i and seg[i] == seg[i - 1] else np.nan
        del images
        for k, pc in enumerate(PCS):
            cols[pc] = scores[:, k].astype(np.float64)
        pca_info = {"explained": [round(float(v), 4) for v in S**2 / total_var], "fit_shots": int(fit.sum())}
        print("PCA explained variance:", pca_info["explained"])

    # every reading onto its own pulse (pulses.py, as the forecasting tables); pulses counted within each chunk file
    trig = stamps["trigger"]
    pid = pulse_ids(trig, np.repeat(np.arange(len(files)), n_per_file))
    delay = stamps["pc_time"] - trig
    delay -= np.round(np.nanmedian(delay) / 3600) * 3600                 # in case pc_time is local time
    shift = {k: rf_shift(stamps[k] - trig) for k in ("PGLA_drop", "PKI15A", "PKI15P", "PKR15A")}
    shift["lab_laser_power"] = meter_shift(delay)
    shift["laser_amp"] = stamp_shift(stamps["laser_amp"] - trig)
    for k in GROUP_B:
        print(f"{k}: shift in pulses {dict(zip(*np.unique(shift[k], return_counts=True)))}")
        cols[k] = align(cols[k], pid, shift[k])
    laser_frozen = align(laser_frozen, pid, shift["laser_amp"]) != 0     # NaN (no reading) counts as frozen
    cam = stamp_shift(stamps["camera"] - trig)
    print(f"camera: shift in pulses {dict(zip(*np.unique(cam, return_counts=True)))}")
    camera_cols = [c for c in cols if c.startswith("BTV_") or c in SVDS or c in PCS]
    for c in camera_cols:
        cols[c] = align(cols[c], pid, cam)
    btv_frozen = align(btv_frozen, pid, cam) != 0
    complete = np.isfinite(np.column_stack([cols[k] for k in GROUP_B])).all(axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"features": FEATURES, "target": TARGET, "cuts_file": str(CUTS_FILE), "data_root": str(data_root),
                "pca": pca_info, "segments": []}
    for k, (r, ps) in enumerate(segs):
        msk = seg == k
        valid = complete[msk] & ~laser_frozen[msk] & ~btv_frozen[msk]
        path = OUT / f"segment_{k:02d}.npz"
        np.savez(path, X=np.column_stack([cols[f][msk] for f in FEATURES]), y=cols[TARGET][msk],
                 features=np.array(FEATURES), run=run[msk], chunk=chunk[msk], shot_in_chunk=shot_in_chunk[msk],
                 laser_frozen=laser_frozen[msk], btv_frozen=btv_frozen[msk], complete=complete[msk], valid=valid,
                 trigger_stamp=trig[msk])
        info = {"segment": k, "run": r, "first_chunk": int(chunk[msk][0]), "last_chunk": int(chunk[msk][-1]),
                "shots": int(msk.sum()), "valid": int(valid.sum()), "file": path.name}
        manifest["segments"].append(info)
        print(f"segment {k}: {r} chunks {info['first_chunk']}-{info['last_chunk']}, "
              f"{info['shots']} shots ({info['valid']} valid) -> {path}")
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"-> {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
