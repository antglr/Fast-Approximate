# 01 — Explore the data (script version of 01_explore_data.ipynb).
# Lists every variable in the HDF5 chunks, checks the layout of all chunks, loads the scalars and plots one shot.
# Tables are printed and saved as CSV, figures saved as PNG.
# Usage: python 01_explore_data.py [data dir] [out dir]   (defaults: nearest data/ folder, results/01_explore_data)
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

HERE = Path(__file__).resolve().parent
SHOT = 0


def find_data_dir():
    """Nearest data/ folder above the current directory or the script directory."""
    for start in (Path.cwd(), HERE):
        for p in [start, *start.parents]:
            if (p / "data").is_dir():
                return p / "data"
    sys.exit("no data/ folder found, pass it as the first argument")


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {path}")


def decode(x):
    return x.decode() if isinstance(x, bytes) else x


# ---------------------------------------------------------------------------
# Variable inventory
# ---------------------------------------------------------------------------
def inventory(path):
    rows = []
    with h5py.File(path, "r") as f:
        devices = dict(f["parameters"].attrs)

        def visit(name, obj):
            ragged = isinstance(obj, h5py.Group) and obj.attrs.get("ragged", False)
            parent = name.rsplit("/", 1)[0] if "/" in name else ""
            if not (isinstance(obj, h5py.Dataset) or ragged) or (parent and f[parent].attrs.get("ragged", False)):
                return
            top = name.split("/")[0]
            if ragged:  # variable-length per shot, stored as one dataset per shot
                lengths = {obj[k].shape for k in obj}
                rows.append({
                    "variable": name,
                    "group": top,
                    "shape": (len(obj), "ragged"),
                    "per_shot_shape": max(lengths),
                    "dtype": str(obj[next(iter(obj))].dtype),
                    "example (shot 0)": f"ragged, per-shot shapes {sorted(lengths)}",
                    "device": devices.get(top, ""),
                })
                return
            sample = obj[0] if obj.shape else obj[()]
            if obj.dtype.kind in "OS":
                example = decode(sample if np.ndim(sample) == 0 else sample.flat[0])
            elif np.ndim(sample) == 0:
                example = sample
            else:
                example = f"array{np.shape(sample)}"
            rows.append({
                "variable": name,
                "group": top if "/" in name else "",
                "shape": obj.shape,
                "per_shot_shape": obj.shape[1:],
                "dtype": str(obj.dtype) if obj.dtype.kind != "O" else "str",
                "example (shot 0)": example,
                "device": devices.get(top.removesuffix("_stamp"), ""),
            })

        f.visititems(visit)
    return pd.DataFrame(rows)


def variable_names(f):
    names = set()

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and obj.attrs.get("ragged", False):
            names.add(name)
        elif isinstance(obj, h5py.Dataset) and not f[name.rsplit("/", 1)[0]].attrs.get("ragged", False):
            names.add(name)

    f.visititems(visit)
    return names


def check_layout(files, ref):
    """Per chunk: variables missing/extra compared to the first chunk, and ragged variables with empty shots."""
    layout = []
    for path in files:
        with h5py.File(path, "r") as f:
            names = variable_names(f)
            ragged = {}
            for n in names:
                if isinstance(f[n], h5py.Group):
                    empty = sum(f[n][k].size == 0 for k in f[n])
                    ragged[n] = f"{empty} empty shots"
            layout.append({
                "file": path.name,
                "n_shots": f.attrs["n_shots"],
                "missing": sorted(ref - names),
                "extra": sorted(names - ref),
                "ragged": ragged,
            })
    return pd.DataFrame(layout)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scalars(paths, variables):
    frames = []
    for path in paths:
        with h5py.File(path, "r") as f:
            df = pd.DataFrame({v: f[v][:] for v in variables})
            df["chunk"] = path.stem
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def one_shot(obj, i):
    """Shot i of a variable, whether stored as one array or as a ragged group of per-shot datasets."""
    return obj[i] if isinstance(obj, h5py.Dataset) else obj[sorted(obj)[i]][:]


def read_variable(f, name):
    obj = f[name]
    if isinstance(obj, h5py.Dataset):
        return obj[:]
    shots = [obj[k][:] for k in sorted(obj)]
    width = max(len(a) for a in shots)
    out = np.full((len(shots), width), np.nan)
    for i, a in enumerate(shots):
        out[i, : len(a)] = a
    return out


def load_variable(name, paths):
    """Concatenate a variable over all chunks; ragged variables are NaN-padded. Beware of BTV0125/image2D (~GBs)."""
    arrays = []
    for path in paths:
        with h5py.File(path, "r") as f:
            arrays.append(read_variable(f, name))
    return np.concatenate(arrays)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_scalars(scalars, names):
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.6 * len(names)), sharex=True)
    for ax, v in zip(axes, names):
        ax.plot(scalars["time"], scalars[v], ".", ms=2)
        ax.set_ylabel(v, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    save(fig, "scalars_vs_time.png")


def plot_waveforms(path, shot):
    with h5py.File(path, "r") as f:
        waveforms = {
            "PGIA": f["PGIA"][shot, 0],
            "PGIP": f["PGIP"][shot, 0],
            "PGRA": f["PGRA"][shot, 0],
            "PGLA/samples": f["PGLA/samples"][shot, 0],
            "PKI15A/samples": f["PKI15A/samples"][shot, 0],
            "PKI15P": f["PKI15P"][shot, 0],
            "PKR15A": f["PKR15A"][shot, 0],
        }
        scope_t = f["laser_scope/times"][shot]
        scope_v = f["laser_scope/signal"][shot]
        scope_units = decode(f["laser_scope/times_unit"][shot]), decode(f["laser_scope/signal_unit"][shot])

    fig, axes = plt.subplots(len(waveforms) + 1, 1, figsize=(12, 1.8 * (len(waveforms) + 1)))
    for ax, (name, y) in zip(axes, waveforms.items()):
        ax.plot(y, lw=0.8)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    axes[-2].set_xlabel("sample")
    axes[-1].plot(scope_t, scope_v, lw=0.8)
    axes[-1].set_ylabel("laser_scope", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel(f"time [{scope_units[0]}]  /  signal [{scope_units[1]}]")
    fig.tight_layout()
    save(fig, f"waveforms_shot{shot}.png")


def plot_btv(path, shot):
    with h5py.File(path, "r") as f:
        btv = f["BTV0125"]
        img = btv["image2D"][shot]
        x = one_shot(btv["imagePositionSet1"], shot)
        y = one_shot(btv["imagePositionSet2"], shot)
        unit = decode(btv["imagePositionSet1_unit"][shot])
        proj_x = one_shot(btv["projDataSet1"], shot)
        proj_y = one_shot(btv["projDataSet2"], shot)

    print("image", img.shape, img.dtype, "min/max:", img.min(), img.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [2, 1, 1]})
    im = axes[0].imshow(img, extent=[x[0], x[-1], y[-1], y[0]], aspect="auto", cmap="viridis")
    axes[0].set(title=f"BTV0125 image2D, shot {shot}", xlabel=f"x [{unit}]", ylabel=f"y [{unit}]")
    fig.colorbar(im, ax=axes[0])
    axes[1].plot(x, proj_x); axes[1].set(title="projDataSet1 (x)", xlabel=f"x [{unit}]")
    axes[2].plot(y, proj_y); axes[2].set(title="projDataSet2 (y)", xlabel=f"y [{unit}]")
    fig.tight_layout()
    save(fig, f"btv0125_shot{shot}.png")


# ---------------------------------------------------------------------------
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else find_data_dir()
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results" / "01_explore_data"

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("chunk_*.h5"))
    if not files:
        sys.exit(f"no chunk_*.h5 in {DATA_DIR}")
    print(f"{len(files)} files in {DATA_DIR}")
    print(f"Total size: {sum(f.stat().st_size for f in files) / 1e9:.2f} GB")

    section("File-level metadata")
    with h5py.File(files[0], "r") as f:
        for k, v in f.attrs.items():
            print(f"{k:15s} {v}")

    section("Variable inventory")
    inv = inventory(files[0])
    print(f"{len(inv)} datasets")
    print(inv.to_string())
    inv.to_csv(OUT_DIR / "inventory.csv", index=False)

    section("Device mapping (parameters attributes)")
    with h5py.File(files[0], "r") as f:
        devices = pd.Series(dict(f["parameters"].attrs), name="device / property | cycle")
    print(devices.to_string())

    section("Variables grouped by kind")
    numeric = inv[inv["dtype"] != "str"]
    kind = np.select(
        [numeric["per_shot_shape"].map(len) == 0, numeric["variable"].str.endswith("image2D")],
        ["scalar", "image"],
        default="waveform",
    )
    numeric = numeric.assign(kind=kind)
    for k, df in numeric.groupby("kind"):
        print(f"\n=== {k} ({len(df)}) ===")
        print(df[["variable", "per_shot_shape", "dtype"]].to_string(index=False))

    section("Layout of all chunks")
    layout = check_layout(files, set(inv["variable"]))
    print(layout.to_string())
    layout.to_csv(OUT_DIR / "layout.csv", index=False)

    section("Scalar variables")
    scalar_vars = numeric.loc[numeric["kind"] == "scalar", "variable"].tolist()
    scalars = load_scalars(files, scalar_vars)
    n_inf = np.isinf(scalars.select_dtypes("number")).sum()
    print("inf values replaced by NaN:", n_inf[n_inf > 0].to_dict())
    scalars = scalars.replace([np.inf, -np.inf], np.nan)
    scalars["time"] = pd.to_datetime(scalars["pc_time"], unit="s")
    print(scalars.shape, "| from", scalars["time"].min(), "to", scalars["time"].max())
    print(scalars.head().to_string())

    # top-level scalar signals, without timestamp/bookkeeping columns
    main_scalars = [v for v in scalar_vars if "/" not in v and not v.endswith("_stamp")
                    and v not in ("pc_time", "trigger_stamp", "shot")]
    summary = scalars[main_scalars].describe().T
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "scalar_summary.csv")
    plot_scalars(scalars, main_scalars)

    section(f"One shot ({SHOT}): waveforms and camera image")
    plot_waveforms(files[0], SHOT)
    plot_btv(files[0], SHOT)

    section("load_variable example")
    pgia = load_variable("PGIA", files)
    btv_x = load_variable("BTV0125/imagePositionSet1", files)
    print("PGIA", pgia.shape)
    print("BTV0125/imagePositionSet1", btv_x.shape, "| shots with no x-axis:", np.isnan(btv_x).all(axis=1).sum())
