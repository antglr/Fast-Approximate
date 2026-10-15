# Plots of the virtual-diagnostic results (notebooks 08_04, 08_05): measured vs inferred BCM per segment.
# `pred` = one row per test shot with columns segment, run, shot, measured [nC] and one column per model [nC].
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K = 51        # rolling mean (shots) for the trend lines and the fluctuation
ZOOM = 150    # shots in the zoomed panel
BINS = 60
PC = 1000     # 1 nC = 1000 pC: differences are shown in pC

MEASURED, INFERRED, COMPARE, DIFF = "#2b2b29", "#eb6834", "#8a8985", "#2a78d6"
INK, GRID = "#2b2b29", "#e6e5e1"


def rolling(a):
    return pd.Series(a).rolling(K, center=True, min_periods=1).mean().to_numpy()


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def summary_table(pred, model):
    """Per segment: bias / spread / tails of the difference (inferred - measured) in pC, and shot-to-shot agreement."""
    def row(d):
        m, p = d["measured"].to_numpy(), d[model].to_numpy()
        e = PC * (p - m)
        fm, fp = m - rolling(m), p - rolling(p)
        return pd.Series({
            "shots": len(d),
            "mean measured [nC]": m.mean(),
            "mean inferred [nC]": p.mean(),
            "bias [pC]": e.mean(),
            "std diff [pC]": e.std(),
            "95% of |diff| below [pC]": np.percentile(np.abs(e), 95),
            "max |diff| [pC]": np.abs(e).max(),
            "std diff / mean": e.std() / (PC * m.mean()),
            "corr (shot by shot)": np.corrcoef(m, p)[0, 1],
            "corr of fluctuations": np.corrcoef(fm, fp)[0, 1],
            "fluctuation size (inferred / measured)": fp.std() / fm.std(),
        })
    return pred.groupby("segment").apply(row)


TABLE_FORMAT = {"shots": "{:.0f}", "mean measured [nC]": "{:.3f}", "mean inferred [nC]": "{:.3f}",
                "bias [pC]": "{:+.1f}", "std diff [pC]": "{:.1f}", "95% of |diff| below [pC]": "{:.1f}",
                "max |diff| [pC]": "{:.1f}", "std diff / mean": "{:.1%}", "corr (shot by shot)": "{:.2f}",
                "corr of fluctuations": "{:.2f}", "fluctuation size (inferred / measured)": "{:.2f}"}


def segment_figure(pred, k, model, compare=None, save_dir=None):
    """Time series (whole test period + zoom), difference in pC, and distributions for one segment.
    `compare`: optional second model column, drawn in grey for reference."""
    d = pred[pred["segment"] == k].reset_index(drop=True)
    shot, m, p = d["shot"].to_numpy(), d["measured"].to_numpy(), d[model].to_numpy()
    e = PC * (p - m)
    c = d[compare].to_numpy() if compare else None
    ec = PC * (c - m) if compare else None
    sd = e.std()

    fig = plt.figure(figsize=(15, 13), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[2.2, 2.2, 1.7, 2.3])
    title = (f"Segment {k} — {d['run'].iloc[0].removeprefix('acquisition_')} — {model}: corr {np.corrcoef(m, p)[0, 1]:.2f}, "
             f"difference std {sd:.1f} pC ({sd / (PC * m.mean()):.1%} of the mean charge)")
    if compare:
        title += f"  |  {compare}: {ec.std():.1f} pC"
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")

    # 1) whole test period: thin lines per shot, thick lines = 51-shot rolling mean
    ax = fig.add_subplot(gs[0, :])
    ax.plot(shot, m, color=MEASURED, lw=0.4, alpha=0.45)
    ax.plot(shot, p, color=INFERRED, lw=0.4, alpha=0.45)
    ax.plot(shot, rolling(m), color=MEASURED, lw=1.6, label="measured BCM")
    ax.plot(shot, rolling(p), color=INFERRED, lw=1.6, label=f"inferred ({model})")
    if compare:
        ax.plot(shot, rolling(c), color=COMPARE, lw=1.3, ls="--", label=f"{compare} (51-shot mean)")
    ax.axvspan(shot[0], shot[min(ZOOM, len(shot)) - 1], color=GRID, alpha=0.6, label="zoom below")
    lo, hi = np.percentile(np.r_[m, p], [0.5, 99.5])
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set(title="Whole test period (thin: every shot, thick: 51-shot mean)", ylabel="BCM_GUN [nC]")
    ax.legend(fontsize=8, loc="lower left", ncol=4)
    style(ax)

    # 2) zoom: shot by shot
    ax = fig.add_subplot(gs[1, :])
    z = slice(0, ZOOM)
    if compare:
        ax.plot(shot[z], c[z], "-", lw=0.9, color=COMPARE, label=compare)
    ax.plot(shot[z], m[z], "-o", ms=3, lw=0.9, color=MEASURED, label="measured BCM")
    ax.plot(shot[z], p[z], "-o", ms=3, lw=0.9, color=INFERRED, label=f"inferred ({model})")
    ax.set(title=f"First {ZOOM} test shots", ylabel="BCM_GUN [nC]")
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    style(ax)

    # 3) difference over time
    ax = fig.add_subplot(gs[2, :])
    ax.axhspan(e.mean() - sd, e.mean() + sd, color=DIFF, alpha=0.12, label=f"mean ± 1 std ({sd:.1f} pC)")
    ax.plot(shot, e, ".", ms=1.8, color=DIFF, alpha=0.5)
    ax.plot(shot, rolling(e), color=INK, lw=1.2, label="51-shot mean of the difference")
    if compare:
        ax.plot(shot, rolling(ec), color=COMPARE, lw=1.2, ls="--", label=f"51-shot mean, {compare}")
    ax.axhline(0, color=INK, lw=0.6)
    lim = np.percentile(np.abs(e), 99.5) * 1.1
    ax.set_ylim(-lim, lim)
    ax.set(title="Difference: inferred − measured", xlabel="shot in segment", ylabel="[pC]")
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    style(ax)

    # 4) distributions
    def hist_pair(ax, a, b, title, xlabel, unit):
        bins = np.histogram_bin_edges(np.r_[a, b], bins=BINS, range=tuple(np.percentile(np.r_[a, b], [0.5, 99.5])))
        ax.hist(a, bins=bins, color=MEASURED, alpha=0.35, label=f"measured: mean {a.mean():.3g}, std {a.std():.3g} {unit}")
        ax.hist(b, bins=bins, histtype="step", lw=1.8, color=INFERRED, label=f"inferred: mean {b.mean():.3g}, std {b.std():.3g} {unit}")
        ax.set(title=title, xlabel=xlabel, ylabel="shots")
        ax.legend(fontsize=7, loc="upper left")
        style(ax)

    hist_pair(fig.add_subplot(gs[3, 0]), m, p, "Charge values", "BCM_GUN [nC]", "nC")
    hist_pair(fig.add_subplot(gs[3, 1]), PC * (m - rolling(m)), PC * (p - rolling(p)), "Shot-to-shot fluctuation",
              "value − 51-shot mean [pC]", "pC")

    ax = fig.add_subplot(gs[3, 2])
    bins = np.linspace(-lim, lim, BINS)
    if compare:
        ax.hist(np.clip(ec, -lim, lim), bins=bins, histtype="step", lw=1.5, color=COMPARE, label=f"{compare}: std {ec.std():.1f}")
    ax.hist(e, bins=bins, color=DIFF, alpha=0.75, label=f"{model}: std {sd:.1f}")
    ax.axvline(0, color=INK, lw=0.8)
    ax.axvline(e.mean(), color=INK, lw=0.8, ls="--")
    ax.text(0.98, 0.95, f"mean {e.mean():+.1f} pC\nstd {sd:.1f} pC ({sd / (PC * m.mean()):.1%})\n"
                        f"95% within ±{np.percentile(np.abs(e), 95):.1f} pC", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK)
    ax.set(title="Difference: inferred − measured", xlabel="[pC]", ylabel="shots")
    if compare:
        ax.legend(fontsize=7, loc="upper left")
    style(ax)

    if save_dir is not None:
        fig.savefig(save_dir / f"segment_{k:02d}.png", dpi=120, bbox_inches="tight")
    plt.show()


def difference_all_segments(pred, model, compare=None, save_path=None):
    """Difference distributions of all segments in pC, each panel with its own range."""
    segments = sorted(pred["segment"].unique())
    fig, axes = plt.subplots(1, len(segments), figsize=(2.9 * len(segments), 3.9), layout="constrained")
    for ax, k in zip(axes, segments):
        d = pred[pred["segment"] == k]
        e = PC * (d[model] - d["measured"])
        ec = PC * (d[compare] - d["measured"]) if compare else None
        lim = np.percentile(np.abs(e if ec is None else np.r_[e, ec]), 99.5) * 1.05
        bins = np.linspace(-lim, lim, 61)
        if compare:
            ax.hist(ec.clip(-lim, lim), bins=bins, color=COMPARE, alpha=0.45, label=compare)
        ax.hist(e.clip(-lim, lim), bins=bins, color=DIFF, alpha=0.75 if not compare else 1.0,
                histtype="bar" if not compare else "step", lw=1.8, label=model)
        ax.axvline(0, color=INK, lw=0.8)
        title = f"seg {k} ({d['measured'].mean():.2f} nC)\nstd {e.std():.1f} pC ({e.std() / (PC * d['measured'].mean()):.1%})"
        if compare:
            title += f"\n{compare}: {ec.std():.1f} pC"
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.supxlabel("inferred − measured [pC]  (each panel has its own range: the charge differs by ~100× between segments; "
                  "the outer 0.5% is piled up at the edges)", fontsize=9)
    fig.suptitle(f"Difference distributions, {model}", fontsize=11)
    if compare:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", fontsize=8, ncol=2)
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
