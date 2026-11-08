"""
Step 2: Signal inspection plots (Methods section 3.4).

For each subject, plot the four wrist signals over time and shade the
baseline and stress periods. Use the figures to look for signal loss,
abnormal jumps, and the physiological response during stress.

Run from the repo root:
    python src/plot_signals.py          # all 15 subjects
    python src/plot_signals.py 2 3      # only S2 and S3

Figures are saved to results/figures/signals/S{id}.png
"""

import sys

import matplotlib
matplotlib.use("Agg")  # save to file without opening windows
import matplotlib.pyplot as plt
import numpy as np

from load_data import FS, REPO_ROOT, SUBJECTS, load_aligned

OUT_DIR = REPO_ROOT / "results" / "figures" / "signals"

# Colors for shaded condition periods
SHADE = {
    1: ("baseline", "tab:blue"),
    2: ("stress", "tab:red"),
    3: ("amusement", "tab:green"),
    4: ("meditation", "tab:purple"),
}


def find_segments(lab, fs):
    """Find continuous runs of the same label.

    Returns a list of (code, start_min, end_min).
    """
    change = np.flatnonzero(np.diff(lab)) + 1          # where the label changes
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    return [(int(lab[s]), s / fs / 60, e / fs / 60) for s, e in zip(starts, ends)]


def plot_subject(sid):
    signals, labels = load_aligned(sid)

    # ACC: raw units -> g (divide by 64), then three-axis magnitude
    acc_g = signals["ACC"] / 64.0
    acc_mag = np.linalg.norm(acc_g, axis=1)

    panels = [
        ("BVP", signals["BVP"], FS["BVP"], "BVP (a.u.)"),
        ("EDA", signals["EDA"], FS["EDA"], "EDA (μS)"),
        ("TEMP", signals["TEMP"], FS["TEMP"], "TEMP (°C)"),
        ("ACC", acc_mag, FS["ACC"], "|ACC| (g)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    segments = find_segments(labels["EDA"], FS["EDA"])

    for ax, (name, x, fs, ylabel) in zip(axes, panels):
        t = np.arange(len(x)) / fs / 60  # time in minutes
        ax.plot(t, x, lw=0.4, color="0.2", zorder=1)
        ax.set_ylabel(ylabel)

        # Shade each condition period. zorder=2 draws the shading on top of
        # the line, so dense signals like BVP do not hide the colors.
        for code, t0, t1 in segments:
            if code in SHADE:
                ax.axvspan(t0, t1, color=SHADE[code][1], alpha=0.2, lw=0, zorder=2)

        # BVP has large spikes; limit the y range to the 0.5 to 99.5 percentile
        if name == "BVP":
            lo, hi = np.percentile(x, [0.5, 99.5])
            ax.set_ylim(lo, hi)

    # Legend for the shaded colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for _, c in SHADE.values()]
    axes[0].legend(handles, [n for n, _ in SHADE.values()],
                   loc="upper right", ncol=4, fontsize=9)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"S{sid}: wrist signals", fontsize=14)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"S{sid}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or SUBJECTS
    for sid in ids:
        plot_subject(sid)
