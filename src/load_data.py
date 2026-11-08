"""
Step 1: Load WESAD wrist (Empatica E4) data and align labels.

Methods section 3.3:
  Labels are recorded at 700 Hz. For a wrist signal sampled at f Hz,
  the i-th sample occurs at i/f seconds, so its label index is floor(700 * i / f).

Run from the repo root:
    python src/load_data.py
"""

import pickle
from pathlib import Path

import numpy as np

# Repo root = the folder that contains src/ and data/
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "WESAD"

# 15 subjects. S1 and S12 were excluded by the original authors.
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Sampling rates of the wrist signals (Hz)
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
LABEL_FS = 700

# Label codes
LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load_subject(sid):
    """Read one subject's pickle file.

    Returns
    -------
    signals : dict
        "BVP", "EDA", "TEMP" -> 1D arrays; "ACC" -> array of shape (n, 3)
    label : 1D int array at 700 Hz
    """
    path = DATA_DIR / f"S{sid}" / f"S{sid}.pkl"
    with open(path, "rb") as f:
        # The files were saved with Python 2, so latin1 encoding is needed
        raw = pickle.load(f, encoding="latin1")

    wrist = raw["signal"]["wrist"]
    signals = {}
    for name in FS:
        x = np.asarray(wrist[name], dtype=float)
        if name != "ACC":
            x = x.ravel()  # (n, 1) -> (n,)
        signals[name] = x

    label = np.asarray(raw["label"]).ravel().astype(int)
    return signals, label


def align_labels(n_samples, fs, label):
    """Give each wrist sample the label at the same time point.

    Sample i is at time i / fs seconds.
    The matching label index is floor(700 * i / fs).
    Integer arithmetic is used, so there is no rounding error.
    """
    idx = (np.arange(n_samples) * LABEL_FS) // fs
    n_out = int(np.sum(idx >= len(label)))  # samples that fall past the end of the labels
    idx = np.minimum(idx, len(label) - 1)
    return label[idx], n_out


def load_aligned(sid):
    """Load one subject and return signals plus one label array per signal."""
    signals, label = load_subject(sid)
    labels = {}
    for name, x in signals.items():
        labels[name], _ = align_labels(len(x), FS[name], label)
    return signals, labels


def check_subject(sid):
    """Print checks for one subject. Used to make sure loading and alignment are correct."""
    signals, label = load_subject(sid)
    label_dur = len(label) / LABEL_FS

    print(f"\n===== S{sid} =====")
    print(f"label: {len(label)} samples, {label_dur:.1f} s")

    for name, x in signals.items():
        dur = len(x) / FS[name]
        aligned, n_out = align_labels(len(x), FS[name], label)
        print(f"{name:5s}: shape {str(x.shape):12s} {dur:8.1f} s  "
              f"diff vs label {dur - label_dur:+6.2f} s  "
              f"samples past label end: {n_out}  NaN: {int(np.isnan(x).sum())}")

    # Minutes in each condition, measured on the EDA label sequence (4 Hz)
    eda_lab, _ = align_labels(len(signals["EDA"]), FS["EDA"], label)
    parts = []
    for code in [1, 2, 3, 4]:
        minutes = np.sum(eda_lab == code) / FS["EDA"] / 60
        parts.append(f"{LABEL_NAMES[code]} {minutes:.1f} min")
    print("conditions: " + ", ".join(parts))


if __name__ == "__main__":
    for sid in SUBJECTS:
        check_subject(sid)
