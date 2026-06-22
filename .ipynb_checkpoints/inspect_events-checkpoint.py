"""
inspect_events.py

Browse spindle/KC annotations on the raw EEG with a spindle-band filtered
overlay. Two rows per panel: broadband signal on top, 11–16 Hz bandpass below.
Annotated event is shaded; onset/offset marked with dashed lines.

Usage:
    python inspect_events.py --subj 01-02-0001
    python inspect_events.py --subj 01-02-0003 --event kc --pad 4
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_DIR  = Path("/Users/elizabethkaplan/Desktop/SS2_Results/New")
EEG_CH   = "C3"
SPINDLE_BAND = (11.0, 16.0)   # Hz

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subj",  default="01-02-0001", help="Subject ID, e.g. 01-02-0001")
parser.add_argument("--event", default="spindle",    choices=["spindle", "kc"],
                    help="Event type to browse: spindle or kc")
parser.add_argument("--pad",   default=3.0, type=float,
                    help="Seconds of context on each side of the event")
args = parser.parse_args()

SUBJ_ID    = args.subj
EVENT_TYPE = args.event
PAD_SEC    = args.pad

# ── load data ──────────────────────────────────────────────────────────────────
fif_path   = RAW_DIR / SUBJ_ID / "cleaned" / f"{SUBJ_ID}_cleaned_raw.fif"
annot_path = RAW_DIR / SUBJ_ID / "cleaned" / f"{SUBJ_ID}_annotations.csv"

print(f"Loading {fif_path.name} ...")
raw      = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
annot_df = pd.read_csv(annot_path)

keyword = "spindle" if EVENT_TYPE == "spindle" else "kcomplex"
events  = annot_df[annot_df["description"].str.lower().str.contains(keyword)].reset_index(drop=True)
label   = "Spindle" if EVENT_TYPE == "spindle" else "K-Complex"
color   = "#e07b54"  if EVENT_TYPE == "spindle" else "#2c3e8c"

print(f"Found {len(events)} {label} events.")

# pre-filter a copy for the spindle-band overlay
raw_filt = raw.copy().filter(SPINDLE_BAND[0], SPINDLE_BAND[1],
                              picks=EEG_CH, verbose=False)

sfreq   = raw.info["sfreq"]
ch_idx  = raw.ch_names.index(EEG_CH)

# ── state ──────────────────────────────────────────────────────────────────────
state = {"idx": 0}

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#f7f7f7")

ax_broad  = fig.add_axes([0.07, 0.52, 0.88, 0.38])   # broadband
ax_filt   = fig.add_axes([0.07, 0.13, 0.88, 0.34])   # spindle-band
ax_prev   = fig.add_axes([0.07, 0.03, 0.08, 0.06])
ax_next   = fig.add_axes([0.17, 0.03, 0.08, 0.06])

btn_prev = mwidgets.Button(ax_prev, "◀  Prev", color="#dde", hovercolor="#bbc")
btn_next = mwidgets.Button(ax_next, "Next  ▶", color="#dde", hovercolor="#bbc")

def draw():
    idx    = state["idx"]
    row    = events.iloc[idx]
    onset  = row["onset_s"]
    dur    = row["duration_s"]
    offset = onset + dur

    t0 = max(0.0, onset - PAD_SEC)
    t1 = min(raw.times[-1], offset + PAD_SEC)

    # extract both signals over the same window
    i0, i1  = int(t0 * sfreq), int(t1 * sfreq)
    times   = raw.times[i0:i1]
    broad   = raw._data[ch_idx, i0:i1]   * 1e6
    filt_   = raw_filt._data[ch_idx, i0:i1] * 1e6

    for ax in (ax_broad, ax_filt):
        ax.cla()
        ax.axvspan(onset, offset, color=color, alpha=0.22, zorder=1)
        ax.axvline(onset,  color="green", linewidth=1.4, linestyle="--",
                   label=f"Onset  {onset:.3f}s", zorder=3)
        ax.axvline(offset, color="red",   linewidth=1.4, linestyle="--",
                   label=f"Offset {offset:.3f}s", zorder=3)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.grid(True, alpha=0.15)
        ax.set_xlim(t0, t1)

    ax_broad.plot(times, broad, color="black", linewidth=0.7, zorder=2)
    ax_broad.set_ylabel("Broadband (µV)", fontsize=10)
    ax_broad.set_xticklabels([])
    ax_broad.legend(fontsize=8, loc="upper right", framealpha=0.8)

    ax_filt.plot(times, filt_, color="#3a7abf", linewidth=0.8, zorder=2)
    ax_filt.set_ylabel(f"{SPINDLE_BAND[0]}–{SPINDLE_BAND[1]} Hz (µV)", fontsize=10)
    ax_filt.set_xlabel("Time (s)", fontsize=10)

    fig.suptitle(
        f"{label}  |  {SUBJ_ID}  |  event {idx + 1} of {len(events)}"
        f"  |  duration = {dur:.3f}s",
        fontsize=12, fontweight="bold"
    )
    fig.canvas.draw_idle()

def on_prev(event):
    if state["idx"] > 0:
        state["idx"] -= 1
        draw()

def on_next(event):
    if state["idx"] < len(events) - 1:
        state["idx"] += 1
        draw()

def on_key(event):
    if event.key in ("right", "n"):
        on_next(None)
    elif event.key in ("left", "p"):
        on_prev(None)

btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
fig.canvas.mpl_connect("key_press_event", on_key)

draw()
plt.show()
