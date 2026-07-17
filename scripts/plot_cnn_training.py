#!/usr/bin/env python
"""
plot_cnn_training.py

Training diagnostics for the from-scratch SimpleCNN COCO classifier
(`psm_final.models.cnn_basemodel`), with the ResNet-50 arm for reference.

Data sources (all produced by the training scripts):
  - results/cnn_basemodel/train_history_simple_cnn.csv  -> one column, per-BATCH
    training cross-entropy for the final all-data model (15,850 rows = 25 * 634).
  - results/cnn_basemodel/simple_cnn_val_history.csv     -> per-EPOCH val loss + acc
    for the SimpleCNN (written by cnn_basemodel.main() once the shared train_and_test
    helper started returning the per-epoch val history). Absent until the model is
    retrained -- the plot then falls back to the FINAL_VAL_LOSS constant below.
  - results/cnn_resnet50/resnet50_val_history.csv        -> per-EPOCH val loss + acc.

Note on accuracy: top-1/top-5 come from re-evaluating results/cnn_basemodel/cnn.pth on
all 40,137 annotated val-2014 images -- they are constants here (FINAL_TOP1/5), not
read from a file; rerun that eval if you retrain. The per-epoch validation LOSS curve,
by contrast, is now read from simple_cnn_val_history.csv when present.

Usage:  python scripts/plot_cnn_training.py [--show]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SIMPLE_CSV = ROOT / "results/cnn_basemodel/train_history_simple_cnn.csv"
SIMPLE_VAL_CSV = ROOT / "results/cnn_basemodel/simple_cnn_val_history.csv"
RESNET_CSV = ROOT / "results/cnn_resnet50/resnet50_val_history.csv"
OUT_PNG = ROOT / "results/cnn_basemodel/cnn_training_matplotlib.png"

# --- config / known constants ----------------------------------------------
EPOCHS = 25                       # SimpleCNN run length (batches must divide evenly)
FINAL_VAL_LOSS = 2.1348           # fallback only: re-eval of cnn.pth on val-2014,
                                  # used when simple_cnn_val_history.csv is absent
FINAL_TOP1 = 45.31                # top-1 %  (chance = 100/80 = 1.25 %)
FINAL_TOP5 = 74.29                # top-5 %
NUM_CLASSES = 80

# colourblind-safe, print-friendly, matching the HTML report. Blue = SimpleCNN train,
# orange = SimpleCNN validation, green = ResNet-50 (its own panels). Hues are the
# validated categorical slots (blue/orange/green), distinct under common CVD.
BLUE, BLUE_SOFT = "#2a78d6", "#a9c9f2"
ORANGE = "#eb6834"
GREEN, MUTED, GRID = "#008300", "#8a8a8a", "#e2e2dd"


def smooth(x: np.ndarray, w: int = 64) -> np.ndarray:
    """Centred moving average for the faint per-batch trace."""
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def load_simplecnn():
    loss = pd.read_csv(SIMPLE_CSV)["batch_loss"].to_numpy()
    n = loss.size
    if n % EPOCHS != 0:
        raise ValueError(f"{n} batches not divisible by {EPOCHS} epochs")
    bpe = n // EPOCHS
    epoch_mean = loss.reshape(EPOCHS, bpe).mean(axis=1)
    return loss, epoch_mean, bpe


def load_simplecnn_val():
    """Per-epoch validation curve written by cnn_basemodel.main(), or None if the
    model hasn't been retrained since per-epoch val logging was added."""
    if SIMPLE_VAL_CSV.exists():
        return pd.read_csv(SIMPLE_VAL_CSV)
    return None


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.margins(x=0.01)


def main(show: bool = False):
    loss, epoch_mean, bpe = load_simplecnn()
    sv = load_simplecnn_val()          # per-epoch val curve, or None (pre-retrain)
    rn = pd.read_csv(RESNET_CSV)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
    })

    fig = plt.figure(figsize=(13, 5.2), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1],
                          wspace=0.22, hspace=0.55)
    ax_loss = fig.add_subplot(gs[:, 0])   # big left panel
    ax_rl = fig.add_subplot(gs[0, 1])     # resnet loss
    ax_ra = fig.add_subplot(gs[1, 1])     # resnet accuracy

    # ----- Panel A: SimpleCNN training vs. validation loss -----------------
    sm = smooth(loss)
    x_sm = (np.arange(sm.size) + 32) / bpe        # batch index -> fractional epoch
    x_ep = np.arange(1, EPOCHS + 1)
    ax_loss.plot(x_sm, sm, color=BLUE_SOFT, lw=1.1, zorder=2, label="Train (per-batch, smoothed)")
    ax_loss.plot(x_ep, epoch_mean, color=BLUE, lw=2.6, zorder=4, label="Train (per-epoch mean)")

    # endpoint marker + label on the training curve
    ax_loss.scatter([EPOCHS], [epoch_mean[-1]], s=42, color=BLUE, zorder=5,
                    edgecolor="white", linewidth=1.5)
    ax_loss.annotate(f"{epoch_mean[-1]:.2f}", (EPOCHS, epoch_mean[-1]),
                     textcoords="offset points", xytext=(8, 2),
                     color=BLUE, fontweight="bold", fontsize=10)

    if sv is not None:
        # Real per-epoch validation curve (gap above the train curve = generalisation
        # gap; an upturn = overfitting), the point of logging val loss over time.
        ax_loss.plot(sv["epoch"], sv["val_loss"], color=ORANGE, lw=2.4, marker="o",
                     ms=4, zorder=6, label="Validation (per-epoch)")
        v_last_x, v_last_y = int(sv["epoch"].iloc[-1]), float(sv["val_loss"].iloc[-1])
        ax_loss.scatter([v_last_x], [v_last_y], s=42, color=ORANGE, zorder=7,
                        edgecolor="white", linewidth=1.5)
        ax_loss.annotate(f"{v_last_y:.2f}", (v_last_x, v_last_y),
                         textcoords="offset points", xytext=(8, 2),
                         color=ORANGE, fontweight="bold", fontsize=10)
        title = "SimpleCNN — training vs. validation loss"
    else:
        # Fallback until the model is retrained with per-epoch val logging: the single
        # final val-loss constant, drawn as a reference line (the old behaviour).
        ax_loss.axhline(FINAL_VAL_LOSS, color=MUTED, ls=(0, (4, 4)), lw=1.3, zorder=3)
        ax_loss.text(EPOCHS, FINAL_VAL_LOSS + 0.03,
                     f"final val loss {FINAL_VAL_LOSS:.2f} (no per-epoch history yet)",
                     ha="right", va="bottom", color=MUTED, fontsize=8.5)
        title = "SimpleCNN — training loss"

    ax_loss.set_title(title, loc="left")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_xlim(0.5, EPOCHS + 1.4)
    ax_loss.xaxis.set_major_locator(MultipleLocator(5))
    ax_loss.legend(frameon=False, fontsize=8.5, loc="upper right")
    style_axes(ax_loss)

    # ----- Panel B: ResNet-50 val loss -------------------------------------
    ax_rl.plot(rn["epoch"], rn["val_loss"], color=GREEN, lw=2.2, marker="o", ms=3)
    ax_rl.set_title("ResNet-50 — val loss (overfits)", loc="left")
    ax_rl.set_ylabel("Val loss")
    ax_rl.tick_params(labelbottom=False)
    style_axes(ax_rl)

    # ----- Panel C: ResNet-50 val accuracy ---------------------------------
    ax_ra.plot(rn["epoch"], rn["val_accuracy_pct"], color=GREEN, lw=2.2, marker="o", ms=3)
    peak_i = int(rn["val_accuracy_pct"].idxmax())
    ax_ra.scatter([rn["epoch"][peak_i]], [rn["val_accuracy_pct"][peak_i]], s=42,
                  color=GREEN, edgecolor="white", linewidth=1.5, zorder=5)
    ax_ra.annotate(f"peak {rn['val_accuracy_pct'][peak_i]:.1f}%",
                   (rn["epoch"][peak_i], rn["val_accuracy_pct"][peak_i]),
                   textcoords="offset points", xytext=(8, -12), color=GREEN, fontsize=8.5)
    ax_ra.set_title("ResNet-50 — val accuracy", loc="left")
    ax_ra.set_xlabel("Epoch")
    ax_ra.set_ylabel("Val acc %")
    style_axes(ax_ra)

    # ----- figure caption with the SimpleCNN's (unlogged) final metrics ----
    fig.suptitle("COCO object classifier — training diagnostics", x=0.012, ha="left",
                 fontsize=13.5, fontweight="bold", y=0.99)
    val_note = ("validation loss logged per epoch; top-1/5 from the saved checkpoint"
                if sv is not None else
                "per-epoch val loss not logged yet — retrain to populate the curve")
    fig.text(0.012, 0.925,
             f"SimpleCNN · 80-way · 64 px · {EPOCHS} epochs · Adam 1e-3   |   "
             f"final val: top-1 {FINAL_TOP1:.1f}%  ·  top-5 {FINAL_TOP5:.1f}%  "
             f"(chance {100/NUM_CLASSES:.2f}%)  —  {val_note}",
             ha="left", fontsize=9, color=MUTED)

    fig.subplots_adjust(top=0.86, left=0.062, right=0.98, bottom=0.1)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    if show:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="open an interactive window")
    main(**vars(p.parse_args()))
