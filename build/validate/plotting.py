#!/usr/bin/env python3

import os
from pathlib import Path

from conf import validation as conf

_mplconfig = conf.GENERATED_ROOT / ".mplconfig"
_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def apply_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": conf.PLOT_DPI,
        "savefig.dpi": conf.PLOT_DPI,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def save_figure(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def angle_tick_labels(angles):
    return [str(int(angle)) if float(angle).is_integer() else f"{angle:g}" for angle in angles]


def plot_heatmap(matrix, x_labels, y_labels, title, cbar_label, output_dir, name, cmap="viridis"):
    apply_style()
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(x_labels)), max(3.5, 0.45 * len(y_labels))))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    return save_figure(fig, output_dir, name)


def plot_lines(x, series, title, xlabel, ylabel, output_dir, name):
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, values in series.items():
        ax.plot(x, values, marker="o", linewidth=1.5, markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return save_figure(fig, output_dir, name)
