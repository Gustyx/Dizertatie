from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..analysis_methods import (
    horizontal_pixel_correlation,
    vertical_pixel_correlation,
    diagonal_pixel_correlation,
)

_THRESHOLD = 0.01

_ALG_COLORS = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#d62728",  # red
    "#8c564b",  # brown
    "#e377c2",  # pink
]

_DIRECTIONS = [
    ("Horizontal", horizontal_pixel_correlation),
    ("Vertical",   vertical_pixel_correlation),
    ("Diagonal",   diagonal_pixel_correlation),
]


def render_correlation_plot(
    encrypted_paths: dict[str, Path],
    output_path: Path,
) -> None:
    alg_labels: list[str] = []
    dir_values: dict[str, list[float]] = {d: [] for d, _ in _DIRECTIONS}

    for alg, p in encrypted_paths.items():
        if not p.exists():
            continue
        alg_labels.append(alg.replace("_", " "))
        for dir_name, fn in _DIRECTIONS:
            result = fn(p)
            dir_values[dir_name].append(abs(float(result["overall"])))

    if not alg_labels:
        return

    n_algs = len(alg_labels)
    n_dirs = len(_DIRECTIONS)
    group_width = 0.8
    bar_width = group_width / n_algs
    dir_positions = np.arange(n_dirs)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_dirs), 5))

    for i, (alg_label, color) in enumerate(zip(alg_labels, _ALG_COLORS)):
        offsets = dir_positions + (i - n_algs / 2 + 0.5) * bar_width
        values = [dir_values[d][i] for d, _ in _DIRECTIONS]
        ax.bar(offsets, values, width=bar_width * 0.92, color=color,
               label=alg_label, zorder=3)

    ax.axhline(_THRESHOLD, color="#E53935", linewidth=1.4, linestyle="--",
               label=f"Threshold ({_THRESHOLD})", zorder=4)

    ax.set_xticks(dir_positions)
    ax.set_xticklabels([d for d, _ in _DIRECTIONS], fontsize=11)
    ax.set_ylabel("Absolute Pearson r", fontsize=10)
    ax.set_title("Pixel correlation — encrypted variants", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
