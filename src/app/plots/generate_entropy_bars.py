from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..analysis_methods import pixel_entropy_with_blocks

_THRESHOLD = 7.999

_ALG_COLORS = [
    "#1f77b4",  # AES-CTR  — blue
    "#2ca02c",  # AES-CBC  — green
    "#ff7f0e",  # 3DES     — orange
    "#9467bd",  # ChaCha20 — purple
    "#d62728",  # Logistic  — red
    "#8c564b",  # Henon    — brown
    "#e377c2",  # extra    — pink
]


def render_entropy_bars(
    encrypted_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Bar chart of overall Shannon entropy per algorithm for a single plain image."""
    alg_labels: list[str] = []
    entropy_values: list[float] = []

    for alg, p in encrypted_paths.items():
        if not p.exists():
            continue
        result = pixel_entropy_with_blocks(p)
        value = result.get("overall") or result.get("grayscale", 0.0)
        alg_labels.append(alg.replace("_", " "))
        entropy_values.append(float(value))

    if not alg_labels:
        return

    x = np.arange(len(alg_labels))
    colors = [_ALG_COLORS[i % len(_ALG_COLORS)] for i in range(len(alg_labels))]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(alg_labels)), 5))

    bars = ax.bar(x, entropy_values, color=colors, width=0.55, zorder=3)

    ax.axhline(_THRESHOLD, color="#E53935", linewidth=1.4, linestyle="--",
               label=f"Threshold ({_THRESHOLD})", zorder=4)

    # Value labels on top of each bar
    for bar, val in zip(bars, entropy_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(alg_labels, fontsize=10)
    ax.set_ylabel("Entropy (bits/pixel)", fontsize=10)
    ax.set_title("Shannon entropy — encrypted variants", fontsize=11)
    ax.set_ylim(min(entropy_values) - 0.05, 8.02)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
