from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_ALG_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def render_speed_plot(
    encryption_times: dict[str, float],
    output_path: Path,
) -> None:
    items = [(alg, t) for alg, t in encryption_times.items() if t > 0]
    if not items:
        return

    items.sort(key=lambda x: x[1])
    labels = [alg.replace("_", " ") for alg, _ in items]
    times = [t for _, t in items]

    colors = [_ALG_COLORS[i % len(_ALG_COLORS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.7 * len(labels) + 1.5)))

    bars = ax.barh(labels, times, color=colors, height=0.55, zorder=3)

    for bar, val in zip(bars, times):
        ax.text(
            val + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} s",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Encryption time (seconds)", fontsize=10)
    ax.set_title("Encryption speed comparison", fontsize=11)
    ax.set_xlim(0, max(times) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
