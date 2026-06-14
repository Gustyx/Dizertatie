from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .plot_generation_helper import _load_preview

_DIRECTIONS = ["Horizontal", "Vertical", "Diagonal"]

_ALG_COLORS = [
    "#00bcd4",  # original — cyan (matches thesis)
    "#4caf50",  # green
    "#ff9800",  # orange
    "#e91e63",  # pink
    "#9c27b0",  # purple
    "#f44336",  # red
    "#795548",  # brown
]

_BG_COLOR = "#0d1117"


def _luminance(img: np.ndarray) -> np.ndarray:
    return (
        0.2989 * img[..., 0].astype(np.float32)
        + 0.5870 * img[..., 1].astype(np.float32)
        + 0.1140 * img[..., 2].astype(np.float32)
    )


def _adjacent_pairs(channel: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    if direction == "Horizontal":
        return channel[:, :-1].ravel(), channel[:, 1:].ravel()
    elif direction == "Vertical":
        return channel[:-1, :].ravel(), channel[1:, :].ravel()
    else:
        return channel[:-1, :-1].ravel(), channel[1:, 1:].ravel()


def render_scatter_plot(
    plain_path: Path,
    encrypted_paths: dict[str, Path],
    output_path: Path,
) -> None:
    images: list[tuple[str, np.ndarray]] = [("Original", _load_preview(plain_path))]
    for alg, p in encrypted_paths.items():
        if p.exists():
            images.append((alg.replace("_", " "), _load_preview(p)))

    n_rows = len(images)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor(_BG_COLOR)

    for row, (label, img) in enumerate(images):
        title_color = _ALG_COLORS[row % len(_ALG_COLORS)]
        lum = _luminance(img)

        for col, direction in enumerate(_DIRECTIONS):
            ax = axes[row][col]
            ax.set_facecolor(_BG_COLOR)

            xi, xi1 = _adjacent_pairs(lum, direction)

            h, xedges, yedges = np.histogram2d(xi, xi1, bins=256,
                                               range=[[0, 255], [0, 255]])
            h = np.log1p(h)
            ax.imshow(
                h.T, origin="lower", aspect="auto",
                extent=[0, 255, 0, 255],
                cmap="inferno",
                interpolation="nearest",
            )

            ax.set_title(f"{label} -- {direction}", color=title_color,
                         fontsize=10, fontweight="bold", pad=4)
            ax.set_xlabel("Pixel xᵢ", color="white", fontsize=8)
            ax.set_ylabel("Pixel xᵢ₊₁", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

    plt.tight_layout(pad=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)
