from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_CH_COLORS = ["#E53935", "#43A047", "#1E88E5"]
_CH_NAMES = ["R", "G", "B"]


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def _plot_rgb_histogram(ax: "plt.Axes", img: np.ndarray, title: str) -> None:
    for c, (col, name) in enumerate(zip(_CH_COLORS, _CH_NAMES)):
        hist, bins = np.histogram(img[..., c].ravel(), bins=256, range=(0, 255))
        ax.plot(bins[:-1], hist, color=col, alpha=0.75, linewidth=0.8, label=name)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Pixel intensity", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_histogram_all_algorithms(
    plain_path: Path,
    encrypted_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """1×(1+N) plot: one panel for plain image, one panel per encrypted algorithm."""
    plain = _load_rgb(plain_path)

    enc_images: list[tuple[str, np.ndarray]] = []
    for alg, p in encrypted_paths.items():
        if p.exists():
            enc_images.append((alg, _load_rgb(p)))

    all_panels: list[tuple[np.ndarray, str]] = [
        (plain, f"Original image ({plain_path.stem})")
    ] + [(img, alg.replace("_", " ").upper() + " encrypted") for alg, img in enc_images]

    n_total = len(all_panels)
    n_cols = min(n_total, 3)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharey=False,
        squeeze=False,
    )

    for i, (img, title) in enumerate(all_panels):
        ax = axes[i // n_cols][i % n_cols]
        _plot_rgb_histogram(ax, img, title)

    # Hide any unused axes in the last row
    for j in range(n_total, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
