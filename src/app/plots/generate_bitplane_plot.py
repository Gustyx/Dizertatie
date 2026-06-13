from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .plot_generation_helper import _load_preview

_CH_COLORS = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}


def _bitplane_percentages(img: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img.shape[0], img.shape[1]
    total = float(h * w)
    out = {}
    for i, name in enumerate(("R", "G", "B")):
        ch = img[..., i].astype(np.uint8)
        planes = np.empty(8, dtype=np.float64)
        for bit in range(8):
            planes[bit] = float(((ch >> bit) & 1).sum()) / total * 100.0
        out[name] = planes
    return out


def render_bitplane_bars_all_algorithms(
    plain_path: Path,
    encrypted_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """One row of R/G/B bitplane bar charts per algorithm (plain vs encrypted)."""
    valid = [(alg, p) for alg, p in encrypted_paths.items() if p.exists()]
    if not valid:
        return

    n_rows = len(valid)
    bits = np.arange(8)
    plain_img = _load_preview(plain_path)
    p_perc = _bitplane_percentages(plain_img)

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, (alg, enc_path) in enumerate(valid):
        e_perc = _bitplane_percentages(_load_preview(enc_path))
        alg_label = alg.replace("_", " ")

        for col, ch in enumerate(("R", "G", "B")):
            ax = axes[row][col]
            ax.bar(bits - 0.15, p_perc[ch], width=0.3, color=_CH_COLORS[ch],
                   label="Plain")
            ax.bar(bits + 0.15, e_perc[ch], width=0.3, color="#bbbbbb",
                   label="Encrypted")
            ax.set_xticks(bits)
            ax.set_xticklabels([str(b) for b in bits])
            ax.set_ylabel("% set bits")
            ax.set_title(f"{alg_label} — {ch}")
            ax.grid(axis="y", alpha=0.18)
            if col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Bit-plane set-bit percentages — {plain_path.stem} vs all algorithms",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
