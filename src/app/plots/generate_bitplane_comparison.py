from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..utils import extract_preview_pixels, load_rgb_image
from .plot_generation_helper import (
    ENCODED_PATTERN,
    _iter_image_files,
    _collect_dataset_roots,
    _find_plain_directories,
    _find_encrypted_directory,
    _group_encrypted_files,
)

_CH_COLORS = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}


def _load_preview(path: Path) -> np.ndarray:
    pixels, _ = extract_preview_pixels(path, max_size=(2048, 2048))
    return load_rgb_image(pixels)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using standard luminance weights."""
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def _bit_planes(channel: np.ndarray) -> list[np.ndarray]:
    """Return 8 bit-plane images (bit 7 = MSB first)."""
    return [((channel >> b) & 1).astype(np.uint8) * 255 for b in range(7, -1, -1)]


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


def _render_bitplane_comparison(
    plain_img: Path,
    enc_img: Path,
    output_path: Path,
    plain_label: str = "Original",
    enc_label: str = "Encrypted",
) -> None:
    """Render a 2×8 grid of bit-plane images: top row = plain, bottom row = encrypted."""
    plain_gray = _to_grayscale(_load_preview(plain_img))
    enc_gray = _to_grayscale(_load_preview(enc_img))

    plain_planes = _bit_planes(plain_gray)
    enc_planes = _bit_planes(enc_gray)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for row, (planes, row_label) in enumerate(
        [(plain_planes, plain_label), (enc_planes, enc_label)]
    ):
        for col, (plane, bit) in enumerate(zip(planes, range(7, -1, -1))):
            ax = axes[row, col]
            ax.imshow(plane, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Bit {bit}", fontsize=8)
        axes[row, 0].set_ylabel(
            row_label, fontsize=9, rotation=90, labelpad=4, va="center"
        )

    fig.suptitle(
        f"Bit-plane decomposition — {plain_img.stem} vs {enc_img.stem}",
        fontsize=11,
    )
    plt.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
            ax.set_xlabel("Bit plane")
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


def generate_bitplane_comparison_plots(
    images_root: Path,
    output_root: Path,
    dataset_filter: str | None = None,
    algorithm_filter: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    generated: list[Path] = []

    for dataset_root in _collect_dataset_roots(images_root.parent):
        if dataset_filter and dataset_root.name != dataset_filter:
            continue

        plain_dirs = _find_plain_directories(dataset_root)
        encrypted_dir = _find_encrypted_directory(dataset_root)
        if not plain_dirs or encrypted_dir is None:
            continue

        encrypted_map = _group_encrypted_files(encrypted_dir)

        for plain_dir in plain_dirs:
            for plain_image in _iter_image_files(plain_dir):
                encrypted_candidates = encrypted_map.get(plain_image.stem, [])
                if not encrypted_candidates:
                    continue

                for encrypted_image in sorted(encrypted_candidates):
                    match = ENCODED_PATTERN.match(encrypted_image.stem)
                    if match is None:
                        continue

                    algorithm = match.group("algorithm")
                    if algorithm_filter and algorithm != algorithm_filter:
                        continue

                    output_path = (
                        output_root
                        / dataset_root.name
                        / algorithm
                        / f"{plain_image.stem}_bitplane_comparison.png"
                    )
                    _render_bitplane_comparison(plain_image, encrypted_image, output_path)
                    generated.append(output_path)

                    if limit is not None and len(generated) >= limit:
                        return generated

    return generated


def _default_images_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "images" / "street_view"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "plots" / "bitplane_comparison"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate visual bit-plane comparison grids (2×8) for plain and encrypted images."
    )
    parser.add_argument("--images-root", type=Path, default=_default_images_root())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)

    generated = generate_bitplane_comparison_plots(
        args.images_root.resolve(),
        args.output_root.resolve(),
        dataset_filter=args.dataset,
        algorithm_filter=args.algorithm,
        limit=args.limit,
    )

    print(f"Generated {len(generated)} bitplane comparison plot(s).")
    for path in generated[:10]:
        print(path)
    if len(generated) > 10:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
