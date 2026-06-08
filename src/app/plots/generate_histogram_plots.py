from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..analysis_methods import image_histogram
from .plot_generation_helper import (
    ENCODED_PATTERN,
    _iter_image_files,
    _collect_dataset_roots,
    _find_plain_directories,
    _find_encrypted_directory,
    _group_encrypted_files,
)

CHANNEL_COLORS = {
    "R": "#d62728",
    "G": "#2ca02c",
    "B": "#1f77b4",
    "luminance": "#555555",
    "grayscale": "#111111",
}


def _normalize_histogram_result(result: dict) -> dict[str, dict[str, np.ndarray]]:
    if "grayscale" in result:
        return {"grayscale": result["grayscale"]["histogram"]}

    normalized = {
        "R": result["per_channel"]["R"]["histogram"],
        "G": result["per_channel"]["G"]["histogram"],
        "B": result["per_channel"]["B"]["histogram"],
    }
    return normalized


def _render_histogram_pair(
    plain_image: Path, encrypted_image: Path, output_path: Path, normalize: bool = False
) -> None:
    plain_result = _normalize_histogram_result(image_histogram(plain_image))
    encrypted_result = _normalize_histogram_result(image_histogram(encrypted_image))

    channel_order = ["R", "G", "B"]
    if "grayscale" in plain_result or "grayscale" in encrypted_result:
        channel_order = ["grayscale"]

    fig, axes = plt.subplots(
        1,
        len(channel_order),
        figsize=(4.5 * len(channel_order), 4.5),
        sharex=True,
        constrained_layout=True,
    )
    if len(channel_order) == 1:
        axes = np.asarray(axes).reshape(1, 1)

    for col_index, channel in enumerate(channel_order):
        ax = axes[col_index] if axes.ndim == 1 else axes[0, col_index]
        p_hist = plain_result[channel]
        e_hist = encrypted_result[channel]
        p_vals = p_hist.astype(np.float64)
        e_vals = e_hist.astype(np.float64)
        if normalize:
            p_total = (
                float(p_hist.sum()) if hasattr(p_hist, "sum") else float(np.sum(p_hist))
            )
            e_total = (
                float(e_hist.sum()) if hasattr(e_hist, "sum") else float(np.sum(e_hist))
            )
            if p_total > 0:
                p_vals = p_vals / p_total
            if e_total > 0:
                e_vals = e_vals / e_total

        bins = np.arange(256)
        ax.plot(
            bins,
            p_vals,
            color=CHANNEL_COLORS.get(channel, "#000000"),
            label="Plain",
            linewidth=1.2,
        )
        ax.fill_between(
            bins, p_vals, color=CHANNEL_COLORS.get(channel, "#000000"), alpha=0.12
        )
        ax.plot(
            bins,
            e_vals,
            color="#000000",
            label="Encrypted",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_title(channel.upper() if channel != "grayscale" else "Grayscale")
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.2)
        if col_index == 0:
            ax.set_ylabel("Frequency")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Histogram comparison\n{plain_image.stem} vs {encrypted_image.stem}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_histogram_plots(
    images_root: Path,
    output_root: Path,
    dataset_filter: str | None = None,
    algorithm_filter: str | None = None,
    limit: int | None = None,
    normalize: bool = False,
) -> list[Path]:
    generated: list[Path] = []

    for dataset_root in _collect_dataset_roots(images_root.parent):
        if dataset_filter and dataset_root.name != dataset_filter:
            continue

        plain_dirs = _find_plain_directories(dataset_root)
        encrypted_dir = _find_encrypted_directory(dataset_root)
        print(dataset_root)
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
                        / f"{plain_image.stem}_histogram.png"
                    )
                    _render_histogram_pair(
                        plain_image, encrypted_image, output_path, normalize=normalize
                    )
                    generated.append(output_path)

                    if limit is not None and len(generated) >= limit:
                        return generated

    return generated


def _default_images_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "images" / "street_view"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "plots" / "histogram"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate histogram comparison plots for plain and encrypted images."
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=_default_images_root(),
        help="Root folder containing the dataset image folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Folder where plot PNG files will be written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only generate plots for one dataset folder, such as architectural_plan or spatial.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Only generate plots for one algorithm prefix, such as aes_cbc or chacha20.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Stop after generating this many plot files.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize histograms to frequency (fraction of pixels). By default raw counts are shown.",
    )
    args = parser.parse_args(argv)

    generated = generate_histogram_plots(
        args.images_root.resolve(),
        args.output_root.resolve(),
        dataset_filter=args.dataset,
        algorithm_filter=args.algorithm,
        limit=args.limit,
        normalize=args.normalize,
    )

    print(f"Generated {len(generated)} histogram plot(s).")
    print(_default_images_root())
    for path in generated[:10]:
        print(path)
    if len(generated) > 10:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
