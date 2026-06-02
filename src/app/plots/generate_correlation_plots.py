from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..utils import align_rgb_images, extract_preview_pixels, load_rgb_image
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
    "grayscale": "#111111",
}


def _load_image(image: Path | str) -> np.ndarray:
    if not isinstance(image, (str, Path)):
        return load_rgb_image(image)
    pixels, _ = extract_preview_pixels(Path(image), max_size=(2048, 2048))
    return load_rgb_image(pixels)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _sample_pairs(
    x: np.ndarray, y: np.ndarray, max_points: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    if x.size != y.size:
        raise ValueError("Paired arrays must have the same size")
    if max_points <= 0 or x.size <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.size, size=max_points, replace=False)
    return x[indices], y[indices]


def _plot_channel(
    ax: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    title: str,
    color: str,
    max_points: int,
) -> float:
    first_sample, second_sample = _sample_pairs(first, second, max_points=max_points)
    correlation = _pearson(first, second)
    ax.scatter(
        first_sample,
        second_sample,
        s=7,
        alpha=0.18,
        color=color,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        [0, 255], [0, 255], color="#444444", linewidth=1.0, linestyle="--", alpha=0.7
    )
    ax.set_title(f"{title}\nr = {correlation:.6f}", fontsize=10)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    return correlation


def _render_correlation_plot(
    first_image: Path,
    second_image: Path,
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    first = _load_image(first_image)
    second = _load_image(second_image)
    first, second = align_rgb_images(first, second)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
    axes_flat = axes.ravel()

    channel_specs = [
        ("R", first[..., 0], second[..., 0]),
        ("G", first[..., 1], second[..., 1]),
        ("B", first[..., 2], second[..., 2]),
    ]
    gray_weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    first_gray = (first.astype(np.float64) * gray_weights).sum(axis=2)
    second_gray = (second.astype(np.float64) * gray_weights).sum(axis=2)
    channel_specs.append(("grayscale", first_gray, second_gray))

    for ax, (channel_name, first_values, second_values) in zip(
        axes_flat, channel_specs
    ):
        _plot_channel(
            ax,
            first_values,
            second_values,
            channel_name.upper() if channel_name != "grayscale" else "Grayscale",
            CHANNEL_COLORS[channel_name],
            max_points=max_points,
        )
        if channel_name in {"R", "G", "B"}:
            ax.set_xlabel("First image intensity")
            ax.set_ylabel("Second image intensity")
        else:
            ax.set_xlabel("First image luminance")
            ax.set_ylabel("Second image luminance")

    fig.suptitle(f"{title}\n{first_image.stem} vs {second_image.stem}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_correlation_plots(
    images_root: Path,
    output_root: Path,
    dataset_filter: str | None = None,
    algorithm_filter: str | None = None,
    limit: int | None = None,
    max_points: int = 12000,
) -> list[Path]:
    generated: list[Path] = []

    for dataset_root in _collect_dataset_roots(images_root.parent):
        if dataset_filter and dataset_root.name != dataset_filter:
            continue

        plain_dirs = _find_plain_directories(dataset_root)
        encrypted_dir = _find_encrypted_directory(dataset_root)
        encrypted_plus_one_bit_dir = _find_encrypted_directory(
            dataset_root, "encrypted_plus_one_bit"
        )
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

                    plain_output = (
                        output_root
                        / dataset_root.name
                        / algorithm
                        / f"{plain_image.stem}_plain_vs_encrypted_correlation.png"
                    )
                    _render_correlation_plot(
                        plain_image,
                        encrypted_image,
                        plain_output,
                        title="Plain vs Encrypted correlation",
                        max_points=max_points,
                    )
                    generated.append(plain_output)

                    if limit is not None and len(generated) >= limit:
                        return generated

                    if encrypted_plus_one_bit_dir is not None:
                        encrypted_plus_one_bit_path = (
                            encrypted_plus_one_bit_dir / encrypted_image.name
                        )
                        if encrypted_plus_one_bit_path.exists():
                            encrypted_output = (
                                output_root
                                / dataset_root.name
                                / algorithm
                                / f"{plain_image.stem}_encrypted_vs_encrypted_correlation.png"
                            )
                            _render_correlation_plot(
                                encrypted_image,
                                encrypted_plus_one_bit_path,
                                encrypted_output,
                                title="Encrypted vs Encrypted correlation",
                                max_points=max_points,
                            )
                            generated.append(encrypted_output)

                            if limit is not None and len(generated) >= limit:
                                return generated

    return generated


def _default_images_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "images" / "street_view"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "plots" / "correlation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate correlation plots for plain-vs-encrypted and encrypted-vs-encrypted image pairs."
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
        help="Folder where correlation plot PNG files will be written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only generate plots for one dataset folder, such as architectural_plan or cctv_footage.",
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
        "--max-points",
        type=int,
        default=12000,
        help="Maximum sampled point pairs per subplot to keep the figure readable.",
    )
    args = parser.parse_args(argv)

    generated = generate_correlation_plots(
        args.images_root.resolve(),
        args.output_root.resolve(),
        dataset_filter=args.dataset,
        algorithm_filter=args.algorithm,
        limit=args.limit,
        max_points=args.max_points,
    )

    print(f"Generated {len(generated)} correlation plot(s).")
    print(_default_images_root())
    for path in generated[:10]:
        print(path)
    if len(generated) > 10:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
