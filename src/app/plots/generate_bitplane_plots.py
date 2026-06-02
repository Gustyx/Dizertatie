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


def _load_preview(path: Path) -> np.ndarray:
    pixels, _ = extract_preview_pixels(path, max_size=(2048, 2048))
    return load_rgb_image(pixels)


def _bitplane_percentages(img: np.ndarray) -> dict[str, np.ndarray]:
    # img: HxWx3 rgb uint8
    out = {}
    h, w = img.shape[0], img.shape[1]
    total = float(h * w)
    for i, name in enumerate(("R", "G", "B")):
        ch = img[..., i].astype(np.uint8)
        planes = np.empty(8, dtype=np.float64)
        for bit in range(8):
            planes[bit] = float(((ch >> bit) & 1).sum()) / total * 100.0
        out[name] = planes
    return out


def _render_bitplane_bar(plain_img: Path, enc_img: Path, output_path: Path) -> None:
    p = _load_preview(plain_img)
    e = _load_preview(enc_img)

    p_perc = _bitplane_percentages(p)
    e_perc = _bitplane_percentages(e)

    bits = np.arange(8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, ch in zip(axes, ("R", "G", "B")):
        # colored = plain, grey = encrypted (user preference)
        ax.bar(
            bits - 0.15,
            p_perc[ch],
            width=0.3,
            color={"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}[ch],
        )
        ax.bar(bits + 0.15, e_perc[ch], width=0.3, color="#bbbbbb")
        ax.set_xticks(bits)
        ax.set_xticklabels([str(b) for b in bits])
        ax.set_xlabel("Bit plane")
        ax.set_ylabel("% set bits")
        ax.set_title(ch)
        ax.grid(alpha=0.18)
    # legend removed to avoid obscuring the bars (user requested)

    fig.suptitle(f"Bit-plane set-bit percentages\n{plain_img.stem} vs {enc_img.stem}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_bitplane_plots(
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

                    output_path = (
                        output_root
                        / dataset_root.name
                        / algorithm
                        / f"{plain_image.stem}_bitplane.png"
                    )
                    _render_bitplane_bar(plain_image, encrypted_image, output_path)
                    generated.append(output_path)

                    if limit is not None and len(generated) >= limit:
                        return generated

                    # also produce encrypted vs encrypted if plus-one-bit exists
                    if encrypted_plus_one_bit_dir is not None:
                        eb_path = encrypted_plus_one_bit_dir / encrypted_image.name
                        if eb_path.exists():
                            output_path2 = (
                                output_root
                                / dataset_root.name
                                / algorithm
                                / f"{plain_image.stem}_enc_vs_enc_bitplane.png"
                            )
                            _render_bitplane_bar(encrypted_image, eb_path, output_path2)
                            generated.append(output_path2)

                            if limit is not None and len(generated) >= limit:
                                return generated

    return generated


def _default_images_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "images" / "street_view"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "plots" / "bitplane"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate bit-plane analysis plots.")
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
    args = parser.parse_args(argv)

    generated = generate_bitplane_plots(
        args.images_root.resolve(),
        args.output_root.resolve(),
        dataset_filter=args.dataset,
        algorithm_filter=args.algorithm,
        limit=args.limit,
    )

    print(f"Generated {len(generated)} bitplane plot(s).")
    print(_default_images_root())
    for path in generated[:10]:
        print(path)
    if len(generated) > 10:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
