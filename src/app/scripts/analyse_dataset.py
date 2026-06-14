from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import numpy as np

CLEAN_UP_ENABLED = False
ROOT_SRC = Path(__file__).resolve().parents[2]
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from app.analysis_methods import (
    correlation_between_images,
    image_histogram,
    mean_squared_error,
    number_of_pixel_change_rate,
    peak_signal_to_noise_ratio,
    pixel_entropy_with_blocks,
    structural_similarity,
    unified_average_changing_intensity,
    horizontal_pixel_correlation,
    vertical_pixel_correlation,
    diagonal_pixel_correlation,
)
from app.utils import encrypt_image, extract_pixels, add_one_bit_to_pixel

ALGORITHMS = (
    "AES_CTR",
    "AES_CBC",
    "TRIPLE_DES",
    "CHACHA20",
    "LOGISTIC_MAP",
    "HENON_MAP",
)
DEFAULT_KEY_PHRASE = "encryptionkey"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _local_dirs(base_dir: Path) -> dict[str, Path]:
    return {
        "meta": base_dir / "meta_files",
        "nonce": base_dir / "nonce_files",
        "encrypted": base_dir / "encrypted",
        "plain_plus_one_bit": base_dir / "plain_plus_one_bit",
        "encrypted_plus_one_bit": base_dir / "encrypted_plus_one_bit",
        "json_output": base_dir / "dataset_analysis_results",
    }


def _nonce_path(base_dir: Path, image_path: Path) -> Path:
    return _local_dirs(base_dir)["nonce"] / f"{image_path.stem}.nonce"


def _meta_path(base_dir: Path, image_path: Path) -> Path:
    return _local_dirs(base_dir)["meta"] / f"{image_path.stem}.meta"


def _iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_metadata(image_path: Path, normalize_rgb: bool = False) -> dict[str, object]:
    pixels, _ = extract_pixels(image_path)
    arr = np.asarray(pixels)

    if normalize_rgb:
        from PIL import Image

        arr = np.asarray(Image.fromarray(arr).convert("RGB"), dtype=np.uint8)

    if arr.ndim == 2:
        shape = f"{arr.shape[0]}x{arr.shape[1]}"
    elif arr.ndim >= 3:
        shape = f"{arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}"
    else:
        shape = "unknown"

    return {
        "shape": shape,
        "file_size_bytes": int(image_path.stat().st_size),
        "memory_size_bytes": int(arr.nbytes),
    }


def _structured_simple_body(result: dict) -> dict:
    body: dict = {}
    if "per_channel" in result:
        body["per_channel"] = {
            channel: round(float(value), 6)
            for channel, value in result["per_channel"].items()
        }
    if "grayscale" in result:
        body["grayscale"] = round(float(result["grayscale"]), 6)
    if "overall" in result:
        body["overall"] = round(float(result["overall"]), 6)
    if "blocks" in result:
        body["blocks"] = int(result["blocks"])
    if "used_size" in result:
        body["used_size"] = [int(result["used_size"][0]), int(result["used_size"][1])]
    return body


def _structured_entropy_body(result: dict) -> dict:
    body = _structured_simple_body(result)
    if "block_entropies" in result:
        block_entropies: dict[str, dict] = {}
        for block_size, block_result in result["block_entropies"].items():
            if "error" in block_result:
                block_entropies[str(block_size)] = {"error": block_result["error"]}
                continue
            block_body = _structured_simple_body(block_result)
            if "error" in block_result:
                block_body["error"] = block_result["error"]
            block_entropies[str(block_size)] = block_body
        body["block_entropies"] = block_entropies
    return body


def _structured_histogram_summary(summary: dict) -> dict:
    return {
        "peak_intensity": int(summary["peak_intensity"]),
        "peak_count": int(summary["peak_count"]),
        "peak_percentage": round(float(summary["peak_percentage"]), 6),
        "total_pixels": int(summary["total_pixels"]),
        "non_zero_bins": int(summary["non_zero_bins"]),
        "mean_intensity": round(float(summary["mean_intensity"]), 6),
        "chi2": round(float(summary["chi2"]), 6),
        "chi2_p": (
            None
            if summary.get("chi2_p") is None
            else round(float(summary["chi2_p"]), 6)
        ),
    }


def _structured_histogram_overall(summary: dict) -> dict:
    return {
        "mean_intensity": round(float(summary["mean_intensity"]), 6),
        "chi2": round(float(summary["chi2"]), 6),
        "chi2_p": (
            None
            if summary.get("chi2_p") is None
            else round(float(summary["chi2_p"]), 6)
        ),
    }


def _structured_histogram_body(result: dict) -> dict:
    body: dict = {}
    if "per_channel" in result:
        body["per_channel"] = {
            channel: _structured_histogram_summary(summary)
            for channel, summary in result["per_channel"].items()
        }
    if "grayscale" in result:
        body["grayscale"] = _structured_histogram_summary(result["grayscale"])
    if "luminance" in result:
        body["luminance"] = _structured_histogram_summary(result["luminance"])
    if "overall" in result:
        body["overall"] = _structured_histogram_overall(result["overall"])
    return body


def _body_to_text(title: str, body: dict, percent: bool = False) -> str:
    suffix = " %" if percent else ""
    lines = [title]

    if "per_channel" in body and body["per_channel"]:
        first_value = next(iter(body["per_channel"].values()))
        if isinstance(first_value, dict):
            lines.append("Per channel:")
            for channel, summary in body["per_channel"].items():
                lines.append(
                    f"  {channel}: peak {summary['peak_intensity']} ({summary['peak_count']} pixels, {summary['peak_percentage']:.2f}% of {summary['total_pixels']})"
                )
                lines.append(f"    non-zero bins: {summary['non_zero_bins']}")
                lines.append(f"    mean intensity: {summary['mean_intensity']:.6f}")
                chi_p = summary.get("chi2_p")
                if chi_p is None:
                    lines.append(f"    chi2: {summary.get('chi2'):.3f} (p: N/A)")
                else:
                    lines.append(
                        f"    chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})"
                    )

            if "grayscale" in body and isinstance(body["grayscale"], dict):
                summary = body["grayscale"]
                lines.append(
                    f"Grayscale peak: {summary['peak_intensity']} ({summary['peak_count']} pixels, {summary['peak_percentage']:.2f}% of {summary['total_pixels']})"
                )
                lines.append(f"Non-zero bins: {summary['non_zero_bins']}")
                lines.append(f"Mean intensity: {summary['mean_intensity']:.6f}")
                chi_p = summary.get("chi2_p")
                if chi_p is None:
                    lines.append(f"chi2: {summary.get('chi2'):.3f} (p: N/A)")
                else:
                    lines.append(f"chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})")

            if "luminance" in body:
                summary = body["luminance"]
                lines.append("Luminance:")
                lines.append(
                    f"  peak: {summary['peak_intensity']} ({summary['peak_count']} pixels, {summary['peak_percentage']:.2f}% of {summary['total_pixels']})"
                )
                lines.append(f"  non-zero bins: {summary['non_zero_bins']}")
                lines.append(f"  mean intensity: {summary['mean_intensity']:.6f}")
                chi_p = summary.get("chi2_p")
                if chi_p is None:
                    lines.append(f"  chi2: {summary.get('chi2'):.3f} (p: N/A)")
                else:
                    lines.append(f"  chi2: {summary.get('chi2'):.3f} (p: {chi_p:.3e})")

            return "\n".join(lines)

    if "per_channel" in body:
        lines.append("Per channel:")
        for channel, value in body["per_channel"].items():
            lines.append(f"  {channel}: {value:.6f}{suffix}")

    if "grayscale" in body:
        value = body["grayscale"]
        lines.append(f"Grayscale: {value:.6f}{suffix}")

    if "overall" in body:
        value = body["overall"]
        lines.append(f"Overall: {value:.6f}{suffix}")

    if "blocks" in body:
        lines.append(
            f"Blocks used: {body['blocks']} ({body['used_size'][0]}x{body['used_size'][1]} area)"
        )

    if "block_entropies" in body:
        for block_size in sorted(body["block_entropies"], key=lambda x: int(x)):
            block_result = body["block_entropies"][block_size]
            lines.append(f"Block entropy ({block_size}x{block_size}):")
            if "error" in block_result:
                lines.append(f"  N/A: {block_result['error']}")
                continue
            if "per_channel" in block_result:
                lines.append("  Per channel:")
                for channel, value in block_result["per_channel"].items():
                    lines.append(f"    {channel}: {value:.6f}")
                lines.append(f"  Overall: {block_result['overall']:.6f}")
            else:
                lines.append(f"  Grayscale: {block_result['grayscale']:.6f}")
            lines.append(
                f"  Blocks used: {block_result['blocks']} ({block_result['used_size'][0]}x{block_result['used_size'][1]} area)"
            )

    if "peak_intensity" in body:
        lines.append(
            f"Peak: {body['peak_intensity']} ({body['peak_count']} pixels, {body['peak_percentage']:.2f}% of {body['total_pixels']})"
        )
        lines.append(f"Non-zero bins: {body['non_zero_bins']}")
        lines.append(f"Mean intensity: {body['mean_intensity']:.6f}")
        chi_p = body.get("chi2_p")
        if chi_p is None:
            lines.append(f"chi2: {body.get('chi2'):.3f} (p: N/A)")
        else:
            lines.append(f"chi2: {body.get('chi2'):.3f} (p: {chi_p:.3e})")

    if "luminance" in body:
        lines.append("Luminance:")
        lum = body["luminance"]
        lines.append(
            f"  peak: {lum['peak_intensity']} ({lum['peak_count']} pixels, {lum['peak_percentage']:.2f}% of {lum['total_pixels']})"
        )
        lines.append(f"  non-zero bins: {lum['non_zero_bins']}")
        lines.append(f"  mean intensity: {lum['mean_intensity']:.6f}")
        chi_p = lum.get("chi2_p")
        if chi_p is None:
            lines.append(f"  chi2: {lum.get('chi2'):.3f} (p: N/A)")
        else:
            lines.append(f"  chi2: {lum.get('chi2'):.3f} (p: {chi_p:.3e})")

    return "\n".join(lines)


def _build_ui_sections(
    encrypted_path: Path, plain_path: Path, encrypted_plus_one_bit_path: Path
) -> list[dict]:
    e_str = str(encrypted_path)
    eb_str = str(encrypted_plus_one_bit_path)
    p_str = str(plain_path)

    return [
        {
            "category": "Encrypted",
            "title": "Horizontal correlation",
            "body": _structured_simple_body(horizontal_pixel_correlation(e_str)),
        },
        {
            "category": "Encrypted",
            "title": "Vertical correlation",
            "body": _structured_simple_body(vertical_pixel_correlation(e_str)),
        },
        {
            "category": "Encrypted",
            "title": "Diagonal correlation",
            "body": _structured_simple_body(diagonal_pixel_correlation(e_str)),
        },
        {
            "category": "Encrypted",
            "title": "Entropy results",
            "body": _structured_entropy_body(pixel_entropy_with_blocks(e_str)),
        },
        {
            "category": "Encrypted",
            "title": "Histogram results",
            "body": _structured_histogram_body(image_histogram(e_str)),
        },
        {
            "category": "Encrypted vs Encrypted",
            "title": "Encrypted vs Encrypted correlation",
            "body": _structured_simple_body(correlation_between_images(e_str, eb_str)),
        },
        {
            "category": "Encrypted vs Encrypted",
            "title": "NPCR results",
            "body": _structured_simple_body(number_of_pixel_change_rate(e_str, eb_str)),
        },
        {
            "category": "Encrypted vs Encrypted",
            "title": "UACI results",
            "body": _structured_simple_body(
                unified_average_changing_intensity(e_str, eb_str)
            ),
        },
        {
            "category": "Encrypted vs Plain",
            "title": "Encrypted vs Plain correlation",
            "body": _structured_simple_body(correlation_between_images(e_str, p_str)),
        },
        {
            "category": "Encrypted vs Plain",
            "title": "MSE results",
            "body": _structured_simple_body(mean_squared_error(e_str, p_str)),
        },
        {
            "category": "Encrypted vs Plain",
            "title": "PSNR results",
            "body": _structured_simple_body(peak_signal_to_noise_ratio(e_str, p_str)),
        },
        {
            "category": "Encrypted vs Plain",
            "title": "SSIM results",
            "body": _structured_simple_body(structural_similarity(e_str, p_str)),
        },
    ]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return format(value, ".6f")
    if isinstance(value, (np.floating,)):
        return format(float(value), ".6f")
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_analysis(
    image_path: Path,
    algorithm: str = "AES_CBC",
    key_phrase: str = DEFAULT_KEY_PHRASE,
    output_json: Path | None = None,
    dataset_root: Path | None = None,
) -> dict:
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    algorithm = algorithm.upper()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    base_dir = (
        dataset_root.resolve() if dataset_root is not None else image_path.parent.parent
    )
    dirs = _local_dirs(base_dir)
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    encrypted_path = (
        dirs["encrypted"] / f"{algorithm.lower()}_enc_{image_path.stem}{image_path.suffix}"
    )
    plus_one_bit_plain_path = dirs["plain_plus_one_bit"] / f"{image_path.stem}{image_path.suffix}"
    encrypted_plus_one_bit_path = dirs["encrypted_plus_one_bit"] / encrypted_path.name

    start = time.perf_counter()
    encrypt_image(image_path, encrypted_path, key_phrase, algorithm=algorithm)
    end = time.perf_counter()
    encryption_time = f"{end - start:.3f}"
    encrypted_nonce_path = _nonce_path(base_dir, encrypted_path)
    encrypted_meta_path = _meta_path(base_dir, encrypted_path)
    encrypt_image(
        plus_one_bit_plain_path,
        encrypted_plus_one_bit_path,
        key_phrase,
        algorithm=algorithm,
    )

    analysis = {
        "input_image": str(image_path),
        "algorithm": algorithm,
        "encryption_time_seconds": encryption_time,
        "plain_image": _image_metadata(image_path, normalize_rgb=True),
        "encrypted_image": _image_metadata(encrypted_path, normalize_rgb=True),
        "analysis_sections": _build_ui_sections(
            encrypted_path, image_path, encrypted_plus_one_bit_path
        ),
        "analysis_text": "\n\n".join(
            f"{section['category']}\n{section['title']}\n{_body_to_text(section['title'], section['body'], percent=section['title'] in {'NPCR results', 'UACI results'})}"
            for section in _build_ui_sections(
                encrypted_path, image_path, encrypted_plus_one_bit_path
            )
        ),
    }

    if output_json is None:
        output_json = (
            dirs["json_output"] / f"{algorithm.lower()}_{image_path.stem}_analysis.json"
        )

    output_json = output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(analysis), indent=2), encoding="utf-8")

    if CLEAN_UP_ENABLED == True:
        try:
            encrypted_plus_one_bit_path.unlink(missing_ok=True)
            encrypted_path.unlink(missing_ok=True)
            encrypted_nonce_path.unlink(missing_ok=True)
            encrypted_meta_path.unlink(missing_ok=True)
        except Exception:
            pass

    return analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Encrypt one image or every image in a folder and run the full analysis suite, saving the results as JSON."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an image or a folder of images to analyze",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default=None,
        help="Optional: specific encryption algorithm to use; omit to run all",
    )
    parser.add_argument(
        "--key-phrase",
        default=DEFAULT_KEY_PHRASE,
        help="Key phrase used to derive the encryption key",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path for single-image mode; folder mode writes per-image JSON files automatically",
    )
    args = parser.parse_args(argv)

    input_path = args.input_path.resolve()
    if not input_path.exists():
        print(f"Error: input path not found: {input_path}")
        return 2

    if input_path.is_dir():
        print(f"Analyzing all images in folder: {input_path}")
        image_paths = _iter_image_files(input_path / "plain")
        if not image_paths:
            print(f"Error: no supported images found in folder: {input_path}")
            return 2
        dataset_root = input_path
    else:
        image_paths = [input_path]
        dataset_root = input_path.parent.parent

    reports: list[dict] = []

    algorithms_to_run = (args.algorithm,) if args.algorithm is not None else ALGORITHMS

    start = time.perf_counter()

    for image_path in image_paths:
        plus_one_bit_plain_path = add_one_bit_to_pixel(image_path)
        for alg in algorithms_to_run:
            try:
                out_json = (
                    args.output_json
                    if len(image_paths) == 1 and args.algorithm is not None
                    else None
                )
                print(f"Running {alg} on {image_path.name}")
                report = run_analysis(
                    image_path,
                    algorithm=alg,
                    key_phrase=args.key_phrase,
                    output_json=out_json,
                    dataset_root=dataset_root,
                )
                reports.append(
                    {"image": str(image_path), "algorithm": alg, "report": report}
                )
            except Exception as exc:
                print(f"Error running {alg} on {image_path}: {exc}")

        if CLEAN_UP_ENABLED == True:
            plus_one_bit_plain_path.unlink(missing_ok=True)
    end = time.perf_counter()

    print(f"Execution time: {end - start:.3f} seconds")

    if CLEAN_UP_ENABLED == True:
        try:
            (input_path / "encrypted").rmdir()
            (input_path / "encrypted_plus_one_bit").rmdir()
            (input_path / "plain_plus_one_bit").rmdir()
            (input_path / "nonce_files").rmdir()
            (input_path / "meta_files").rmdir()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
