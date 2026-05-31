from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

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
from app.utils import encrypt_image, extract_pixels
from app.utils.handle_image_pixels import reconstruct_image

ALGORITHMS = (
    "AES_CTR",
    "AES_CBC",
    "AES_GCM",
    "AES_CCM",
    "TRIPLE_DES",
    "CHACHA20",
    "LOGISTIC_MAP",
    "HENON_MAP",
)
DEFAULT_KEY_PHRASE = "encryptionkey"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
GLOBAL_SHARED_DIR = ROOT_SRC / "app" / "shared"
GLOBAL_META_DIR = GLOBAL_SHARED_DIR / "meta_files"
GLOBAL_NONCE_DIR = GLOBAL_SHARED_DIR / "nonce_files"


def _local_dirs(base_dir: Path) -> dict[str, Path]:
    return {
        "meta": base_dir.parent / "meta_files",
        "nonce": base_dir.parent / "nonce_files",
        "encrypted": base_dir.parent / "encrypted",
        "plain_plus_one_bit": base_dir.parent / "plain_plus_one_bit",
        "encrypted_plus_one_bit": base_dir.parent / "encrypted_plus_one_bit",
        "json_output": base_dir.parent / "dataset_analysis_results",
    }


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.resolve(strict=False) == target.resolve(strict=False):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _local_nonce_path(base_dir: Path, image_path: Path) -> Path:
    return _local_dirs(base_dir)["nonce"] / f"{image_path.stem}.nonce"


def _local_meta_path(base_dir: Path, image_path: Path) -> Path:
    return _local_dirs(base_dir)["meta"] / f"{image_path.stem}.meta"


def _nonce_path(base_dir: Path, image_path: Path) -> Path:
    return _local_dirs(base_dir)["nonce"] / f"{image_path.stem}.nonce"


def _meta_path(base_dir: Path, image_path: Path) -> Path:
    return _local_dirs(base_dir)["meta"] / f"{image_path.stem}.meta"


def _image_token(image_path: Path, dataset_root: Path | None = None) -> str:
    if dataset_root is None:
        return image_path.stem

    try:
        relative_path = image_path.relative_to(dataset_root)
    except ValueError:
        return image_path.stem

    token_parts = list(relative_path.with_suffix("").parts)
    return "__".join(token_parts) if token_parts else image_path.stem


def _iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _make_one_bit_variant(input_path: Path, output_path: Path) -> Path:
    pixels, mode = extract_pixels(input_path)
    arr = np.asarray(pixels, dtype=np.uint8).copy()

    if arr.ndim == 2:
        arr[0, 0] = (int(arr[0, 0]) + 1) % 256
    elif arr.ndim == 3 and arr.shape[2] >= 1:
        arr[0, 0, 0] = (int(arr[0, 0, 0]) + 1) % 256
    else:
        raise ValueError("Image must be grayscale or RGB")

    reconstruct_image(arr, mode, output_path)
    return output_path


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
    # Serialize floats as fixed-point strings to avoid scientific notation
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

    image_token = _image_token(
        image_path, base_dir if dataset_root is not None else None
    )

    encrypted_path = (
        dirs["encrypted"] / f"{algorithm.lower()}_enc_{image_token}{image_path.suffix}"
    )
    plus_one_bit_plain_path = dirs["plain_plus_one_bit"] / image_token
    plus_one_bit_plain_path = plus_one_bit_plain_path.with_suffix(image_path.suffix)
    encrypted_plus_one_bit_path = dirs["encrypted_plus_one_bit"] / encrypted_path.name

    encrypt_image(image_path, encrypted_path, key_phrase, algorithm=algorithm)
    encrypted_nonce_path = _nonce_path(base_dir, encrypted_path)
    encrypted_meta_path = _meta_path(base_dir, encrypted_path)
    _copy_if_exists(encrypted_nonce_path, _local_nonce_path(base_dir, encrypted_path))
    _copy_if_exists(encrypted_meta_path, _local_meta_path(base_dir, encrypted_path))
    _make_one_bit_variant(image_path, plus_one_bit_plain_path)
    encrypt_image(
        plus_one_bit_plain_path,
        encrypted_plus_one_bit_path,
        key_phrase,
        algorithm=algorithm,
    )
    plus_one_bit_nonce_path = _nonce_path(base_dir, encrypted_plus_one_bit_path)
    _copy_if_exists(
        plus_one_bit_nonce_path,
        _local_nonce_path(base_dir, encrypted_plus_one_bit_path),
    )
    _copy_if_exists(
        _meta_path(base_dir, encrypted_plus_one_bit_path),
        _local_meta_path(base_dir, encrypted_plus_one_bit_path),
    )

    analysis = {
        "input_image": str(image_path),
        "algorithm": algorithm,
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
            dirs["json_output"] / f"{algorithm.lower()}_{image_token}_analysis.json"
        )

    output_json = output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(analysis), indent=2), encoding="utf-8")

    try:
        plus_one_bit_plain_path.unlink(missing_ok=True)
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
        image_paths = _iter_image_files(input_path)
        if not image_paths:
            print(f"Error: no supported images found in folder: {input_path}")
            return 2
        dataset_root = input_path
    else:
        image_paths = [input_path]
        dataset_root = input_path.parent.parent

    reports: list[dict] = []

    algorithms_to_run = (args.algorithm,) if args.algorithm is not None else ALGORITHMS

    for image_path in image_paths:
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
