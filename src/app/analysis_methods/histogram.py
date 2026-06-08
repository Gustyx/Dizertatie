from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from PIL import Image

from ..utils import extract_preview_pixels

MAX_ANALYSIS_SIZE = (2048, 2048)


def _downsample_array(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2 and max(img.shape) > MAX_ANALYSIS_SIZE[0]:
        preview = Image.fromarray(img)
        preview.thumbnail(MAX_ANALYSIS_SIZE)
        return np.asarray(preview, dtype=np.uint8)

    if img.ndim == 3 and max(img.shape[:2]) > MAX_ANALYSIS_SIZE[0]:
        preview = Image.fromarray(img[..., :3].astype(np.uint8))
        preview.thumbnail(MAX_ANALYSIS_SIZE)
        return np.asarray(preview, dtype=np.uint8)

    return img


def _load_image(image):
    if isinstance(image, (str, Path)):
        pixels, _ = extract_preview_pixels(Path(image), max_size=MAX_ANALYSIS_SIZE)
        arr = np.asarray(pixels, dtype=np.uint8)
        if arr.ndim == 2:
            return np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3]
        raise ValueError("Image must be grayscale or RGB")

    img = _downsample_array(np.asarray(image))
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[..., :3].astype(np.uint8, copy=False)
    raise ValueError("Image must be grayscale or RGB")


def _channel_histogram(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint8).ravel()
    return np.bincount(flat, minlength=256).astype(np.int64)


def _chi_square_uniform(histogram: np.ndarray) -> dict:
    """Compute chi-square statistic comparing histogram to uniform distribution.

    Returns dict with keys: chi2, dof, p (p may be None if scipy not available).
    """
    total = int(histogram.sum())
    dof = 256 - 1
    if total == 0:
        return {"chi2": float("nan"), "dof": dof, "p": None}

    expected = float(total) / 256.0
    chi2 = float(((histogram.astype(np.float64) - expected) ** 2 / expected).sum())
    p = None
    try:
        from scipy.stats import chi2 as _chi2

        p = float(_chi2.sf(chi2, dof))
    except Exception:
        pass
    return {"chi2": chi2, "dof": dof, "p": p}


def _summarize_histogram(histogram: np.ndarray) -> dict:
    peak_intensity = int(np.argmax(histogram))
    peak_count = int(histogram[peak_intensity])
    non_zero_bins = int(np.count_nonzero(histogram))
    total_pixels = int(histogram.sum())
    peak_percentage = (
        float((peak_count / total_pixels) * 100.0) if total_pixels else 0.0
    )
    chi = _chi_square_uniform(histogram)
    return {
        "histogram": histogram,
        "peak_intensity": peak_intensity,
        "peak_count": peak_count,
        "peak_percentage": peak_percentage,
        "total_pixels": total_pixels,
        "chi2": chi["chi2"],
        "chi2_p": chi["p"],
        "chi2_dof": chi["dof"],
        "non_zero_bins": non_zero_bins,
        "mean_intensity": float(np.average(np.arange(256), weights=histogram)),
    }


def _overall_histogram_summary(per_channel: dict[str, dict]) -> dict:
    numeric_keys = (
        "peak_intensity",
        "peak_count",
        "peak_percentage",
        "total_pixels",
        "non_zero_bins",
        "mean_intensity",
        "chi2",
        "chi2_p",
        "chi2_dof",
    )
    summary: dict[str, object] = {}

    for key in numeric_keys:
        values = [channel_summary.get(key) for channel_summary in per_channel.values()]
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        if not numeric_values:
            continue
        summary[key] = float(np.mean(numeric_values))

    return summary


def image_histogram(image):
    """Compute 256-bin histograms for image channels.

    Returns a dict with per-channel RGB histograms for color images and an
    overall histogram summary computed as the mean of the R, G and B summaries.
    """
    img = _load_image(image)

    if img.ndim == 2:
        return {"overall": _summarize_histogram(_channel_histogram(img))}

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Image must be grayscale or RGB")

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        per_channel[channel_name] = _summarize_histogram(
            _channel_histogram(img[..., index])
        )

    luminance = np.rint(
        0.299 * img[..., 0].astype(np.float64)
        + 0.587 * img[..., 1].astype(np.float64)
        + 0.114 * img[..., 2].astype(np.float64)
    ).astype(np.uint8)

    return {
        "per_channel": per_channel,
        "overall": _overall_histogram_summary(per_channel),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute histogram metrics for an image"
    )
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args(argv)

    result = image_histogram(args.image)
    if "overall" in result:
        summary = result["overall"]
        print("Overall histogram:")
        print(f"  peak intensity: {summary['peak_intensity']}")
        print(
            f"  peak count: {summary['peak_count']} ({summary['peak_percentage']:.2f}% of {summary['total_pixels']} pixels)"
        )
        print(f"  non-zero bins: {summary['non_zero_bins']}")
        print(f"  mean intensity: {summary['mean_intensity']:.6f}")
        chi_p = summary.get("chi2_p")
        if chi_p is None:
            print(f"  chi2: {summary.get('chi2'):.3f} (p: N/A)")
        else:
            print(f"  chi2: {summary.get('chi2'):.3e} (p: {chi_p:.3e})")
    else:
        print("RGB histogram:")
        for channel, summary in result["per_channel"].items():
            print(f"  {channel}:")
            print(f"    peak intensity: {summary['peak_intensity']}")
            print(
                f"    peak count: {summary['peak_count']} ({summary['peak_percentage']:.2f}% of {summary['total_pixels']} pixels)"
            )
            print(f"    non-zero bins: {summary['non_zero_bins']}")
            print(f"    mean intensity: {summary['mean_intensity']:.6f}")
            chi_p = summary.get("chi2_p")
            if chi_p is None:
                print(f"    chi2: {summary.get('chi2'):.3f} (p: N/A)")
            else:
                print(f"    chi2: {summary.get('chi2'):.3e} (p: {chi_p:.3e})")
        overall = result["overall"]
        print("  overall:")
        print(f"    peak intensity: {overall['peak_intensity']}")
        print(
            f"    peak count: {overall['peak_count']} ({overall['peak_percentage']:.2f}% of {overall['total_pixels']} pixels)"
        )
        print(f"    non-zero bins: {overall['non_zero_bins']}")
        print(f"    mean intensity: {overall['mean_intensity']:.6f}")
        chi_p = overall.get("chi2_p")
        if chi_p is None:
            print(f"    chi2: {overall.get('chi2'):.3f} (p: N/A)")
        else:
            print(f"    chi2: {overall.get('chi2'):.3e} (p: {chi_p:.3e})")


if __name__ == "__main__":
    raise SystemExit(main())
