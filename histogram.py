from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
from PIL import Image


def _load_image(image):
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert("RGB"), dtype=np.uint8)

    img = np.asarray(image)
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


def image_histogram(image):
    """Compute 256-bin histograms for image channels.

    Returns a dict with per-channel RGB histograms for color images and a
    grayscale histogram for monochrome images.
    """
    img = _load_image(image)

    if img.ndim == 2:
        return {"grayscale": _summarize_histogram(_channel_histogram(img))}

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
        "luminance": _summarize_histogram(_channel_histogram(luminance)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute histogram metrics for an image"
    )
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args(argv)

    result = image_histogram(args.image)
    if "grayscale" in result:
        summary = result["grayscale"]
        print("Grayscale histogram:")
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
        lum = result["luminance"]
        print("  luminance:")
        print(f"    peak intensity: {lum['peak_intensity']}")
        print(
            f"    peak count: {lum['peak_count']} ({lum['peak_percentage']:.2f}% of {lum['total_pixels']} pixels)"
        )
        print(f"    non-zero bins: {lum['non_zero_bins']}")
        print(f"    mean intensity: {lum['mean_intensity']:.6f}")
        chi_p = lum.get("chi2_p")
        if chi_p is None:
            print(f"    chi2: {lum.get('chi2'):.3f} (p: N/A)")
        else:
            print(f"    chi2: {lum.get('chi2'):.3e} (p: {chi_p:.3e})")


if __name__ == "__main__":
    raise SystemExit(main())
