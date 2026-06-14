from __future__ import annotations

import argparse

import numpy as np

from PIL import Image

from ..utils import load_rgb_image

MAX_ANALYSIS_SIZE = (2048, 2048)


def _channel_histogram(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint8).ravel()
    return np.bincount(flat, minlength=256).astype(np.int64)


def _chi_square_uniform(histogram: np.ndarray) -> dict:
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
        "non_zero_bins": non_zero_bins,
        "mean_intensity": float(np.average(np.arange(256), weights=histogram)),
    }


def _overall_histogram_summary(per_channel: dict[str, dict]) -> dict:
    averaged_keys = (
        "mean_intensity",
    )
    summary: dict[str, object] = {}

    for key in averaged_keys:
        values = [ch.get(key) for ch in per_channel.values()]
        numeric = [v for v in values if isinstance(v, (int, float))]
        if numeric:
            summary[key] = float(np.mean(numeric))

    # chi2: sum (combined test statistic)
    chi2_values = [ch.get("chi2") for ch in per_channel.values()]
    chi2_numeric = [v for v in chi2_values if isinstance(v, (int, float))]
    if chi2_numeric:
        summary["chi2"] = float(np.sum(chi2_numeric))

    # chi2_p: min (worst-case channel — conservative security criterion)
    p_values = [ch.get("chi2_p") for ch in per_channel.values()]
    p_numeric = [v for v in p_values if isinstance(v, (int, float))]
    if p_numeric:
        summary["chi2_p"] = float(np.min(p_numeric))

    return summary


def image_histogram(image):
    img = load_rgb_image(image)

    if img.ndim == 2:
        return {"overall": _summarize_histogram(_channel_histogram(img))}

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Image must be grayscale or RGB")

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        per_channel[channel_name] = _summarize_histogram(
            _channel_histogram(img[..., index])
        )

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
