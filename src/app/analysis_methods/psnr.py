import argparse
import numpy as np
from pathlib import Path

from ..utils.imagePixels import extract_pixels


def _load_image(image):
    if isinstance(image, (str, Path)):
        pixels, _ = extract_pixels(Path(image))
        arr = np.asarray(pixels, dtype=np.uint8)
        if arr.ndim == 2:
            return np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3]
        raise ValueError("Image must be grayscale or RGB")
    return np.asarray(image)


def _psnr_uint8(first, second):
    mse = _mse_uint8(first, second)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def _mse_uint8(first, second):
    diff = first.astype(np.float64) - second.astype(np.float64)
    return float(np.mean(diff * diff))


def mean_squared_error(first_image, second_image):
    """Compute MSE between two images.

    For RGB inputs, returns per-channel MSE and overall MSE.
    For grayscale inputs, returns grayscale MSE.
    """
    img_a = _load_image(first_image)
    img_b = _load_image(second_image)

    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape")

    if img_a.ndim == 2:
        return {"grayscale": _mse_uint8(img_a, img_b)}

    if img_a.ndim != 3:
        raise ValueError("Images must be grayscale or RGB images")

    img_a = img_a[..., :3]
    img_b = img_b[..., :3]

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        per_channel[channel_name] = _mse_uint8(img_a[..., index], img_b[..., index])

    return {
        "per_channel": per_channel,
        "overall": _mse_uint8(img_a, img_b),
    }


def peak_signal_to_noise_ratio(first_image, second_image):
    """Compute PSNR between two images.

    For RGB inputs, returns per-channel PSNR and overall PSNR.
    For grayscale inputs, returns grayscale PSNR.
    """
    img_a = _load_image(first_image)
    img_b = _load_image(second_image)

    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape")

    if img_a.ndim == 2:
        return {"grayscale": _psnr_uint8(img_a, img_b)}

    if img_a.ndim != 3:
        raise ValueError("Images must be grayscale or RGB images")

    img_a = img_a[..., :3]
    img_b = img_b[..., :3]

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        per_channel[channel_name] = _psnr_uint8(img_a[..., index], img_b[..., index])

    return {
        "per_channel": per_channel,
        "overall": _psnr_uint8(img_a, img_b),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute PSNR/MSE between two images")
    parser.add_argument("first_image", help="Path to the first image")
    parser.add_argument("second_image", help="Path to the second image")
    parser.add_argument(
        "--metric",
        choices=("psnr", "mse"),
        default="psnr",
        help="Metric to compute",
    )
    args = parser.parse_args(argv)

    if args.metric == "psnr":
        result = peak_signal_to_noise_ratio(args.first_image, args.second_image)
        title = "PSNR"
        suffix = " dB"
    else:
        result = mean_squared_error(args.first_image, args.second_image)
        title = "MSE"
        suffix = ""

    if "grayscale" in result:
        print(f"{title}: {result['grayscale']:.6f}{suffix}")
    else:
        print(f"{title} per channel{(' (dB)' if args.metric == 'psnr' else '')}:")
        for channel, value in result["per_channel"].items():
            print(f"  {channel}: {value:.6f}")
        print(f"  overall: {result['overall']:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
