import argparse
import numpy as np

from ..utils import align_rgb_images, load_rgb_image
from skimage.metrics import structural_similarity as ski_ssim


def _ssim_uint8(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(ski_ssim(a, b, data_range=255.0))


def structural_similarity(first_image, second_image):
    """Compute SSIM between two images.

    Returns a dict matching the same shape as PSNR/MSE helpers:
    - grayscale: value for grayscale inputs
    - per_channel: dict for R/G/B when input is RGB
    - overall: overall SSIM computed on luminance
    """
    img_a = load_rgb_image(first_image)
    img_b = load_rgb_image(second_image)

    img_a, img_b = align_rgb_images(img_a, img_b)

    # Grayscale image
    if img_a.ndim == 2:
        return {"grayscale": _ssim_uint8(img_a, img_b)}

    if img_a.ndim != 3:
        raise ValueError("Images must be grayscale or RGB images")

    img_a = img_a[..., :3]
    img_b = img_b[..., :3]

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        per_channel[channel_name] = _ssim_uint8(img_a[..., index], img_b[..., index])

    # compute overall SSIM on luminance (standard weights)
    r_w, g_w, b_w = 0.299, 0.587, 0.114
    lum_a = img_a[..., 0] * r_w + img_a[..., 1] * g_w + img_a[..., 2] * b_w
    lum_b = img_b[..., 0] * r_w + img_b[..., 1] * g_w + img_b[..., 2] * b_w
    overall = _ssim_uint8(lum_a, lum_b)

    return {"per_channel": per_channel, "overall": overall}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute SSIM between two images")
    parser.add_argument("first_image", help="Path to the first image")
    parser.add_argument("second_image", help="Path to the second image")
    args = parser.parse_args(argv)

    result = structural_similarity(args.first_image, args.second_image)

    if "grayscale" in result:
        print(f"SSIM: {result['grayscale']:.6f}")
    else:
        print("SSIM per channel:")
        for channel, value in result["per_channel"].items():
            print(f"  {channel}: {value:.6f}")
        print(f"  overall: {result['overall']:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
