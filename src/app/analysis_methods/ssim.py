from PIL import Image
import argparse
import numpy as np


def _load_image(image):
    if isinstance(image, str):
        return np.array(Image.open(image).convert("RGB"), dtype=np.uint8)
    return np.asarray(image)


def _ssim_uint8(a, b, K1=0.01, K2=0.03, L=255.0):
    # Compute a simple global SSIM (single-window) for uint8 images
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu_x = np.mean(a)
    mu_y = np.mean(b)

    sigma_x2 = np.mean((a - mu_x) ** 2)
    sigma_y2 = np.mean((b - mu_y) ** 2)
    sigma_xy = np.mean((a - mu_x) * (b - mu_y))

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    if den == 0:
        # If images are constant and identical, return 1.0
        return 1.0 if np.allclose(a, b) else 0.0
    return float(num / den)


def structural_similarity(first_image, second_image):
    """Compute SSIM between two images.

    Returns a dict matching the same shape as PSNR/MSE helpers:
    - grayscale: value for grayscale inputs
    - per_channel: dict for R/G/B when input is RGB
    - overall: overall SSIM computed on luminance
    """
    img_a = _load_image(first_image)
    img_b = _load_image(second_image)

    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape")

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
