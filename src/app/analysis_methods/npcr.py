import argparse
import numpy as np

from ..utils import align_rgb_images, load_rgb_image


def number_of_pixel_change_rate(first_image, second_image):
    """Compute NPCR between two images.

    NPCR measures the percentage of pixels that change between two images.
    For RGB images, each channel is compared independently and the result is
    reported per channel plus an overall grayscale-style rate.
    """
    img_a = load_rgb_image(first_image)
    img_b = load_rgb_image(second_image)

    img_a, img_b = align_rgb_images(img_a, img_b)

    if img_a.ndim == 2:
        changed = img_a != img_b
        return {"grayscale": float(changed.mean() * 100.0)}

    if img_a.ndim != 3:
        raise ValueError("Images must be grayscale or RGB images")

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        changed = img_a[..., index] != img_b[..., index]
        per_channel[channel_name] = float(changed.mean() * 100.0)

    overall_changed = np.any(img_a != img_b, axis=2)

    return {
        "per_channel": per_channel,
        "overall": float(overall_changed.mean() * 100.0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute NPCR between two images")
    parser.add_argument("first_image", help="Path to the first image")
    parser.add_argument("second_image", help="Path to the second image")
    args = parser.parse_args(argv)

    result = number_of_pixel_change_rate(args.first_image, args.second_image)
    if "grayscale" in result:
        print(f"NPCR: {result['grayscale']:.6f}%")
    else:
        print("NPCR per channel:")
        for channel, value in result["per_channel"].items():
            print(f"  {channel}: {value:.6f}%")
        print(f"  overall: {result['overall']:.6f}%")


if __name__ == "__main__":
    raise SystemExit(main())
