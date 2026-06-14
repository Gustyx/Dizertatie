import argparse
import numpy as np

from ..utils import align_rgb_images, load_rgb_image


def unified_average_changing_intensity(first_image, second_image):
    img_a = load_rgb_image(first_image)
    img_b = load_rgb_image(second_image)

    img_a, img_b = align_rgb_images(img_a, img_b)

    if img_a.ndim == 2:
        diff = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16))
        uaci_value = float(diff.mean() / 255.0 * 100.0)
        return {"grayscale": uaci_value}

    if img_a.ndim != 3:
        raise ValueError("Images must be grayscale or RGB images")

    per_channel = {}
    for index, channel_name in enumerate(("R", "G", "B")):
        diff = np.abs(
            img_a[..., index].astype(np.int16) - img_b[..., index].astype(np.int16)
        )
        per_channel[channel_name] = float(diff.mean() / 255.0 * 100.0)

    # Overall: mean absolute difference across all channels
    all_diff = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16))
    overall = float(all_diff.mean() / 255.0 * 100.0)

    return {"per_channel": per_channel, "overall": overall}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute UACI between two images")
    parser.add_argument("first_image", help="Path to the first image")
    parser.add_argument("second_image", help="Path to the second image")
    args = parser.parse_args(argv)

    result = unified_average_changing_intensity(args.first_image, args.second_image)
    if "grayscale" in result:
        print(f"UACI: {result['grayscale']:.6f}%")
    else:
        print("UACI per channel:")
        for channel, value in result["per_channel"].items():
            print(f"  {channel}: {value:.6f}%")
        print(f"  overall: {result['overall']:.6f}%")


if __name__ == "__main__":
    raise SystemExit(main())
