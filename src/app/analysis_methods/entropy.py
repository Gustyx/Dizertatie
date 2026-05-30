import argparse
import numpy as np
from pathlib import Path

from ..utils.imagePixels import extract_preview_pixels

MAX_ANALYSIS_SIZE = (2048, 2048)


def _load_image(image):
    if isinstance(image, (str, Path)):
        pixels, _ = extract_preview_pixels(Path(image), max_size=MAX_ANALYSIS_SIZE)
        return np.asarray(pixels, dtype=np.uint8)
    return np.asarray(image)


def _shannon_entropy_uint8(values):
    """Compute Shannon entropy (base-2) for uint8 intensity values."""
    flat = np.asarray(values, dtype=np.uint8).ravel()
    if flat.size == 0:
        return float("nan")

    hist = np.bincount(flat, minlength=256).astype(np.float64)
    probabilities = hist / float(flat.size)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def pixel_entropy(image):
    """Compute global pixel entropy.

    Returns a dict:
    - grayscale image: {"grayscale": H}
    - RGB image: {"per_channel": {"R": Hr, "G": Hg, "B": Hb}, "overall": Ho}
    """
    img = _load_image(image)

    if img.ndim == 2:
        return {"grayscale": _shannon_entropy_uint8(img)}

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Image must be grayscale or RGB")

    img = img[..., :3]
    per_channel = {}
    for i, name in enumerate(("R", "G", "B")):
        per_channel[name] = _shannon_entropy_uint8(img[..., i])

    # Entropy of all channel samples taken together
    overall = _shannon_entropy_uint8(img.reshape(-1))

    return {"per_channel": per_channel, "overall": overall}


def normalized_pixel_entropy(image):
    """Compute entropy normalized to [0, 1] by dividing by max 8 bits."""
    result = pixel_entropy(image)
    if "grayscale" in result:
        return {"grayscale": float(result["grayscale"] / 8.0)}

    return {
        "per_channel": {
            channel: float(value / 8.0)
            for channel, value in result["per_channel"].items()
        },
        "overall": float(result["overall"] / 8.0),
    }


def block_entropy(image, block_size=8):
    """Compute average Shannon entropy over non-overlapping blocks.

    Useful for measuring local randomness instead of only whole-image randomness.
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    img = _load_image(image)

    if img.ndim == 2:
        h, w = img.shape
        h_trim = (h // block_size) * block_size
        w_trim = (w // block_size) * block_size
        if h_trim == 0 or w_trim == 0:
            raise ValueError("Image is smaller than the block size")

        trimmed = img[:h_trim, :w_trim]
        entropies = []
        for y in range(0, h_trim, block_size):
            for x in range(0, w_trim, block_size):
                block = trimmed[y : y + block_size, x : x + block_size]
                entropies.append(_shannon_entropy_uint8(block))

        return {
            "grayscale": float(np.mean(entropies)),
            "blocks": len(entropies),
            "used_size": (h_trim, w_trim),
        }

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Image must be grayscale or RGB")

    img = img[..., :3]
    h, w, _ = img.shape
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    if h_trim == 0 or w_trim == 0:
        raise ValueError("Image is smaller than the block size")

    trimmed = img[:h_trim, :w_trim]
    per_channel = {}
    for channel_index, channel_name in enumerate(("R", "G", "B")):
        entropies = []
        channel = trimmed[..., channel_index]
        for y in range(0, h_trim, block_size):
            for x in range(0, w_trim, block_size):
                block = channel[y : y + block_size, x : x + block_size]
                entropies.append(_shannon_entropy_uint8(block))
        per_channel[channel_name] = float(np.mean(entropies))

    return {
        "per_channel": per_channel,
        "overall": float(np.mean(list(per_channel.values()))),
        "blocks": int((h_trim // block_size) * (w_trim // block_size)),
        "used_size": (h_trim, w_trim),
    }


def pixel_entropy_with_blocks(image, block_sizes=(8, 16, 32)):
    """Compute global entropy and block entropy summaries together."""
    result = pixel_entropy(image)
    block_results = {}

    for block_size in block_sizes:
        try:
            block_results[int(block_size)] = block_entropy(image, block_size=block_size)
        except ValueError as exc:
            block_results[int(block_size)] = {"error": str(exc)}

    result["block_entropies"] = block_results
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute entropy metrics for images")
    parser.add_argument("image", help="Path to image")
    parser.add_argument(
        "--mode",
        choices=("global", "normalized", "block"),
        default="global",
        help="Entropy algorithm to use",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Block size used by --mode block",
    )
    args = parser.parse_args(argv)

    if args.mode == "global":
        result = pixel_entropy(args.image)
    elif args.mode == "normalized":
        result = normalized_pixel_entropy(args.image)
    else:
        result = block_entropy(args.image, block_size=args.block_size)

    if "grayscale" in result:
        print(f"Entropy: {result['grayscale']:.6f}")
    else:
        print("Entropy per channel:")
        for channel, value in result["per_channel"].items():
            print(f"  {channel}: {value:.6f}")
        print(f"  overall: {result['overall']:.6f}")

    if "blocks" in result:
        print(
            f"Blocks used: {result['blocks']} ({result['used_size'][0]}x{result['used_size'][1]} area)"
        )

    if "block_entropies" in result:
        for block_size, block_result in result["block_entropies"].items():
            print(f"Block entropy ({block_size}x{block_size}):")
            if "grayscale" in block_result:
                print(f"  Grayscale: {block_result['grayscale']:.6f}")
            else:
                print("  Per channel:")
                for channel, value in block_result["per_channel"].items():
                    print(f"    {channel}: {value:.6f}")
                print(f"    overall: {block_result['overall']:.6f}")
            print(
                f"  Blocks used: {block_result['blocks']} ({block_result['used_size'][0]}x{block_result['used_size'][1]} area)"
            )


if __name__ == "__main__":
    raise SystemExit(main())
