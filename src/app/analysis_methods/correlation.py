from PIL import Image
import numpy as np
import argparse


def _load_image(path):
    img = Image.open(path)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _pearson(x, y):
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def correlation_between_images(path_orig, path_enc):
    """Compute correlation metrics between two images.

    Returns a dict with per-channel Pearson correlations and a grayscale correlation.
    Values near 0 indicate low linear correlation (desired for strong encryption).
    """
    a = _load_image(path_orig)
    b = _load_image(path_enc)
    if a.shape != b.shape:
        raise ValueError("Images must have the same dimensions and channels")

    # per-channel
    channels = ("R", "G", "B")
    per_channel = {}
    for i, name in enumerate(channels):
        per_channel[name] = _pearson(a[..., i], b[..., i])

    # grayscale (luminosity) correlation
    lum = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    a_gray = (a.astype(np.float64) * lum).sum(axis=2)
    b_gray = (b.astype(np.float64) * lum).sum(axis=2)
    gray_corr = _pearson(a_gray, b_gray)

    return {"per_channel": per_channel, "grayscale": gray_corr}


def horizontal_pixel_correlation(path_img):
    """Compute correlation between horizontally adjacent pixels.

    Correlates each pixel with its right neighbor (x, y) with (x+1, y).
    High values (close to 1) indicate structure; low values (close to 0)
    indicate randomness/encryption.

    Returns dict with per-channel and grayscale correlations.
    """
    img = _load_image(path_img)
    if img.shape[1] < 2:
        raise ValueError("Image width must be >= 2")

    # per-channel: pixel[i, j] vs pixel[i, j+1]
    channels = ("R", "G", "B")
    per_channel = {}
    for i, name in enumerate(channels):
        ch = img[..., i]
        # pixel values and shifted right by 1 column
        left = ch[:, :-1]  # all rows, all cols except last
        right = ch[:, 1:]  # all rows, all cols except first
        per_channel[name] = _pearson(left, right)

    # grayscale
    lum = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = (img.astype(np.float64) * lum).sum(axis=2)
    left = gray[:, :-1]
    right = gray[:, 1:]
    gray_corr = _pearson(left, right)

    return {"per_channel": per_channel, "grayscale": gray_corr}


def vertical_pixel_correlation(path_img):
    """Compute correlation between vertically adjacent pixels.

    Correlates each pixel with its bottom neighbor (x, y) with (x, y+1).
    High values (close to 1) indicate structure; low values (close to 0)
    indicate randomness/encryption.

    Returns dict with per-channel and grayscale correlations.
    """
    img = _load_image(path_img)
    if img.shape[0] < 2:
        raise ValueError("Image height must be >= 2")

    # per-channel: pixel[i, j] vs pixel[i+1, j]
    channels = ("R", "G", "B")
    per_channel = {}
    for i, name in enumerate(channels):
        ch = img[..., i]
        # pixel values and shifted down by 1 row
        top = ch[:-1, :]  # all rows except last, all cols
        bottom = ch[1:, :]  # all rows except first, all cols
        per_channel[name] = _pearson(top, bottom)

    # grayscale
    lum = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = (img.astype(np.float64) * lum).sum(axis=2)
    top = gray[:-1, :]
    bottom = gray[1:, :]
    gray_corr = _pearson(top, bottom)

    return {"per_channel": per_channel, "grayscale": gray_corr}


def diagonal_pixel_correlation(path_img):
    """Compute correlation between diagonally adjacent pixels.

    Correlates each pixel with its bottom-right neighbor (x, y) with (x+1, y+1).
    High values (close to 1) indicate structure; low values (close to 0)
    indicate randomness/encryption.

    Returns dict with per-channel and grayscale correlations.
    """
    img = _load_image(path_img)
    if img.shape[0] < 2 or img.shape[1] < 2:
        raise ValueError("Image width and height must be >= 2")

    # per-channel: pixel[i, j] vs pixel[i+1, j+1]
    channels = ("R", "G", "B")
    per_channel = {}
    for i, name in enumerate(channels):
        ch = img[..., i]
        top_left = ch[:-1, :-1]
        bottom_right = ch[1:, 1:]
        per_channel[name] = _pearson(top_left, bottom_right)

    # grayscale
    lum = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = (img.astype(np.float64) * lum).sum(axis=2)
    top_left = gray[:-1, :-1]
    bottom_right = gray[1:, 1:]
    gray_corr = _pearson(top_left, bottom_right)

    return {"per_channel": per_channel, "grayscale": gray_corr}


def main(argv=None):
    p = argparse.ArgumentParser(description="Compute correlation metrics for images")
    sub = p.add_subparsers(dest="cmd", help="command")

    # Image-to-image correlation
    sp1 = sub.add_parser("compare", help="Compare two images")
    sp1.add_argument("original", help="Path to original/plain image")
    sp1.add_argument("encrypted", help="Path to encrypted image")

    # Horizontal pixel correlation
    sp2 = sub.add_parser("horizontal", help="Compute horizontal pixel correlation")
    sp2.add_argument("image", help="Path to image")

    # Vertical pixel correlation
    sp3 = sub.add_parser("vertical", help="Compute vertical pixel correlation")
    sp3.add_argument("image", help="Path to image")

    # Diagonal pixel correlation
    sp4 = sub.add_parser("diagonal", help="Compute diagonal pixel correlation")
    sp4.add_argument("image", help="Path to image")

    args = p.parse_args(argv)

    if not args.cmd:
        p.print_help()
        return 1

    try:
        if args.cmd == "compare":
            res = correlation_between_images(args.original, args.encrypted)
            print("Correlation results (plain vs encrypted):")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  grayscale: {res['grayscale']:.6f}")
        elif args.cmd == "horizontal":
            res = horizontal_pixel_correlation(args.image)
            print("Horizontal pixel correlation:")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  grayscale: {res['grayscale']:.6f}")
        elif args.cmd == "vertical":
            res = vertical_pixel_correlation(args.image)
            print("Vertical pixel correlation:")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  grayscale: {res['grayscale']:.6f}")
        elif args.cmd == "diagonal":
            res = diagonal_pixel_correlation(args.image)
            print("Diagonal pixel correlation:")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  grayscale: {res['grayscale']:.6f}")
    except Exception as e:
        print("Error:", e)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
