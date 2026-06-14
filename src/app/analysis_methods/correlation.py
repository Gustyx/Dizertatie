import numpy as np
import argparse

from ..utils import align_rgb_images, load_rgb_image

MAX_ANALYSIS_SIZE = (2048, 2048)


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
    a, b = align_rgb_images(load_rgb_image(path_orig), load_rgb_image(path_enc))

    # per-channel
    channels = ("R", "G", "B")
    per_channel = {}
    for i, name in enumerate(channels):
        per_channel[name] = _pearson(a[..., i], b[..., i])

    overall_corr = float(np.max(np.abs(list(per_channel.values()))))

    return {"per_channel": per_channel, "overall": overall_corr}


def horizontal_pixel_correlation(path_img):
    img = load_rgb_image(path_img)
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

    overall_corr = float(np.max(np.abs(list(per_channel.values()))))

    return {"per_channel": per_channel, "overall": overall_corr}


def vertical_pixel_correlation(path_img):
    img = load_rgb_image(path_img)
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

    overall_corr = float(np.max(np.abs(list(per_channel.values()))))

    return {"per_channel": per_channel, "overall": overall_corr}


def diagonal_pixel_correlation(path_img):
    img = load_rgb_image(path_img)
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

    overall_corr = float(np.max(np.abs(list(per_channel.values()))))

    return {"per_channel": per_channel, "overall": overall_corr}


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
            print(f"  overall: {res['overall']:.6f}")
        elif args.cmd == "horizontal":
            res = horizontal_pixel_correlation(args.image)
            print("Horizontal pixel correlation:")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  overall: {res['overall']:.6f}")
        elif args.cmd == "vertical":
            res = vertical_pixel_correlation(args.image)
            print("Vertical pixel correlation:")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  overall: {res['overall']:.6f}")
        elif args.cmd == "diagonal":
            res = diagonal_pixel_correlation(args.image)
            print("Diagonal pixel correlation:")
            for ch, v in res["per_channel"].items():
                print(f"  {ch}: {v:.6f}")
            print(f"  overall: {res['overall']:.6f}")
    except Exception as e:
        print("Error:", e)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
