from pathlib import Path

import numpy as np

from .handle_image_pixels import extract_pixels


def load_rgb_image(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        pixels, _ = extract_pixels(Path(image))
        arr = np.asarray(pixels, dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    raise ValueError("Image must be grayscale or RGB")


def align_rgb_images(first_image, second_image) -> tuple[np.ndarray, np.ndarray]:
    img_a = load_rgb_image(first_image)
    img_b = load_rgb_image(second_image)

    height = min(img_a.shape[0], img_b.shape[0])
    width = min(img_a.shape[1], img_b.shape[1])
    return img_a[:height, :width, :3], img_b[:height, :width, :3]
