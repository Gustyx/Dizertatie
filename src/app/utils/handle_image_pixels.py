import numpy as np
from pathlib import Path
from contextlib import contextmanager
from PIL import Image


@contextmanager
def _allow_large_images():
    """Temporarily disable PIL's decompression bomb limit while loading known datasets."""
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def extract_pixels(image_path: Path) -> tuple[np.ndarray, str]:
    """Load an image and return its pixel array plus mode."""
    with _allow_large_images():
        with Image.open(image_path) as img:
            return np.array(img), img.mode


def extract_preview_pixels(
    image_path: Path,
    max_size: tuple[int, int] = (2048, 2048),
) -> tuple[np.ndarray, str]:
    """Load a downscaled preview image and return its pixel array plus mode."""
    with _allow_large_images():
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            preview = img.copy()
            return np.array(preview), preview.mode


def reconstruct_image(pixel_array: np.ndarray, mode: str, output_path: Path) -> None:
    """Create and save an image from a pixel array using the original mode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixel_array.astype(np.uint8), mode=mode)
    image.save(output_path)


def add_one_bit_to_pixel(image_path: Path) -> Path:
    pixels, mode = extract_pixels(image_path)
    if pixels.ndim == 2:
        pixels[0, 0] = (int(pixels[0, 0]) + 1) % 256
    elif pixels.ndim == 3 and pixels.shape[2] >= 1:
        pixels[0, 0, 0] = (int(pixels[0, 0, 0]) + 1) % 256
    else:
        raise ValueError("Image must be grayscale or RGB")
    output_path = image_path.parent.parent / "plain_plus_one_bit" / image_path.name
    reconstruct_image(pixels, mode, output_path)

    return output_path


def main() -> None:
    pass


if __name__ == "__main__":
    main()
