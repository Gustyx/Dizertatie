import numpy as np
from pathlib import Path
from PIL import Image


def extract_pixels(image_path: Path) -> tuple[np.ndarray, str]:
    """Load an image and return its pixel array plus mode."""

    with Image.open(image_path) as img:
        return np.array(img), img.mode


def reconstruct_image(pixel_array: np.ndarray, mode: str, output_path: Path) -> None:
    """Create and save an image from a pixel array using the original mode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixel_array.astype(np.uint8), mode=mode)
    image.save(output_path)


def add_one_bit_to_pixel(image_path: Path) -> None:
    pixels, mode = extract_pixels(image_path)
    pixels[0, 0] += 1
    reconstruct_image(pixels, mode, Path("images/plain/original2.png"))


def main() -> None:
    pixels, mode = extract_pixels(
        Path("images/encrypted/chacha20_encrypted_original.png")
    )
    print(f"Extracted pixels {pixels[0, 0]}")
    print(f"Extracted pixels {pixels[0, 1]}")
    print(f"Extracted pixels {pixels[0, 2]}")
    print(f"Extracted pixels {pixels[0, 3]}")
    print("-----------------------------------------")
    pixels, mode = extract_pixels(
        Path("images/encrypted/chacha20_encrypted_original2.png")
    )
    print(f"Extracted pixels {pixels[0, 0]}")
    print(f"Extracted pixels {pixels[0, 1]}")
    print(f"Extracted pixels {pixels[0, 2]}")
    print(f"Extracted pixels {pixels[0, 3]}")
    # pixels[0, 0] += 1
    # reconstruct_image(pixels, mode, Path("images/plain/original2.png"))
    pass


if __name__ == "__main__":
    main()
