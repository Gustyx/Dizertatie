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


def main() -> None:
    pass


if __name__ == "__main__":
    main()
