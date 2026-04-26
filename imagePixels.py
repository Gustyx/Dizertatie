import numpy as np
from pathlib import Path
from PIL import Image


def extract_pixels(image_path: Path) -> np.ndarray:
    """Load an image from disk and return its pixels as a NumPy array."""

    with Image.open(image_path) as img:
        return np.array(img)


def shift_pixel_values(pixel_array: np.ndarray) -> np.ndarray:
    """Invert pixel values using the mapping 0->255, 1->254, ..., 255->0."""

    pixel_uint8 = pixel_array.astype(np.uint8)

    # Preserve alpha channel for RGBA images so output is not fully transparent.
    if pixel_uint8.ndim == 3 and pixel_uint8.shape[2] == 4:
        shifted = pixel_uint8.copy()
        shifted[..., :3] = 255 - shifted[..., :3]
        return shifted

    return 255 - pixel_uint8


def reconstruct_image(pixel_array: np.ndarray, output_path: Path) -> None:
    """Create and save an image file from a pixel array."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixel_array.astype(np.uint8))
    image.save(output_path)


def main() -> None:
    # Folder containing the image(s), resolved from this script's directory
    images_dir = Path(__file__).resolve().parent / "images" / "plain"
    image_files = [
        p
        for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
    ]

    if not image_files:
        raise FileNotFoundError(f"No image files found in: {images_dir.resolve()}")

    image_path = image_files[0]
    pixel_matrix = extract_pixels(image_path)
    shifted_pixels = shift_pixel_values(pixel_matrix)

    output_path = (
        Path(__file__).resolve().parent
        / "images"
        / "encrypted"
        / f"shifted_{image_path.name}"
    )
    reconstruct_image(shifted_pixels, output_path)

    print(f"Image: {image_path.name}")
    print(pixel_matrix)
    print(f"Saved shifted image to: {output_path}")


if __name__ == "__main__":
    main()
