import os
import numpy as np
from PIL import Image


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a numeric array to uint8 (0-255)."""
    if np.issubdtype(arr.dtype, np.floating):
        amin = float(arr.min())
        amax = float(arr.max())
        if amin == amax:
            return np.zeros(arr.shape, dtype=np.uint8)
        norm = (arr - amin) / (amax - amin)
        return (norm * 255).astype(np.uint8)

    # integer types: clip to 0-255 and cast
    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    data = np.load(r"src/app/scripts/medical_scans.npz", allow_pickle=True)

    images = data["image"]

    out_dir = "src/app/shared/images/plain/medical_scans"
    os.makedirs(out_dir, exist_ok=True)

    for index, img in enumerate(images):
        arr = np.asarray(img)

        # If channels are first (3, H, W), convert to (H, W, 3)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))

        out = normalize_to_uint8(arr)

        out_path = os.path.join(out_dir, f"medical_scan_{index + 1}.png")
        Image.fromarray(out).save(out_path)


if __name__ == "__main__":
	main()
