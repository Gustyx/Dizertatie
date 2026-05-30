import os
import shutil
from pathlib import Path

SOURCE_DIR = Path(r"src/app/shared/images/plain/Newfolder/dataset")
TARGET_DIR = Path(r"src/app/shared/images/plain/street_view")
TARGET_PREFIX = "street_view_"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sort_key(path: Path) -> tuple[int, str]:
    """Sort numeric filenames first, then fall back to lexicographic order."""
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files.sort(key=sort_key)

    for index, source_path in enumerate(image_files[:500]):
        target_path = (
            TARGET_DIR / f"{TARGET_PREFIX}{index + 1}{source_path.suffix.lower()}"
        )
        shutil.move(str(source_path), str(target_path))

    print(f"Moved {len(image_files[:500])} images from {SOURCE_DIR} to {TARGET_DIR}.")


if __name__ == "__main__":
    main()
