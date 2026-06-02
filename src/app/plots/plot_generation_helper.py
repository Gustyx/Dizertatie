from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ENCODED_PATTERN = re.compile(r"^(?P<algorithm>[a-z0-9_]+)_enc_(?P<plain_stem>.+)$")


def _iter_image_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _collect_dataset_roots(images_root: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in sorted(images_root.iterdir()):
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def _find_plain_directories(dataset_root: Path) -> list[Path]:
    result: list[Path] = []
    for name in ("plain", "plain1"):
        candidate = dataset_root / name
        if candidate.exists() and candidate.is_dir():
            result.append(candidate)
    return result


def _find_encrypted_directory(
    dataset_root: Path, name: str = "encrypted"
) -> Path | None:
    candidate = dataset_root / name
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _group_encrypted_files(encrypted_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image_path in _iter_image_files(encrypted_dir):
        match = ENCODED_PATTERN.match(image_path.stem)
        if match is None:
            continue
        grouped.setdefault(match.group("plain_stem"), []).append(image_path)
    return grouped
