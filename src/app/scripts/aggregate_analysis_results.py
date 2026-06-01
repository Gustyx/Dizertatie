from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

ALGORITHM_PREFIXES = (
    "aes_ctr",
    "aes_cbc",
    "aes_gcm",
    "aes_ccm",
    "triple_des",
    "chacha20",
    "logistic_map",
    "henon_map",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_numeric(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _render_like_reference(value: float, reference: Any, operation: str) -> Any:
    if isinstance(reference, str):
        return f"{value:.6f}"

    if isinstance(reference, int) and not isinstance(reference, bool):
        if operation in {"min", "max"}:
            return int(round(value))
        if value.is_integer():
            return int(value)
        return round(value, 6)

    if isinstance(reference, float):
        return round(value, 6)

    return round(value, 6)


def _aggregate_nodes(nodes: list[Any], operation: str) -> Any:
    numeric_values = [_parse_numeric(node) for node in nodes]
    if all(value is not None for value in numeric_values):
        values = [value for value in numeric_values if value is not None]
        if operation == "min":
            return _render_like_reference(min(values), nodes[0], operation)
        if operation == "max":
            return _render_like_reference(max(values), nodes[0], operation)
        if operation == "avg":
            return _render_like_reference(
                sum(values) / len(values), nodes[0], operation
            )
        if operation == "std":
            if len(values) <= 1:
                return _render_like_reference(0.0, nodes[0], operation)
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            return _render_like_reference(math.sqrt(variance), nodes[0], operation)
        raise ValueError(f"Unsupported operation: {operation}")

    first = nodes[0]

    if isinstance(first, dict):
        if not all(isinstance(node, dict) for node in nodes):
            return first

        ordered_keys = list(first.keys())
        for node in nodes[1:]:
            for key in node.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)

        result: dict[str, Any] = {}
        for key in ordered_keys:
            present_values = [node[key] for node in nodes if key in node]
            if not present_values:
                continue
            result[key] = _aggregate_nodes(present_values, operation)
        return result

    if isinstance(first, list):
        if not all(isinstance(node, list) for node in nodes):
            return first

        min_len = min(len(node) for node in nodes)
        return [
            _aggregate_nodes([node[index] for node in nodes], operation)
            for index in range(min_len)
        ]

    return first


def _extract_group_name(file_path: Path) -> str | None:
    stem = file_path.stem
    if not stem.endswith("_analysis"):
        return None

    base_name = stem[: -len("_analysis")]

    # Expected base pattern: <algorithm>_<dataset_prefix>_<image_id>
    # Example: aes_cbc_architectural_plan_1
    for algorithm in sorted(ALGORITHM_PREFIXES, key=len, reverse=True):
        algorithm_prefix = f"{algorithm}_"
        if not base_name.startswith(algorithm_prefix):
            continue

        tail = base_name[len(algorithm_prefix) :]
        if "_" not in tail:
            return None

        dataset_prefix, _image_token = tail.rsplit("_", 1)
        if not dataset_prefix:
            return None
        return f"{algorithm}_{dataset_prefix}"

    return None


def _iter_analysis_files(root_dir: Path) -> Iterable[Path]:
    return sorted(root_dir.rglob("*_analysis.json"))


def aggregate_analysis_files(root_dir: Path) -> list[Path]:
    grouped_files: dict[tuple[Path, str], list[Path]] = {}

    for json_file in _iter_analysis_files(root_dir):
        group_name = _extract_group_name(json_file)
        if group_name is None:
            continue

        output_dir = json_file.parent.parent / "results_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        group_key = (output_dir, group_name)
        grouped_files.setdefault(group_key, []).append(json_file)

    generated_files: list[Path] = []

    for (output_dir, group_name), files in sorted(grouped_files.items()):
        records = []
        for file_path in sorted(files):
            with file_path.open("r", encoding="utf-8") as fh:
                records.append(json.load(fh))

        if not records:
            continue

        min_data = _aggregate_nodes(records, "min")
        max_data = _aggregate_nodes(records, "max")
        avg_data = _aggregate_nodes(records, "avg")
        std_data = _aggregate_nodes(records, "std")

        for suffix, payload in (
            ("min", min_data),
            ("max", max_data),
            ("avg", avg_data),
            ("std", std_data),
        ):
            out_path = output_dir / f"{group_name}_{suffix}.json"
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            generated_files.append(out_path)

    return generated_files


def _default_root_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "shared" / "images"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate *_analysis.json files into *_min.json, *_max.json, *_avg.json "
            "for each algorithm and dataset prefix."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_root_dir(),
        help="Root folder to scan recursively (default: src/app/shared/images).",
    )
    args = parser.parse_args()

    root_dir = args.root.resolve()
    generated = aggregate_analysis_files(root_dir)

    print(f"Generated {len(generated)} files.")
    # for path in generated:
    # print(path.name)


if __name__ == "__main__":
    main()
