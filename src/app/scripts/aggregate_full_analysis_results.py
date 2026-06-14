from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ALGORITHM_PREFIXES = (
    "aes_ctr",
    "aes_cbc",
    "triple_des",
    "chacha20",
    "logistic_map",
    "henon_map",
)

IMAGES_ROOT = Path(__file__).resolve().parents[1] / "shared" / "images"

CATEGORY_DIRS = [
    IMAGES_ROOT / "architectural_plan",
    IMAGES_ROOT / "cctv_footage",
    IMAGES_ROOT / "medical_scan",
    IMAGES_ROOT / "portrait",
    IMAGES_ROOT / "street_view",
    IMAGES_ROOT / "spatial",
]

OUTPUT_DIR = IMAGES_ROOT / "full_analysis_results"


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


def _render_like_reference(value: float, reference: Any) -> Any:
    if isinstance(reference, str):
        return f"{value:.6f}"
    if isinstance(reference, int) and not isinstance(reference, bool):
        if value == int(value):
            return int(value)
        return round(value, 6)
    return round(value, 6)


def _avg_nodes(nodes: list[Any]) -> Any:
    numeric_values = [_parse_numeric(n) for n in nodes]

    if all(v is not None for v in numeric_values):
        values = [v for v in numeric_values if v is not None]
        mean = sum(values) / len(values)
        return _render_like_reference(mean, nodes[0])

    first = nodes[0]

    if isinstance(first, dict):
        if not all(isinstance(n, dict) for n in nodes):
            return first
        all_keys: list[str] = list(first.keys())
        for node in nodes[1:]:
            for key in node:
                if key not in all_keys:
                    all_keys.append(key)
        result: dict[str, Any] = {}
        for key in all_keys:
            present = [n[key] for n in nodes if key in n]
            if present:
                result[key] = _avg_nodes(present)
        return result

    if isinstance(first, list):
        if not all(isinstance(n, list) for n in nodes):
            return first
        min_len = min(len(n) for n in nodes)
        return [_avg_nodes([n[i] for n in nodes]) for i in range(min_len)]

    return first


def _avg_file_for_algorithm(category_dir: Path, algorithm: str) -> Path | None:
    results_dir = category_dir / "aggregated_analysis_results"
    if not results_dir.exists():
        candidates = sorted(category_dir.glob("aggregated_analysis_results*"))
        if not candidates:
            return None
        results_dir = candidates[0]

    matches = list(results_dir.glob(f"{algorithm}_*_avg.json"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  Warning: multiple avg files for {algorithm} in {results_dir}, using first.")
    return matches[0]


def aggregate_full(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for algorithm in ALGORITHM_PREFIXES:
        records: list[tuple[str, Any]] = []

        for category_dir in CATEGORY_DIRS:
            if not category_dir.exists():
                print(f"  Skipping missing directory: {category_dir}")
                continue

            avg_file = _avg_file_for_algorithm(category_dir, algorithm)
            if avg_file is None:
                print(f"  No avg file for {algorithm} in {category_dir.name} — skipping.")
                continue

            with avg_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            records.append((category_dir.name, data))
            print(f"  Loaded {avg_file.name}  ({category_dir.name})")

        if not records:
            print(f"  No data found for {algorithm} — skipping.")
            continue

        just_data = [r[1] for r in records]
        full_avg = _avg_nodes(just_data)

        output = {
            "_meta": {
                "algorithm": algorithm,
                "categories_included": [r[0] for r in records],
                "n_categories": len(records),
                "description": (
                    "Cross-category average of per-category avg JSON files. "
                    "Each category contributes equally regardless of image count."
                ),
            },
            **full_avg,
        }

        out_path = output_dir / f"{algorithm}_full_avg.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

        print(f"  -> Saved {out_path.name}  ({len(records)} categories)")
        generated.append(out_path)

    return generated


def main() -> None:
    print(f"Images root : {IMAGES_ROOT}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print()

    generated = aggregate_full()

    print()
    print(f"Done. Generated {len(generated)} file(s) in {OUTPUT_DIR}")
    for path in generated:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
