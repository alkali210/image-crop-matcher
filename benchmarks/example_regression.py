import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.determinism import DEFAULT_SEED, seed_opencv
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.imaging import read_image
from crop_matcher.matcher import ImageMatcher


@dataclass(frozen=True, slots=True)
class RegressionCase:
    query_name: str
    expected_path: str
    image: np.ndarray


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in regression manifest: {key}")
        result[key] = value
    return result


def load_cases(
    examples: Path,
    catalog: ImageCatalog,
    max_image_pixels: int,
) -> list[RegressionCase]:
    manifest_path = examples / "expected.json"
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed regression manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise TypeError(f"Regression manifest must be a JSON object: {manifest_path}")
    if not manifest:
        raise ValueError(f"Regression manifest must contain at least one case: {manifest_path}")

    gallery_paths = {record.relative_path.as_posix() for record in catalog.records}
    cases: list[RegressionCase] = []
    examples_root = examples.resolve()
    for query_name, expected_path in manifest.items():
        if (
            not isinstance(query_name, str)
            or not query_name.strip()
            or not isinstance(expected_path, str)
            or not expected_path.strip()
        ):
            raise TypeError("Regression manifest entries must be nonempty string pairs")
        query_relative = Path(query_name)
        if query_relative.is_absolute() or ".." in query_relative.parts:
            raise ValueError(f"Regression query must be examples-relative: {query_name}")
        query_path = examples / query_relative
        try:
            query_path.resolve(strict=True).relative_to(examples_root)
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(f"Regression query does not exist: {query_name}") from exc
        if not query_path.is_file():
            raise FileNotFoundError(f"Regression query does not exist: {query_name}")
        if expected_path not in gallery_paths:
            raise ValueError(f"Expected path is not in the gallery: {expected_path}")
        cases.append(
            RegressionCase(
                query_name=query_name,
                expected_path=expected_path,
                image=read_image(query_path, max_image_pixels),
            )
        )
    return cases


def run_regression(gallery: Path, examples: Path, max_p95_ms: float) -> int:
    seed_opencv(DEFAULT_SEED)
    settings = Settings(gallery_dir=gallery)
    catalog = ImageCatalog.scan(settings.gallery_dir, settings.max_image_pixels)
    cases = load_cases(examples, catalog, settings.max_image_pixels)
    index = FeatureIndex.load_or_build(catalog, settings)
    matcher = ImageMatcher(catalog, index, settings)

    matcher.match(cases[0].image)

    correct_count = 0
    latencies_ms: list[float] = []
    for case in cases:
        started = perf_counter_ns()
        result = matcher.match(case.image)
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        latencies_ms.append(latency_ms)
        predicted_path = result.record.relative_path.as_posix()
        correct = predicted_path == case.expected_path
        correct_count += int(correct)
        print(
            f"query={case.query_name} expected={case.expected_path} "
            f"predicted={predicted_path} method={result.method} "
            f"similarity={result.similarity:.1f} latency_ms={latency_ms:.3f} "
            f"correct={str(correct).lower()}"
        )

    case_count = len(cases)
    accuracy = correct_count / case_count
    p95_ms = float(np.percentile(latencies_ms, 95))
    print(f"top1={correct_count}/{case_count} accuracy={accuracy:.2%} latency_ms_p95={p95_ms:.3f}")
    return int(correct_count != case_count or p95_ms > max_p95_ms)


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-example crop matching regression cases")
    parser.add_argument("--gallery", type=Path, default=Path("gallery"))
    parser.add_argument("--examples", type=Path, default=Path("examples"))
    parser.add_argument("--max-p95-ms", type=_nonnegative_float, default=1000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return run_regression(arguments.gallery, arguments.examples, arguments.max_p95_ms)


if __name__ == "__main__":
    raise SystemExit(main())
