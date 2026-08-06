import argparse
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from crop_matcher.catalog import ImageCatalog, ImageRecord
from crop_matcher.config import Settings
from crop_matcher.determinism import DEFAULT_SEED, seed_opencv
from crop_matcher.feature_index import FeatureIndex
from crop_matcher.imaging import read_image
from crop_matcher.matcher import ImageMatcher


OUTPUT_SIZES = (64, 90, 128, 192)
ACCURACY_TARGET = 0.95
OWNERSHIP_MARKER = ".benchmark-owned"
OWNERSHIP_CONTENT = "crop-matcher-benchmark-failures-v1\n"


@dataclass(frozen=True, slots=True)
class CropSpec:
    image_index: int
    sample_index: int
    crop_fraction: float
    x_fraction: float
    y_fraction: float
    output_size: int
    grayscale: bool


def crop_specs(
    seed: int = DEFAULT_SEED,
    image_count: int = 0,
    samples_per_image: int = 4,
) -> Iterator[CropSpec]:
    rng = random.Random(seed)
    for image_index in range(image_count):
        for sample_index in range(samples_per_image):
            yield CropSpec(
                image_index=image_index,
                sample_index=sample_index,
                crop_fraction=rng.uniform(0.10, 0.40),
                x_fraction=rng.random(),
                y_fraction=rng.random(),
                output_size=rng.choice(OUTPUT_SIZES),
                grayscale=(image_index * samples_per_image + sample_index) % 2 == 1,
            )


def prepare_query(source: np.ndarray, spec: CropSpec) -> tuple[np.ndarray, int, int, int]:
    height, width = source.shape[:2]
    side = max(1, min(min(height, width), round(min(height, width) * spec.crop_fraction)))
    x = round((width - side) * spec.x_fraction)
    y = round((height - side) * spec.y_fraction)
    crop = source[y : y + side, x : x + side]
    interpolation = cv2.INTER_AREA if spec.output_size <= side else cv2.INTER_CUBIC
    query = cv2.resize(crop, (spec.output_size, spec.output_size), interpolation=interpolation)
    if spec.grayscale:
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        query = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)
    return query, x, y, side


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def write_failure(
    failure_dir: Path,
    query_number: int,
    *,
    source_id: str,
    predicted_id: str,
    x: int,
    y: int,
    side: int,
    spec: CropSpec,
    similarity: float,
    latency_ms: float,
) -> Path:
    metadata = {
        "source_id": source_id,
        "predicted_id": predicted_id,
        "crop_x": x,
        "crop_y": y,
        "crop_side": side,
        "output_size": spec.output_size,
        "grayscale": spec.grayscale,
        "similarity": similarity,
        "latency_ms": round(latency_ms, 3),
    }
    path = failure_dir / f"failure-{query_number:06d}.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def failure_run_dir(
    failure_dir: Path,
    gallery: Path,
    samples_per_image: int,
    max_images: int | None,
    seed: int,
) -> Path:
    identity = {
        "gallery": gallery.resolve().as_posix(),
        "samples_per_image": samples_per_image,
        "max_images": max_images,
        "seed": seed,
    }
    digest = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return failure_dir / f"run-{digest}"


def prepare_failure_run_dir(
    failure_dir: Path,
    gallery: Path,
    samples_per_image: int,
    max_images: int | None,
    seed: int,
) -> Path:
    failure_dir.mkdir(parents=True, exist_ok=True)
    run_dir = failure_run_dir(
        failure_dir,
        gallery,
        samples_per_image,
        max_images,
        seed,
    )
    marker = run_dir / OWNERSHIP_MARKER
    if run_dir.exists():
        if not marker.is_file() or marker.read_text("utf-8") != OWNERSHIP_CONTENT:
            raise RuntimeError(f"Failure run directory is not benchmark-owned: {run_dir}")
    else:
        run_dir.mkdir()
        marker.write_text(OWNERSHIP_CONTENT, encoding="utf-8")
    for old_failure in run_dir.glob("failure-*.json"):
        old_failure.unlink()
    return run_dir


def accuracy_exit_status(accuracy: float) -> int:
    return int(accuracy < ACCURACY_TARGET)


def _limited_catalog(catalog: ImageCatalog, max_images: int | None) -> ImageCatalog:
    if max_images is None:
        return catalog
    return ImageCatalog(
        catalog.root,
        catalog.records[:max_images],
        catalog.manifest[:max_images],
    )


def _bounded_cache_dir(catalog: ImageCatalog, settings: Settings, max_images: int) -> Path:
    identity = {
        "gallery": catalog.root.as_posix(),
        "max_images": max_images,
        "feature_settings": {
            "working_max_edge": settings.working_max_edge,
            "sift_features": settings.sift_features,
            "sift_contrast_threshold": settings.sift_contrast_threshold,
            "tile_sizes": list(settings.tile_sizes),
            "coarse_template_edge": settings.coarse_template_edge,
        },
        "images": [asdict(entry) for entry in catalog.manifest],
    }
    digest = sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return settings.cache_dir / "benchmarks" / f"bounded-{digest}"


def _query_for(
    record: ImageRecord,
    spec: CropSpec,
    max_image_pixels: int,
) -> tuple[np.ndarray, int, int, int]:
    source = read_image(record.path, max_image_pixels)
    return prepare_query(source, spec)


def run_benchmark(
    gallery: Path,
    samples_per_image: int,
    max_images: int | None,
    seed: int,
    failure_dir: Path,
) -> int:
    seed_opencv(seed)
    run_failure_dir = prepare_failure_run_dir(
        failure_dir,
        gallery,
        samples_per_image,
        max_images,
        seed,
    )
    settings = Settings(gallery_dir=gallery)
    catalog = _limited_catalog(
        ImageCatalog.scan(settings.gallery_dir, settings.max_image_pixels),
        max_images,
    )
    if not catalog.records:
        raise ValueError(f"Gallery contains no supported source images: {gallery}")
    if max_images is not None:
        settings = replace(
            settings,
            cache_dir=_bounded_cache_dir(catalog, settings, max_images),
        )

    index = FeatureIndex.load_or_build(catalog, settings)
    matcher = ImageMatcher(catalog, index, settings)
    specs = list(crop_specs(seed, len(catalog.records), samples_per_image))

    warmup_record = catalog.get(catalog.records[0].image_id)
    warmup, *_ = _query_for(warmup_record, specs[0], settings.max_image_pixels)
    matcher.match(warmup)

    correct = 0
    method_counts = {"sift": 0, "template": 0}
    latencies_ms: list[float] = []
    for query_number, spec in enumerate(specs, start=1):
        record = catalog.records[spec.image_index]
        record = catalog.get(record.image_id)
        query, x, y, side = _query_for(record, spec, settings.max_image_pixels)
        started = time.perf_counter_ns()
        result = matcher.match(query)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        latencies_ms.append(latency_ms)
        method_counts[result.method] += 1
        if result.record.image_id == record.image_id:
            correct += 1
            continue
        write_failure(
            run_failure_dir,
            query_number,
            source_id=record.image_id,
            predicted_id=result.record.image_id,
            x=x,
            y=y,
            side=side,
            spec=spec,
            similarity=result.similarity,
            latency_ms=latency_ms,
        )

    query_count = len(specs)
    accuracy = correct / query_count
    p50, p95 = np.percentile(latencies_ms, [50, 95])
    print(f"images={len(catalog.records)} queries={query_count}")
    print(f"top1={correct}/{query_count} accuracy={accuracy:.2%}")
    print(f"method_sift={method_counts['sift']} method_template={method_counts['template']}")
    print(f"latency_ms_p50={p50:.3f} latency_ms_p95={p95:.3f}")
    return accuracy_exit_status(accuracy)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic source-image crop matching"
    )
    parser.add_argument("--gallery", type=Path, default=Path("songs"))
    parser.add_argument("--samples-per-image", type=_positive_int, default=4)
    parser.add_argument("--max-images", type=_positive_int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--failure-dir", type=Path, default=Path("benchmark-failures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return run_benchmark(
        gallery=arguments.gallery,
        samples_per_image=arguments.samples_per_image,
        max_images=arguments.max_images,
        seed=arguments.seed,
        failure_dir=arguments.failure_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
