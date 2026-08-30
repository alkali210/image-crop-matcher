import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex


def write_textured_image(path: Path, offset: int, shape: tuple[int, int] = (160, 160)) -> None:
    image = np.zeros((*shape, 3), np.uint8)
    cv2.putText(
        image,
        f"MATCH-{offset}",
        (8, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.circle(image, (80 + offset, 110), 24, (80, 180, 240), 3)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.tobytes())


def write_blank_image(path: Path, shape: tuple[int, int] = (40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".png", np.zeros((*shape, 3), np.uint8))
    assert ok
    path.write_bytes(payload.tobytes())


def feature_dir(settings: Settings) -> Path:
    metadata = json.loads((settings.cache_dir / "manifest.json").read_text("utf-8"))
    return settings.cache_dir / metadata["feature_index_dir"]


def replace_array(
    settings: Settings,
    name: str,
    update: Callable[[np.ndarray], None] | np.ndarray,
) -> None:
    path = feature_dir(settings) / f"{name}.npy"
    array = np.load(path, allow_pickle=False).copy()
    if callable(update):
        update(array)
    else:
        array = update
    np.save(path, array, allow_pickle=False)


def test_builds_uint8_visual_index_and_round_trips_mmap_cache(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    write_textured_image(gallery / "two" / "base.jpg", 10)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.loaded_from_cache is False
    assert loaded.loaded_from_cache is True
    assert built.descriptors.dtype == np.uint8
    assert built.descriptors.shape[0] > 0
    assert isinstance(loaded.descriptors, np.memmap)
    assert isinstance(loaded.points, np.memmap)
    assert isinstance(loaded.coarse_templates.pixels, np.memmap)
    assert set(built.by_image) == {record.image_id for record in catalog.records}
    assert built.representative_descriptors.dtype == np.float32
    assert built.representative_image_indices.dtype == np.int32
    assert len(built.representative_descriptors) <= 128 * len(catalog.records)
    assert np.array_equal(loaded.descriptors, built.descriptors)
    assert np.array_equal(
        loaded.representative_descriptors,
        built.representative_descriptors,
    )
    assert not (settings.cache_dir / "features.npz").exists()


def test_visual_index_ranks_source_image_first(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    write_textured_image(gallery / "two" / "base.jpg", 35)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    index = FeatureIndex.load_or_build(catalog, settings)

    expected_id = index.image_ids[1]
    query_descriptors = index.by_image[expected_id].descriptors
    ranking = index.rank_image_indices(query_descriptors)

    assert index.image_ids[int(ranking[0])] == expected_id


def test_uint8_bf_l2_matches_float32_exactly() -> None:
    rng = np.random.default_rng(7)
    train = rng.integers(0, 224, (100, 128), dtype=np.uint8)
    query = train[::20].copy()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    uint8_matches = matcher.knnMatch(query, train, k=2)
    float_matches = matcher.knnMatch(
        query.astype(np.float32),
        train.astype(np.float32),
        k=2,
    )

    assert [match[0].trainIdx for match in uint8_matches] == [
        match[0].trainIdx for match in float_matches
    ]
    assert [match[0].distance for match in uint8_matches] == pytest.approx(
        [match[0].distance for match in float_matches]
    )


def test_feature_index_builds_typed_contiguous_coarse_template_pyramid(
    tmp_path: Path,
) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0, shape=(120, 160))
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        tile_sizes=(40, 80, 120, 200),
        coarse_template_edge=16,
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    expected_dtypes = (np.uint8, np.int64, np.int32, np.int32, np.int32, np.int32)
    built_arrays = (
        built.coarse_templates.pixels,
        built.coarse_templates.offsets,
        built.coarse_templates.widths,
        built.coarse_templates.heights,
        built.coarse_templates.image_indices,
        built.coarse_templates.region_sizes,
    )
    loaded_arrays = (
        loaded.coarse_templates.pixels,
        loaded.coarse_templates.offsets,
        loaded.coarse_templates.widths,
        loaded.coarse_templates.heights,
        loaded.coarse_templates.image_indices,
        loaded.coarse_templates.region_sizes,
    )
    assert built.coarse_templates.region_sizes.tolist() == [40, 60, 80, 100, 120]
    assert built.coarse_templates.widths.tolist() == [64, 43, 32, 26, 21]
    assert built.coarse_templates.heights.tolist() == [48, 32, 24, 19, 16]
    assert built.coarse_templates.image_indices.tolist() == [0, 0, 0, 0, 0]
    assert built.coarse_templates.offsets.tolist() == [0, 3072, 4448, 5216, 5710, 6046]
    for built_array, loaded_array, dtype in zip(
        built_arrays, loaded_arrays, expected_dtypes, strict=True
    ):
        assert built_array.dtype == dtype
        assert built_array.flags.c_contiguous
        assert built_array.flags.writeable is False
        assert np.array_equal(loaded_array, built_array)
        assert loaded_array.flags.c_contiguous
        assert loaded_array.flags.writeable is False


def test_coarse_template_pixels_are_grayscale_level_images(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_blank_image(gallery / "one" / "base.png", shape=(40, 60))
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        tile_sizes=(20,),
        coarse_template_edge=10,
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    index = FeatureIndex.load_or_build(catalog, settings)

    assert index.coarse_templates.widths.tolist() == [30]
    assert index.coarse_templates.heights.tolist() == [20]
    assert index.coarse_templates.offsets.tolist() == [0, 600]
    assert index.coarse_templates.pixels.shape == (600,)
    assert np.all(index.coarse_templates.pixels == 0)


def test_image_smaller_than_every_region_size_gets_short_edge_fallback_level(
    tmp_path: Path,
) -> None:
    gallery = tmp_path / "songs"
    write_blank_image(gallery / "small" / "base.png")
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        tile_sizes=(64,),
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.coarse_templates.region_sizes.tolist() == [40]
    assert built.coarse_templates.image_indices.tolist() == [0]
    assert built.coarse_templates.widths.tolist() == [24]
    assert built.coarse_templates.heights.tolist() == [16]
    assert built.coarse_templates.offsets.tolist() == [0, 384]
    assert np.array_equal(loaded.coarse_templates.pixels, built.coarse_templates.pixels)
    assert loaded.loaded_from_cache is True


def test_region_sizes_keep_fitting_midpoint_when_larger_configured_size_does_not_fit() -> None:
    assert FeatureIndex._region_sizes(240, (192, 256)) == [192, 224]


def test_cache_without_coarse_level_for_every_image_is_rebuilt(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    write_textured_image(gallery / "two" / "base.jpg", 10)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache", tile_sizes=(64,))
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(catalog, settings)
    directory = feature_dir(settings)
    image_indices = np.load(directory / "coarse_image_indices.npy", allow_pickle=False)
    level_count = int(np.count_nonzero(image_indices == 0))
    offsets = np.load(directory / "coarse_offsets.npy", allow_pickle=False)
    for name in ("coarse_widths", "coarse_heights", "coarse_image_indices", "coarse_region_sizes"):
        array = np.load(directory / f"{name}.npy", allow_pickle=False)
        np.save(directory / f"{name}.npy", array[:level_count], allow_pickle=False)
    pixels = np.load(directory / "coarse_pixels.npy", allow_pickle=False)
    np.save(directory / "coarse_pixels.npy", pixels[: offsets[level_count]], allow_pickle=False)
    np.save(directory / "coarse_offsets.npy", offsets[: level_count + 1], allow_pickle=False)

    rebuilt = FeatureIndex.load_or_build(catalog, settings)

    assert rebuilt.loaded_from_cache is False
    assert set(rebuilt.coarse_templates.image_indices.tolist()) == {0, 1}


def test_manifest_change_invalidates_cache(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    path = gallery / "one" / "base.jpg"
    write_textured_image(path, 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    first_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(first_catalog, settings)
    first_directory = feature_dir(settings)
    write_textured_image(path, 20)
    second_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    rebuilt = FeatureIndex.load_or_build(second_catalog, settings)

    assert rebuilt.loaded_from_cache is False
    assert feature_dir(settings) != first_directory


@pytest.mark.parametrize(
    ("setting_name", "changed_value"),
    [
        ("working_max_edge", 96),
        ("sift_features", 25),
        ("sift_contrast_threshold", 0.08),
        ("tile_sizes", (64,)),
        ("coarse_template_edge", 8),
    ],
)
def test_feature_setting_change_invalidates_cache(
    tmp_path: Path, setting_name: str, changed_value: float | tuple[int, ...]
) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(catalog, settings)

    changed_settings = replace(settings, **{setting_name: changed_value})
    rebuilt = FeatureIndex.load_or_build(catalog, changed_settings)

    assert rebuilt.loaded_from_cache is False
    metadata = json.loads((settings.cache_dir / "manifest.json").read_text("utf-8"))
    expected_value = list(changed_value) if isinstance(changed_value, tuple) else changed_value
    assert metadata["feature_settings"][setting_name] == expected_value


def test_malformed_manifest_is_rebuilt_and_replaced(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(catalog, settings)
    manifest_path = settings.cache_dir / "manifest.json"
    manifest_path.write_text("{not-json", "utf-8")

    rebuilt = FeatureIndex.load_or_build(catalog, settings)

    assert rebuilt.loaded_from_cache is False
    assert json.loads(manifest_path.read_text("utf-8"))["images"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_array",
        "bad_offsets",
        "bad_coarse_owner",
        "bad_coarse_dtype",
        "bad_representative_owner",
        "wrong_image_ids",
        "wrong_cache_identity",
    ],
)
def test_corrupt_feature_cache_is_rebuilt(tmp_path: Path, corruption: str) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    write_textured_image(gallery / "two" / "base.jpg", 10)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(catalog, settings)
    directory = feature_dir(settings)

    if corruption == "missing_array":
        (directory / "working_scales.npy").unlink()
    elif corruption == "bad_offsets":
        replace_array(settings, "point_offsets", lambda array: array.__setitem__(-1, array[-1] + 1))
    elif corruption == "bad_coarse_owner":
        replace_array(
            settings,
            "coarse_image_indices",
            lambda array: array.__setitem__(0, len(catalog.records)),
        )
    elif corruption == "bad_coarse_dtype":
        replace_array(
            settings,
            "coarse_pixels",
            np.load(directory / "coarse_pixels.npy", allow_pickle=False).astype(np.float32),
        )
    elif corruption == "bad_representative_owner":
        replace_array(
            settings,
            "representative_image_indices",
            lambda array: array.__setitem__(0, len(catalog.records)),
        )
    elif corruption == "wrong_image_ids":
        replace_array(settings, "image_ids", lambda array: array.__setitem__(slice(None), "wrong"))
    else:
        replace_array(settings, "cache_identity", np.asarray("wrong-cache-identity"))

    rebuilt = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert rebuilt.loaded_from_cache is False
    assert loaded.loaded_from_cache is True


def test_failed_manifest_save_preserves_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    original_index = FeatureIndex.load_or_build(catalog, settings)
    manifest_path = settings.cache_dir / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    original_replace = Path.replace

    def fail_manifest_replace(source: Path, target: Path) -> Path:
        if source.name.endswith(".tmp.json"):
            raise RuntimeError("manifest replace failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)
    with pytest.raises(RuntimeError, match="manifest replace failed"):
        FeatureIndex.load_or_build(catalog, replace(settings, working_max_edge=96))

    assert manifest_path.read_bytes() == original_manifest
    monkeypatch.undo()
    rebuilt = FeatureIndex.load_or_build(catalog, settings)
    assert rebuilt.loaded_from_cache is True
    assert np.array_equal(rebuilt.descriptors, original_index.descriptors)


def test_preserves_per_image_features_and_descriptor_offsets(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_blank_image(gallery / "a" / "blank.png")
    write_textured_image(gallery / "b" / "texture.jpg", 4)
    settings = Settings(
        gallery_dir=gallery,
        cache_dir=tmp_path / "cache",
        working_max_edge=80,
    )
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.image_ids == tuple(record.image_id for record in catalog.records)
    assert built.descriptor_offsets.dtype == np.int64
    for image_index, image_id in enumerate(built.image_ids):
        features = built.by_image[image_id]
        cached = loaded.by_image[image_id]
        assert features.points.dtype == np.float32
        assert features.points.shape == (len(features.descriptors), 2)
        assert features.descriptors.dtype == np.uint8
        assert features.descriptors.shape[1:] == (128,)
        assert np.array_equal(cached.points, features.points)
        assert np.array_equal(cached.descriptors, features.descriptors)
        assert cached.working_width == features.working_width
        assert cached.working_height == features.working_height
        assert cached.working_scale == pytest.approx(features.working_scale)
        start, end = built.descriptor_offsets[image_index : image_index + 2]
        assert np.array_equal(built.descriptors[start:end], features.descriptors)

    blank = built.by_image[built.image_ids[0]]
    assert blank.points.shape == (0, 2)
    assert blank.descriptors.shape == (0, 128)


def test_cache_contains_only_non_object_npy_arrays(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    FeatureIndex.load_or_build(catalog, settings)
    directory = feature_dir(settings)
    files = list(directory.glob("*.npy"))

    assert files
    assert all(np.load(path, allow_pickle=False).dtype != np.dtype("O") for path in files)
    assert np.load(directory / "descriptors.npy", mmap_mode="r").dtype == np.uint8
    assert isinstance(np.load(directory / "descriptors.npy", mmap_mode="r"), np.memmap)


def test_empty_catalog_round_trips_typed_empty_arrays(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.image_ids == ()
    assert built.by_image == {}
    assert built.descriptors.shape == (0, 128)
    assert built.descriptors.dtype == np.uint8
    assert built.descriptor_offsets.tolist() == [0]
    assert built.representative_descriptors.shape == (0, 128)
    assert built.representative_image_indices.shape == (0,)
    assert built.coarse_templates.pixels.dtype == np.uint8
    assert loaded.loaded_from_cache is True


def test_failed_save_does_not_replace_existing_feature_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = tmp_path / "songs"
    image_path = gallery / "one" / "base.jpg"
    write_textured_image(image_path, 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(catalog, settings)
    original_manifest = (settings.cache_dir / "manifest.json").read_bytes()
    original_directory = feature_dir(settings)
    write_textured_image(image_path, 20)
    changed_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    original_save = np.save

    def fail_save(path: Path, array: np.ndarray, **kwargs: Any) -> None:
        original_save(path, array, **kwargs)
        raise RuntimeError("save failed")

    monkeypatch.setattr(np, "save", fail_save)
    with pytest.raises(RuntimeError, match="save failed"):
        FeatureIndex.load_or_build(changed_catalog, settings)

    assert (settings.cache_dir / "manifest.json").read_bytes() == original_manifest
    assert original_directory.exists()
    assert not list(settings.cache_dir.glob("*.tmp-*"))
