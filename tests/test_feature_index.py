from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from crop_matcher.catalog import ImageCatalog
from crop_matcher.config import Settings
from crop_matcher.feature_index import FeatureIndex


def write_textured_image(path: Path, offset: int) -> None:
    image = np.zeros((160, 160, 3), np.uint8)
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


def write_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".png", np.zeros((40, 60, 3), np.uint8))
    assert ok
    path.write_bytes(payload.tobytes())


def test_builds_global_descriptors_and_round_trips_cache(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    write_textured_image(gallery / "two" / "base.jpg", 10)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.descriptors.dtype == np.float32
    assert built.descriptors.shape[0] > 0
    assert set(built.by_image) == {record.image_id for record in catalog.records}
    assert len(built.global_matcher.getTrainDescriptors()) == 1
    assert np.array_equal(built.global_matcher.getTrainDescriptors()[0], built.descriptors)
    assert loaded.loaded_from_cache is True
    assert np.array_equal(loaded.descriptors, built.descriptors)
    assert np.array_equal(loaded.descriptor_image_indices, built.descriptor_image_indices)


def test_manifest_change_invalidates_cache(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    path = gallery / "one" / "base.jpg"
    write_textured_image(path, 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    first_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(first_catalog, settings)
    write_textured_image(path, 20)
    second_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    rebuilt = FeatureIndex.load_or_build(second_catalog, settings)
    assert rebuilt.loaded_from_cache is False


@pytest.mark.parametrize(
    ("setting_name", "changed_value"),
    [
        ("working_max_edge", 96),
        ("sift_features", 25),
        ("sift_contrast_threshold", 0.08),
    ],
)
def test_feature_setting_change_invalidates_cache(
    tmp_path: Path, setting_name: str, changed_value: int | float
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
    assert isinstance(metadata["schema_version"], int)
    assert metadata["feature_settings"][setting_name] == changed_value


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
        "malformed_npz",
        "malformed_zip",
        "missing_array",
        "bad_offsets",
        "bad_descriptor_owner",
        "wrong_image_ids",
        "wrong_cache_identity",
    ],
)
def test_corrupt_feature_cache_is_rebuilt(tmp_path: Path, corruption: str) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)
    FeatureIndex.load_or_build(catalog, settings)
    cache_path = settings.cache_dir / "features.npz"

    if corruption == "malformed_npz":
        cache_path.write_bytes(b"not-an-npz")
    elif corruption == "malformed_zip":
        cache_path.write_bytes(b"PK\x03\x04broken")
    else:
        with np.load(cache_path, allow_pickle=False) as cache:
            arrays = {name: cache[name] for name in cache.files}
        if corruption == "missing_array":
            del arrays["working_scales"]
        elif corruption == "bad_offsets":
            arrays["point_offsets"][-1] += 1
        elif corruption == "bad_descriptor_owner":
            arrays["descriptor_image_indices"][:] = len(catalog.records)
        elif corruption == "wrong_image_ids":
            arrays["image_ids"][:] = "wrong-image-id"
        else:
            arrays["cache_identity"] = np.asarray("wrong-cache-identity")
        np.savez(cache_path, **arrays)

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
    assert list(settings.cache_dir.glob("*.tmp.json")) == []

    monkeypatch.undo()
    rebuilt = FeatureIndex.load_or_build(catalog, settings)

    assert rebuilt.loaded_from_cache is False
    assert np.array_equal(rebuilt.descriptors, original_index.descriptors)


def test_preserves_per_image_features_and_descriptor_mapping(tmp_path: Path) -> None:
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
    assert built.descriptor_image_indices.dtype == np.int32
    assert built.descriptor_image_indices.shape == (len(built.descriptors),)
    for image_index, image_id in enumerate(built.image_ids):
        features = built.by_image[image_id]
        cached = loaded.by_image[image_id]
        assert features.points.dtype == np.float32
        assert features.points.shape == (len(features.descriptors), 2)
        assert features.descriptors.dtype == np.float32
        assert features.descriptors.shape[1:] == (128,)
        assert features.points.flags.c_contiguous
        assert features.descriptors.flags.c_contiguous
        assert np.array_equal(cached.points, features.points)
        assert np.array_equal(cached.descriptors, features.descriptors)
        assert cached.working_width == features.working_width
        assert cached.working_height == features.working_height
        assert cached.working_scale == pytest.approx(features.working_scale)
        descriptor_slice = built.descriptor_image_indices == image_index
        assert np.array_equal(built.descriptors[descriptor_slice], features.descriptors)

    blank = built.by_image[built.image_ids[0]]
    assert blank.points.shape == (0, 2)
    assert blank.descriptors.shape == (0, 128)
    assert (blank.working_width, blank.working_height, blank.working_scale) == (60, 40, 1.0)


def test_cache_contains_only_non_object_flattened_arrays(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    write_textured_image(gallery / "one" / "base.jpg", 0)
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    FeatureIndex.load_or_build(catalog, settings)

    with np.load(settings.cache_dir / "features.npz", allow_pickle=False) as cache:
        assert all(cache[name].dtype != np.dtype("O") for name in cache.files)
        assert cache["image_ids"].dtype.kind == "U"
        assert cache["point_offsets"].dtype == np.int64
        assert cache["descriptor_offsets"].dtype == np.int64
        assert cache["cache_identity"].dtype.kind == "U"
        assert cache["cache_identity"].shape == ()
        metadata = json.loads((settings.cache_dir / "manifest.json").read_text("utf-8"))
        assert cache["cache_identity"].item() == metadata["cache_identity"]


def test_empty_catalog_round_trips_typed_empty_arrays(tmp_path: Path) -> None:
    gallery = tmp_path / "songs"
    settings = Settings(gallery_dir=gallery, cache_dir=tmp_path / "cache")
    catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    built = FeatureIndex.load_or_build(catalog, settings)
    loaded = FeatureIndex.load_or_build(catalog, settings)

    assert built.image_ids == ()
    assert built.by_image == {}
    assert built.descriptors.shape == (0, 128)
    assert built.descriptors.dtype == np.float32
    assert built.descriptor_image_indices.shape == (0,)
    assert built.descriptor_image_indices.dtype == np.int32
    assert built.tiles.hashes.dtype == np.uint64
    assert built.tiles.image_indices.dtype == np.int32
    assert built.tiles.xs.dtype == np.int32
    assert built.tiles.ys.dtype == np.int32
    assert built.tiles.sizes.dtype == np.int32
    assert all(
        array.shape == (0,)
        for array in (
            loaded.tiles.hashes,
            loaded.tiles.image_indices,
            loaded.tiles.xs,
            loaded.tiles.ys,
            loaded.tiles.sizes,
        )
    )
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
    cache_path = settings.cache_dir / "features.npz"
    original_cache = cache_path.read_bytes()
    write_textured_image(image_path, 20)
    changed_catalog = ImageCatalog.scan(gallery, settings.max_image_pixels)

    def fail_save(path: Path, **_arrays: np.ndarray) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("save failed")

    monkeypatch.setattr(np, "savez", fail_save)

    with pytest.raises(RuntimeError, match="save failed"):
        FeatureIndex.load_or_build(changed_catalog, settings)

    assert cache_path.read_bytes() == original_cache
    assert list(settings.cache_dir.glob("*.tmp.npz")) == []
