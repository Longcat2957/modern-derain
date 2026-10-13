import ast
import hashlib
import json
import random
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from modern_derain.data import sources
from modern_derain.data.images import prepare_rgb, read_rgb, recover, write_recovered
from modern_derain.data.paired import PairedDataset
from modern_derain.data.rain1400_identity import _RAINY_RECORDS, GROUND_TRUTH, rainy_identity


def put_image(path, image=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None:
        image = np.zeros((32, 48, 3), np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def paired_folder(root, names=("sequence/a.png", "sequence/b.png")):
    for name in names:
        put_image(root / "rainy" / name)
        put_image(root / "clean" / name)


def test_scan_sorted_once_no_decode_and_snapshot(tmp_path, monkeypatch):
    paired_folder(tmp_path)
    monkeypatch.setattr(cv2, "imread", lambda *a: pytest.fail("scan must not decode images"))
    samples = sources.scan_pairs(tmp_path)
    assert [s.id for s in samples] == ["sequence/a.png", "sequence/b.png"]
    assert {s.source_id for s in samples} == {"sequence"}
    records = sources.snapshot(samples)
    assert records[0]["rainy"]["size"] == samples[0].rainy.stat().st_size
    assert records[0]["clean"]["path"] == str(samples[0].clean.resolve())
    with pytest.raises(ValueError, match="leakage"):
        sources.check_leakage(records, records)
    sources.check_leakage(
        records, [{"source_id": "other", "rainy": {"path": "other"}, "clean": None}]
    )


def test_missing_ambiguous_duplicate_and_unmatched(tmp_path):
    paired_folder(tmp_path, ("a.png",))
    (tmp_path / "clean/a.png").unlink()
    with pytest.raises(ValueError, match="No images"):
        sources.scan_pairs(tmp_path)
    put_image(tmp_path / "clean/b.png")
    with pytest.raises(ValueError, match="Missing clean"):
        sources.scan_pairs(tmp_path)
    put_image(tmp_path / "clean/a.png")
    with pytest.raises(ValueError, match="unmatched"):
        sources.scan_pairs(tmp_path)
    (tmp_path / "clean/b.png").unlink()
    (tmp_path / "rainy/duplicate.png").symlink_to(tmp_path / "rainy/a.png")
    with pytest.raises(ValueError, match="Duplicate"):
        sources.scan_pairs(tmp_path)
    (tmp_path / "rainy/duplicate.png").unlink()
    (tmp_path / "rainy_image").mkdir()
    with pytest.raises(ValueError, match="Ambiguous"):
        sources.scan_pairs(tmp_path)


def test_verified_rain1400_filename_and_shared_gt(tmp_path, monkeypatch):
    gt = put_image(tmp_path / "ground_truth/1.jpg")
    data = gt.read_bytes()
    monkeypatch.setitem(sources.GROUND_TRUTH, 1, (len(data), zlib.crc32(data)))
    put_image(tmp_path / "rainy_image/1_1.jpg")
    put_image(tmp_path / "rainy_image/1_14.jpg")
    samples = sources.scan_pairs(tmp_path, "rain1400")
    assert len(samples) == 2 and samples[0].clean == samples[1].clean
    assert samples[0].source_id == "fu-cvpr2017-rain1400:1"
    put_image(tmp_path / "rainy_image/1_01.jpg")
    with pytest.raises(ValueError, match="Unverified"):
        sources.scan_pairs(tmp_path, "rain1400")
    (tmp_path / "rainy_image/1_01.jpg").unlink()
    gt.write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="Unverified"):
        sources.scan_pairs(tmp_path, "rain1400")


def test_rain1400_observed_archive_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/rain1400.json").read_text())
    assert fixture["archive_bytes"] == 898292342
    assert fixture["training"]["rainy_count"] == 900 * 14
    assert fixture["testing"]["rainy_count"] == 100 * 14
    assert sorted(GROUND_TRUTH) == list(range(1, 1001))
    assert GROUND_TRUTH[1000] == (31556, 1179166386)
    assert rainy_identity(1000, 1) == (33502, 276213994)
    assert hashlib.sha256(_RAINY_RECORDS).hexdigest() == fixture["rainy_identity_sha256"]
    gt_records = b"".join(struct.pack("<II", *GROUND_TRUTH[i]) for i in range(1, 1001))
    assert hashlib.sha256(gt_records).hexdigest() == fixture["ground_truth_identity_sha256"]


def test_verified_rain1400_source_roles_preserve_final_test():
    train = [sources.Sample("1_1.jpg", Path("rainy"), None, "fu-cvpr2017-rain1400:1")]
    test = [{"source_id": "fu-cvpr2017-rain1400:901"}]
    for split in ("train", "val", "calibration"):
        sources.validate_split_sources(train, split)
        with pytest.raises(ValueError, match="final test"):
            sources.validate_split_sources(test, split)
    sources.validate_split_sources(test, "test")
    with pytest.raises(ValueError, match="final test"):
        sources.validate_split_sources(train, "test")
    sources.validate_split_sources([{"source_id": "drone-flight-1"}], "calibration")


def test_calibration_consumes_rainy_only(tmp_path, monkeypatch):
    paired_folder(tmp_path, ("a.png",))
    monkeypatch.setattr(cv2, "imread", lambda *a: pytest.fail("scan must not decode images"))
    samples = sources.scan_calibration(tmp_path)
    assert len(samples) == 1 and samples[0].clean is None
    assert samples[0].rainy.parent.name == "rainy"
    assert sources.scan_calibration(tmp_path, "rain1400") == samples


def test_verified_rain1400_rainy_only_and_generic_drone_calibration(tmp_path, monkeypatch):
    original = tmp_path / "official"
    rainy = put_image(original / "rainy_image/1_1.jpg")
    data = rainy.read_bytes()
    monkeypatch.setattr(
        sources, "rainy_identity", lambda clean, rain: (len(data), zlib.crc32(data))
    )
    (original / "ground_truth").mkdir()
    (original / "ground_truth/1.jpg").write_bytes(b"must not be consumed")
    samples = sources.scan_calibration(original, "rain1400")
    assert samples[0].source_id == "fu-cvpr2017-rain1400:1" and samples[0].clean is None
    (original / "ground_truth/1.jpg").unlink()
    (original / "ground_truth").rmdir()
    assert (
        sources.scan_calibration(original / "rainy_image", "rain1400")[0].source_id
        == samples[0].source_id
    )
    rainy.write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="Unverified"):
        sources.scan_calibration(original, "rain1400")
    rainy.write_bytes(data)
    put_image(original / "rainy_image/901_1.jpg")
    with pytest.raises(ValueError, match="final test"):
        sources.scan_calibration(original, "rain1400")
    drone = tmp_path / "drone"
    put_image(drone / "rainy/flight-2026-01/a.png")
    generic = sources.scan_calibration(drone, "rain1400")
    assert generic[0].source_id == "flight-2026-01" and generic[0].clean is None
    assert sources.scan_calibration(drone / "rainy", "rain1400") == generic


def test_rgb_resize_float_range_and_invalid_decode(tmp_path):
    rgb = np.full((17, 31, 3), (10, 20, 30), np.uint8)
    path = put_image(tmp_path / "test.png", rgb[..., ::-1])
    np.testing.assert_array_equal(read_rgb(path), rgb)
    chw, size = prepare_rgb(rgb)
    assert size == (17, 31) and chw.shape == (3, 32, 32)
    assert chw.dtype == np.float32 and chw.flags.c_contiguous
    np.testing.assert_array_equal(chw[:, 0, 0], np.array([10, 20, 30], np.float32) / 255)
    fixed, _ = prepare_rgb(rgb, "fixed", [1, 3, 16, 48])
    assert fixed.shape == (3, 16, 48)
    with pytest.raises(ValueError, match="shape"):
        prepare_rgb(rgb, "fixed", None)
    put_image(tmp_path / "gray.png", np.zeros((16, 16), np.uint8))
    with pytest.raises(ValueError, match="three-channel"):
        read_rgb(tmp_path / "gray.png")
    with pytest.raises(ValueError, match="decode"):
        read_rgb(tmp_path / "absent.png")


def test_recover_matches_upstream_truncation_float_resize(tmp_path):
    oracle_path = (
        Path(__file__).parent
        / "reference/EfficientDeRain/465759da2428a4263eeee6ab464a0135fca2900c/utils.py"
    )
    function = next(
        n
        for n in ast.parse(oracle_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "recover_process"
    )
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(oracle_path), "exec"), namespace)
    chw = np.linspace(-0.2, 1.2, 3 * 16 * 32, dtype=np.float32).reshape(3, 16, 32)
    actual = recover(chw, (19, 21))
    expected = namespace["recover_process"](torch.from_numpy(chw)[None], 19, 21)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    path = tmp_path / "restore.png"
    write_recovered(path, actual)
    with pytest.raises(ValueError, match="overwrite"):
        write_recovered(path, actual)


def test_crop_alignment_inclusive_coordinates_and_exact_rng_resume(tmp_path):
    rgb = np.stack(np.meshgrid(np.arange(48), np.arange(32)), axis=-1)
    rgb = np.concatenate([rgb, np.full((32, 48, 1), 200)], axis=-1).astype(np.uint8)
    paired_folder(tmp_path, ("a.png",))
    for folder in ("rainy", "clean"):
        put_image(tmp_path / folder / "a.png", rgb[..., ::-1])
    dataset = PairedDataset(sources.scan_pairs(tmp_path), crop_size=16, seed=5)
    rainy, clean = dataset[0]
    assert torch.equal(rainy, clean) and rainy.is_contiguous()
    generator = random.Random(5)
    top, left = generator.randint(0, 16), generator.randint(0, 32)
    np.testing.assert_array_equal(rainy[:, 0, 0].numpy(), rgb[top, left].astype(np.float32) / 255)
    state = dataset.get_augmentation_state()
    expected = dataset[0][0]
    dataset.restore_augmentation_state(state)
    assert torch.equal(dataset[0][0], expected)
    put_image(tmp_path / "clean/a.png", np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="shapes differ"):
        dataset[0]
