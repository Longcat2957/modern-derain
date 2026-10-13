import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from modern_derain.cli import app
from modern_derain.config import DEFAULT_PREPROCESSING, DEFAULTS
from modern_derain.data.sources import scan_pairs, snapshot, training_snapshots
from modern_derain.evaluation import quality
from modern_derain.models import kpn
from modern_derain.training.loop import save_checkpoint, train_model
from modern_derain.training.quantize import run_qat


def test_real_checkpoint_infer_eval_cli_and_overwrite_protection(tmp_path):
    checkpoint = Path(__file__).resolve().parents[1] / "weights/efficientderain-rain1400-v3.pt"
    if not checkpoint.exists():
        pytest.skip("verified default weight is a separately acquired Release asset")
    torch.set_num_threads(1)
    image = np.random.default_rng(1).integers(0, 256, (17, 19, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    assert cv2.imwrite(str(input_path), image)
    runner = CliRunner()
    args = [
        "infer",
        "--input",
        str(input_path),
        "--output",
        str(tmp_path / "restored.png"),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["count"] == 1
    restored = cv2.imread(str(tmp_path / "restored.png"))
    assert restored.shape == image.shape
    metadata = json.loads((tmp_path / "restored.json").read_text())
    assert metadata["images"][0]["input_shape"] == [1, 3, 32, 32]
    assert metadata["checkpoint_sha256"]
    assert runner.invoke(app, args).exit_code == 1
    data_root = tmp_path / "dataset"
    for folder in ("rainy", "clean"):
        path = data_root / "val" / folder
        path.mkdir(parents=True)
        assert cv2.imwrite(str(path / "sample.png"), image)
    output = tmp_path / "eval.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--data",
            str(data_root),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["count"] == 1 and np.isfinite(summary["psnr"])
    evaluation = json.loads(output.read_text())
    assert evaluation["samples"][0]["id"] == "sample.png"
    assert evaluation["data_snapshot"][0]["clean"]["size"] > 0
    assert evaluation["metric_protocol"]["ssim_dtype"] == "float64"


def test_explicit_fixed_protocol_keeps_checkpoint_contract(monkeypatch):
    config = deepcopy(DEFAULTS)
    config["export"]["shape"] = [1, 3, 16, 32]
    config["inference"]["protocol"] = "fixed"
    payload = {"kind": "fp32"}

    def load(path, model, preprocessing, device):
        assert preprocessing == DEFAULT_PREPROCESSING
        return object(), payload

    monkeypatch.setattr(quality, "load_checkpoint", load)
    quality._load_model(config, config["inference"], Path("weight.pt"), "cpu")
    payload.update({"kind": "ptq", "calibration": {"shape": [1, 3, 16, 32]}})
    quality._load_model(config, config["inference"], Path("weight.pt"), "cpu")
    config["export"]["shape"] = [1, 3, 32, 32]
    with pytest.raises(ValueError, match="calibrated"):
        quality._load_model(config, config["inference"], Path("weight.pt"), "cpu")


def test_finetune_preserves_training_history_and_rejects_old_source_eval(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = deepcopy(DEFAULTS)
    config["training"].update(device="cpu", epochs=1, batch_size=1, crop_size=16)
    config["evaluation"].update(device="cpu", output=str(tmp_path / "eval.json"))
    model = torch.nn.Conv2d(3, 3, 1)
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, {"state_dict": model.state_dict()})
    config["training"]["checkpoint"] = str(initial)
    payload = {}
    for source in ("A", "B", "B"):
        data_root = tmp_path / source
        for side in ("rainy", "clean"):
            folder = data_root / "train" / side
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / f"{source}.png"
            if not image_path.exists():
                assert cv2.imwrite(str(image_path), np.full((16, 16, 3), 90, np.uint8))
        config["data"]["path"] = str(data_root)
        config["training"]["output_dir"] = str(
            tmp_path / f"run-{source}-{len(list(tmp_path.glob('run-*')))}"
        )
        result = train_model(model, config, payload)
        config["training"]["checkpoint"] = result["checkpoint"]
        payload = torch.load(result["checkpoint"], weights_only=True)
    assert {record["source_id"] for record in payload["data_snapshot"]} == {"B"}
    assert len(payload["inherited_training_snapshots"]) == 1
    assert {record["source_id"] for saved in training_snapshots(payload) for record in saved} == {
        "A",
        "B",
    }
    for side in ("rainy", "clean"):
        folder = tmp_path / "B" / "val" / side
        folder.mkdir(parents=True)
        assert cv2.imwrite(str(folder / "A.png"), np.full((16, 16, 3), 90, np.uint8))
    monkeypatch.setattr(quality, "_load_model", lambda *args: (model, payload))
    with pytest.raises(ValueError, match="leakage"):
        quality.run_eval(config, Path(result["checkpoint"]))
    assert not (tmp_path / "eval.json").exists()
    calibration = tmp_path / "B" / "calibration"
    calibration.mkdir()
    assert cv2.imwrite(str(calibration / "A.png"), np.full((16, 16, 3), 90, np.uint8))
    monkeypatch.setattr(kpn, "load_checkpoint", lambda *args, **kwargs: (model, payload))
    config["export"]["shape"] = [1, 3, 16, 16]
    config["training"]["output_dir"] = str(tmp_path / "blocked-ptq")
    with pytest.raises(ValueError, match="leakage"):
        run_qat(config)
    assert not (tmp_path / "blocked-ptq").exists()
    (calibration / "A.png").rename(calibration / "D.png")
    with pytest.raises(ValueError, match="leakage"):
        run_qat(config)
    heldout = snapshot(scan_pairs(tmp_path / "B" / "val"))
    stale = deepcopy(heldout)
    stale[0]["source_id"] = "C"
    stale[0]["rainy"]["path"] = "unrelated-old-val-C.png"
    stale[0]["clean"]["path"] = "unrelated-old-val-C-clean.png"
    payload["heldout_snapshots"] = {"val": stale}
    config["training"]["output_dir"] = str(tmp_path / "blocked-train")
    with pytest.raises(ValueError, match="leakage"):
        train_model(model, config, payload)
    assert not (tmp_path / "blocked-train").exists()
