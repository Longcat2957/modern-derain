import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modern_derain.artifacts import write_json
from modern_derain.cli import app
from modern_derain.config import DEFAULTS, load_config, resolve_split

runner = CliRunner()


def test_default_file_matches_code():
    assert json.loads(Path("configs/default.json").read_text()) == DEFAULTS


def test_merge_paths_and_explicit_false_zero(tmp_path):
    config_file = tmp_path / "experiment.json"
    config_file.write_text(
        json.dumps(
            {
                "training": {"amp": True, "checkpoint": "custom.pt"},
                "data": {"path": "dataset", "test": str(tmp_path)},
            }
        )
    )
    config = load_config(
        config_file, {"training": {"amp": False, "lr": None}, "quantization": {"steps": 0}}
    )
    assert config["training"]["amp"] is False
    assert config["training"]["lr"] == DEFAULTS["training"]["lr"]
    assert config["quantization"]["steps"] == 0
    assert config["export"]["checkpoint"] == "custom.pt"
    assert resolve_split(config, "train") == Path("dataset/train")
    assert resolve_split(config, "test") == tmp_path
    assert (
        load_config(config_file, {"training": {"resume": "epoch.pt"}})["training"]["checkpoint"]
        is None
    )
    assert (
        load_config(overrides={"training": {"resume": "epoch.pt"}})["training"]["checkpoint"]
        is None
    )


@pytest.mark.parametrize(
    "override",
    [
        {"data": {"manifest": "list.json"}},
        {"data": {"layout": "auto"}},
        {"data": {"path": " "}},
        {"training": {"amp": 1}},
        {"training": {"epochs": True}},
        {"model": {"color": 1}},
        {"model": {"kernel_size": [3.0]}},
        {"inference": {"checkpoint": None}},
        {"training": {"checkpoint": None}},
        {"export": {"shape": [2, 3, 16, 16]}},
        {"export": {"shape": [1, 3, 17, 16]}},
        {"training": {"checkpoint": "a.pt", "resume": "b.pt"}},
        {"quantization": {"steps": -1}},
        {"benchmark": {"target_fps": 0}},
    ],
)
def test_invalid_config(override, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError):
        load_config(path)


def test_help_never_imports_optional_packages():
    program = """
import sys
from typer.testing import CliRunner
from modern_derain.cli import app
for command in ([], ['train'], ['qat'], ['export'], ['infer'], ['eval']):
    result = CliRunner().invoke(app, command + ['--help'])
    assert result.exit_code == 0, result.output
assert not any(name.split('.')[0] in {'torch','modelopt','cv2','albumentations','skimage','onnx'} for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_module_entrypoint_help_never_imports_optional_packages():
    program = """
import runpy
import sys
sys.argv = ['modern_derain', '--help']
try:
    runpy.run_module('modern_derain', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not any(name.split('.')[0] in {'torch','modelopt','cv2','albumentations','skimage','onnx'} for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout


@pytest.mark.parametrize(
    ("command", "module", "function"),
    [
        ("train", "modern_derain.training.loop", "run_train"),
        ("qat", "modern_derain.training.quantize", "run_qat"),
    ],
)
def test_nonprimary_worker_has_no_result_stdout(monkeypatch, command, module, function):
    fake = types.ModuleType(module)

    def run_worker(config):
        print("worker log")
        return None

    setattr(fake, function, run_worker)
    monkeypatch.setitem(sys.modules, module, fake)
    result = runner.invoke(app, [command])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "worker log" in result.stderr


def test_cli_override_and_json_streams(tmp_path, monkeypatch):
    fake = types.ModuleType("modern_derain.training.quantize")

    def run_qat(config):
        print("SDK log")
        return {"steps": config["quantization"]["steps"], "amp": config["training"]["amp"]}

    fake.run_qat = run_qat
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config = tmp_path / "train.json"
    config.write_text('{"training":{"amp":true},"quantization":{"steps":10}}')
    result = runner.invoke(
        app, ["qat", "--config", str(config), "--data", str(tmp_path), "--steps", "0", "--no-amp"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"steps": 0, "amp": False}
    assert "SDK log" in result.stderr


def test_cli_rejects_manifest_missing_path_and_shape():
    for args in (
        ["train", "--manifest", "data.json"],
        ["train", "--data", "/nonexistent-data-modern-derain"],
        ["export", "--height", "16"],
        ["train", "--checkpoint", "a.pt", "--resume", "b.pt"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code != 0
        assert not result.stdout.lstrip().startswith("{")


def test_json_output_never_overwrites(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_json(invalid, {"nan": float("nan")})
    assert not invalid.exists()


def test_quantized_eval_missing_dependency_keeps_metrics_hint(monkeypatch):
    fake = types.ModuleType("modern_derain.evaluation.quality")

    def run_eval(*args):
        raise ModuleNotFoundError("ModelOpt unavailable", name="modelopt")

    fake.run_eval = run_eval
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "--extra qat --extra metrics" in result.stderr
